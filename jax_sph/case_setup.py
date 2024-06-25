"""Simulation setup."""

import importlib
import os
import warnings
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from jax_sph.eos import RIEMANNEoS, TaitEoS
from jax_sph.io_state import read_h5
from jax_sph.jax_md import space
from jax_sph.utils import (
    Tag,
    compute_nws_jax_wrapper,
    compute_nws_scipy,
    get_noise_masked,
    pos_init_cartesian_2d,
    pos_init_cartesian_3d,
    wall_tags,
)

EPS = jnp.finfo(float).eps


class SimulationSetup(ABC):
    """Parent class for all simulation setups."""

    def __init__(self, cfg):
        # If a value is changed in one of these, it also changes in the others
        self.cfg = cfg
        self.case = cfg.case
        self.special = cfg.case.special

        if cfg.case.r0_type == "relaxed":
            assert (cfg.case.state0_path is not None) and ("r" in cfg.case.state0_keys)

        # get the config file name, e.g. "cases/db.yaml" -> "db"
        cfg.case.name = os.path.splitext(os.path.basename(cfg.config))[0]

    def initialize(self):
        """Initialize and return everything needed for the numerical setup.

        This function sets up all physical and numerical quantities including:

            - reference values and case specific parameters
            - integration time step
            - relaxation or simulation mode
            - equation of state
            - boundary conditions
            - state dictionary
            - function including external force and boundary conditions

        Returns: A tuple containing the following elements:

           - cfg (DictConfig): configuration arguments
           - box_size (ndarray): size of the simulation box starting at 0.0
           - state (dict): dictionary containing all field values
           - g_ext_fn (Callable): external force function
           - bc_fn (Callable): boundary conditions function (e.g. velocity at walls)
           - nw_fn (Callable): jit-able wall normal funct. when moving walls, else None
           - eos (Callable): equation of state function
           - key (PRNGKey): random key for sampling
           - displacement_fn (Callable): displacement function for edge features
           - shift_fn (Callable): shift function for position updates
        """

        cfg = self.cfg
        dx = cfg.case.dx
        dim = cfg.case.dim
        rho_ref = cfg.case.rho_ref
        viscosity = cfg.case.viscosity
        u_ref = cfg.case.u_ref
        cfl = cfg.solver.cfl

        key_prng = jax.random.PRNGKey(cfg.seed)

        # Primal: reference density, dynamic viscosity, and velocity
        # Derived: reference speed of sound, pressure
        c_ref = cfg.case.c_ref_factor * u_ref
        p_ref = rho_ref * c_ref**2 / cfg.eos.gamma
        # for free surface simulation p_background in the EoS has to be 0.0
        p_bg = cfg.eos.p_bg_factor * p_ref

        # calculate volume and mass
        h = dx
        volume_ref = h**dim
        mass_ref = volume_ref * rho_ref

        # time integration step dt
        dt_convective = cfl * h / (c_ref + u_ref)
        dt_viscous = cfl * h**2 * rho_ref / (viscosity + EPS)
        dt_body_force = cfl * (h / (cfg.case.g_ext_magnitude + EPS)) ** 0.5
        dt = np.amin([dt_convective, dt_viscous, dt_body_force]).item()
        # TODO: consider adaptive time step sizes

        print("dt_convective :", dt_convective)
        print("dt_viscous    :", dt_viscous)
        print("dt_body_force :", dt_body_force)
        print("dt_max_allowed:", dt)

        if cfg.solver.dt is not None:
            if cfg.solver.dt > dt:
                warnings.warn("Explicit dt should comply with CFL.", UserWarning)
            dt = cfg.solver.dt

        print("dt_final      :", dt)

        if cfg.case.mode == "rlx":
            # run a relaxation of randomly initialized state for 500 steps
            sequence_length = 5000
            cfg.solver.t_end = dt * sequence_length
            # turn background pressure on for homogeneous particle distribution
        else:
            sequence_length = int(cfg.solver.t_end / dt)

        # Equation of state
        if cfg.solver.name == "RIE":
            eos = RIEMANNEoS(rho_ref, p_bg, u_ref)
        else:
            eos = TaitEoS(p_ref, rho_ref, p_bg, cfg.eos.gamma)

        # initialize box and positions of particles
        if dim == 2:
            box_size = self._box_size2D(cfg.solver.n_walls)
            r, tag = self._init_pos2D(box_size, dx, cfg.solver.n_walls)
        elif dim == 3:
            box_size = self._box_size3D(cfg.solver.n_walls)
            r, tag = self._init_pos3D(box_size, dx, cfg.solver.n_walls)
        displacement_fn, shift_fn = space.periodic(side=box_size)

        num_particles = len(r)
        print("Total number of particles = ", num_particles)

        # add noise to the fluid particles to break symmetry
        key, subkey = jax.random.split(key_prng)
        if cfg.case.r0_noise_factor != 0.0:
            noise_std = cfg.case.r0_noise_factor * dx
            noise = get_noise_masked(r.shape, tag == Tag.FLUID, subkey, std=noise_std)
            # PBC: move all particles to the box limits after noise addition
            r = shift_fn(r, noise)

        # initialize the velocity given the coordinates r with the noise
        if dim == 2:
            v = vmap(self._init_velocity2D)(r)
        elif dim == 3:
            v = vmap(self._init_velocity3D)(r)

        # initialize all other field values
        rho, mass, eta, temperature, kappa, Cp = self._set_field_properties(
            num_particles, mass_ref, cfg.case
        )
        # whether to compute wall normals
        is_nw = (
            cfg.solver.free_slip or cfg.solver.name == "RIE"
        ) and cfg.solver.is_bc_trick
        # calculate wall normals if necessary
        nw = self._compute_wall_normals("scipy")(r, tag) if is_nw else jnp.zeros_like(r)

        # initialize the state dictionary
        state = {
            "r": r,
            "tag": tag,
            "u": v,
            "v": v,
            "dudt": jnp.zeros_like(v),
            "dvdt": jnp.zeros_like(v),
            "drhodt": jnp.zeros_like(rho),
            "rho": rho,
            "p": eos.p_fn(rho),
            "mass": mass,
            "eta": eta,
            "dTdt": jnp.zeros_like(rho),
            "T": temperature,
            "kappa": kappa,
            "Cp": Cp,
            "nw": nw,
        }

        # overwrite the state dictionary with the provided one, only for the fluid
        if cfg.case.state0_path is not None:
            _state = read_h5(cfg.case.state0_path)
            for k in state:
                if k not in cfg.case.state0_keys:
                    continue
                assert k in _state, ValueError(f"Key {k} not found in state0 file.")
                mask, _mask = state["tag"]==Tag.FLUID, _state["tag"]==Tag.FLUID
                assert state[k][mask].shape == _state[k][_mask].shape, ValueError(
                    f"Shape mismatch for key {k} in state0 file."
                )
                state[k][mask] = _state[k][_mask]

        # the following arguments are needed for dataset generation
        cfg.case.c_ref, cfg.case.p_ref, cfg.case.p_bg = c_ref, p_ref, p_bg
        cfg.solver.dt, cfg.solver.sequence_length = dt, sequence_length
        cfg.case.num_particles_max = num_particles
        cfg.case.pbc = [True, True, True]  # TODO: matscipy needs 3D
        cfg.case.bounds = np.array([np.zeros_like(box_size), box_size]).T.tolist()

        state = self._boundary_conditions_fn(state)

        g_ext_fn = self._external_acceleration_fn
        bc_fn = self._boundary_conditions_fn

        # whether to recompute the wall normals at every integration step
        is_nw_recompute = (tag == Tag.MOVING_WALL).any() and is_nw
        if is_nw_recompute:
            assert cfg.nl.backend != "matscipy", NotImplementedError(
                "Wall normals not yet implemented for matscipy neighbor list when "
                "working with moving boundaries. \nIf you work with moving boundaries, "
                "don't use one of: `nl.backend=matscipy` or `solver.free_slip=True` or "
                "`solver.name=RIE`."
            )
        kwargs = {"disp_fn": displacement_fn, "box_size": box_size, "state0": state}
        nw_fn = self._compute_wall_normals("jax", **kwargs) if is_nw_recompute else None

        return (
            cfg,
            box_size,
            state,
            g_ext_fn,
            bc_fn,
            nw_fn,
            eos,
            key,
            displacement_fn,
            shift_fn,
        )

    @abstractmethod
    def _box_size2D(self, n_walls):
        pass

    @abstractmethod
    def _box_size3D(self, n_walls):
        pass

    def _init_pos2D(self, box_size, dx, n_walls):
        r = pos_init_cartesian_2d(box_size, dx)
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        r = pos_init_cartesian_3d(box_size, dx)
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    @abstractmethod
    def _init_walls_2d(self):
        """Create all solid walls of a 2D case."""
        pass

    @abstractmethod
    def _init_walls_3d(self):
        """Create all solid walls of a 3D case."""
        pass

    @abstractmethod
    def _init_velocity2D(self, r):
        pass

    @abstractmethod
    def _init_velocity3D(self, r):
        pass

    @abstractmethod
    def _external_acceleration_fn(self, r):
        pass

    @abstractmethod
    def _boundary_conditions_fn(self, state):
        pass

    def _get_relaxed_r0(self, box_size, dx):
        assert hasattr(self, "_load_only_fluid"), AttributeError

        cfg = self.cfg
        name = "_".join([cfg.case.name, str(cfg.case.dim), str(dx), str(cfg.seed)])
        init_path = os.path.join("data_relaxed", name + ".h5")

        if not os.path.isfile(init_path):
            message = (
                f"python main.py config={cfg.config} mode=rlx seed={str(cfg.seed)} "
                f"case.dim={str(cfg.case.dim)} case.dx={str(cfg.case.dx)} "
                f"solver.name=SPH solver.tvf=1 solver.r0_noise_factor=0.25 "
                f"io.write_type=['h5'] io.data_path=data_relaxed/"
            )
            raise FileNotFoundError(f"First execute this: \n{message}")

        state = read_h5(init_path)
        if self._load_only_fluid:
            return state["r"][state["tag"] == Tag.FLUID]
        else:
            return state["r"], state["tag"]

    def _set_field_properties(self, num_particles, mass_ref, case):
        rho = jnp.ones(num_particles) * case.rho_ref
        mass = jnp.ones(num_particles) * mass_ref
        eta = jnp.ones(num_particles) * case.viscosity
        temperature = jnp.ones(num_particles) * case.T_ref
        kappa = jnp.ones(num_particles) * case.kappa_ref
        Cp = jnp.ones(num_particles) * case.Cp_ref
        return rho, mass, eta, temperature, kappa, Cp

    def _set_default_rlx(self):
        """Set default values for relaxation case setup.

        These would only change if the domain is not full.
        """

        self._box_size2D_rlx = self._box_size2D
        self._box_size3D_rlx = self._box_size3D
        self._init_pos2D_rlx = self._init_pos2D
        self._init_pos3D_rlx = self._init_pos3D

    def _compute_wall_normals(self, backend="scipy", **kwargs):
        if self.cfg.case.dim == 2:
            wall_part_fn = self._init_walls_2d
        elif self.cfg.case.dim == 3:
            wall_part_fn = self._init_walls_3d
        else:
            raise NotImplementedError("1D wall BCs not yet implemented")

        if backend == "scipy":
            # If one makes `tag` static (-> `self.tag`), this function can be jitted.
            # But it is significatly slower than `backend="jax"` due to `pure_callback`.
            def body(r, tag):
                return compute_nws_scipy(
                    r,
                    tag,
                    self.cfg.case.dx,
                    self.cfg.solver.n_walls,
                    self.offset_vec,
                    wall_part_fn,
                )
        elif backend == "jax":
            # This implementation is used in the integrator when having moving walls.
            body = compute_nws_jax_wrapper(
                state0=kwargs["state0"],
                dx=self.cfg.case.dx,
                n_walls=self.cfg.solver.n_walls,
                offset_vec=self.offset_vec,
                box_size=kwargs["box_size"],
                pbc=self.cfg.case.pbc,
                cfg_nl=self.cfg.nl,
                displacement_fn=kwargs["disp_fn"],
                wall_part_fn=wall_part_fn,
            )

        return body


def set_relaxation(Case, cfg):
    """Make a relaxation case from a SimulationSetup instance.

    Create a child class of a particular SimulationSetup instance and overwrite:

        - _init_pos{2|3}D
        - _box_size{2|3}D
        - _tag{2|3}D
        - _init_velocity{2|3}D
        - _external_acceleration_fn
        - _boundary_conditions_fn
    """

    class Rlx(Case):
        """Relax particles in a box"""

        def __init__(self, cfg):
            super().__init__(cfg)

            # custom variables related only to this Simulation
            self.case.g_ext_magnitude = 0.0

            # use the relaxation setup from the main case
            self._init_pos2D = self._init_pos2D_rlx
            self._init_pos3D = self._init_pos3D_rlx
            self._box_size2D = self._box_size2D_rlx
            self._box_size3D = self._box_size3D_rlx

        def _init_velocity2D(self, r):
            return jnp.zeros_like(r)

        def _init_velocity3D(self, r):
            return self._init_velocity2D(r)

        def _external_acceleration_fn(self, r):
            return jnp.zeros_like(r)

        def _boundary_conditions_fn(self, state):
            mask1 = jnp.isin(state["tag"], wall_tags)[:, None]

            state["u"] = jnp.where(mask1, 0.0, state["u"])
            state["v"] = jnp.where(mask1, 0.0, state["v"])

            state["dudt"] = jnp.where(mask1, 0.0, state["dudt"])
            state["dvdt"] = jnp.where(mask1, 0.0, state["dvdt"])

            return state

    return Rlx(cfg)


def load_case(case_root: str, case_py_file: str) -> SimulationSetup:
    """Load Case class from Python file.

    Args:
        case_root (str): Path to the case root directory, e.g. "cases/".
        case_py_file (str): Name of the Python case file, e.g. "db.py".
    """

    file_path = os.path.join(case_root, case_py_file)
    spec = importlib.util.spec_from_file_location("case_module", file_path)
    case_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case_module)

    # the class name has to be capital version of case_py_file without .py extension
    # e.g. "db.py" -> "DB"
    class_name = case_py_file.split(".")[0].upper()
    Case = getattr(case_module, class_name)

    return Case
