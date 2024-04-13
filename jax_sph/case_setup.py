"""Simulation setup."""

import os
import warnings
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import space

from jax_sph.eos import RIEMANNEoS, TaitEoS
from jax_sph.io_state import read_h5
from jax_sph.utils import (
    Tag,
    get_noise_masked,
    pos_init_cartesian_2d,
    pos_init_cartesian_3d,
    wall_tags,
)

EPS = jnp.finfo(float).eps


class SimulationSetup(ABC):
    """Parent class for all simulation setups."""

    def __init__(self, args):
        self.args = args

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

           - args (Namespace): configuration arguments
           - box_size (ndarray): size of the simulation box starting at 0.0
           - state (dict): dictionary containing all field values
           - g_ext_fn (Callable): external force function
           - bc_fn (Callable): boundary conditions function (e.g. velocity at walls)
           - eos (Callable): equation of state function
           - key (PRNGKey): random key for sampling
           - displacement_fn (Callable): displacement function for edge features
           - shift_fn (Callable): shift function for position updates
        """

        # check whether these variables were defined in the child class
        assert hasattr(self.args, "g_ext_magnitude"), AttributeError
        assert hasattr(self.args, "is_bc_trick"), AttributeError
        assert hasattr(self.args, "p_bg_factor"), AttributeError

        args = self.args

        key_prng = jax.random.PRNGKey(args.seed)

        # reference temperature, kappa, cp
        T_ref = 1.0
        kappa_ref = 0.0 if not hasattr(args, "kappa") else args.kappa
        Cp_ref = 0.0 if not hasattr(args, "Cp") else args.Cp

        # Primal: reference density, dynamic viscosity, and velocity
        rho_ref = 1.00
        eta_ref = args.viscosity
        args.u_ref = 1.0 if args.u_ref is None else args.u_ref
        gamma_eos = 1.0
        print(f"Using gamma_EoS={gamma_eos}.")
        # Derived: reference speed of sound, pressure
        args.c_ref = 10.0 * args.u_ref
        p_ref = rho_ref * args.c_ref**2 / gamma_eos
        # for free surface simulation p_background has to be 0
        p_bg = args.p_bg_factor * p_ref

        # calculate volume and mass
        h = args.dx
        volume_ref = h**args.dim
        mass_ref = volume_ref * rho_ref

        # time integration step dt
        CFL = 0.25
        dt_convective = CFL * h / (args.c_ref + args.u_ref)
        dt_viscous = CFL * h**2 * rho_ref / eta_ref if eta_ref != 0.0 else 999
        dt_body_force = CFL * (h / (self.args.g_ext_magnitude + EPS)) ** 0.5
        dt = np.amin([dt_convective, dt_viscous, dt_body_force])
        # TODO: consider adaptive time step sizes

        print("dt_convective :", dt_convective)
        print("dt_viscous    :", dt_viscous)
        print("dt_body_force :", dt_body_force)
        print("dt_max        :", dt)

        # assert dt > args.dt, ValueError("Explicit dt has to comply with CFL.")
        if args.dt is not None:
            if args.dt > dt:
                warnings.warn("Explicit dt should comply with CFL.", UserWarning)
            dt = args.dt

        print("dt_final      :", dt)

        if args.mode == "rlx":
            # run a relaxation of randomly initialized state for 500 steps
            sequence_length = 5000
            args.t_end = dt * sequence_length
            # turn background pressure on for homogeneous particle distribution
        else:
            sequence_length = int(args.t_end / dt)

        # Equation of state
        if args.solver == "RIE":
            eos = RIEMANNEoS(rho_ref, p_bg, args.u_ref)
        else:
            eos = TaitEoS(p_ref, rho_ref, p_bg, gamma_eos)

        # initialize box and positions of particles
        if args.dim == 2:
            box_size = self._box_size2D()
            r = self._init_pos2D(box_size, args.dx)
            tag = self._tag2D(r)
        elif args.dim == 3:
            box_size = self._box_size3D()
            r = self._init_pos3D(box_size, args.dx)
            tag = self._tag3D(r)
        displacement_fn, shift_fn = space.periodic(side=box_size)

        num_particles = len(r)
        print("Total number of particles = ", num_particles)

        # add noise to the fluid particles to break symmetry
        key, subkey = jax.random.split(key_prng)
        if args.r0_noise_factor != 0.0:
            noise_std = args.r0_noise_factor * args.dx
            noise = get_noise_masked(r.shape, tag == Tag.FLUID, subkey, std=noise_std)
            # PBC: move all particles to the box limits after noise addition
            r = shift_fn(r, noise)

        # initialize the velocity given the coordinates r with the noise
        if args.dim == 2:
            v = vmap(self._init_velocity2D)(r)
        elif args.dim == 3:
            v = vmap(self._init_velocity3D)(r)

        # initialize all other field values
        rho = jnp.ones(num_particles) * rho_ref
        mass = jnp.ones(num_particles) * mass_ref
        eta = jnp.ones(num_particles) * eta_ref
        temp = jnp.ones(num_particles) * T_ref
        kappa = jnp.ones(num_particles) * kappa_ref
        Cp = jnp.ones(num_particles) * Cp_ref

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
            "T": temp,
            "kappa": kappa,
            "Cp": Cp,
        }

        # overwrite the state dictionary with the provided one
        if args.state0_path != "none":
            _state = read_h5(args.state0_path)
            for k in state:
                if k not in _state:
                    warnings.warn(f"Key {k} not found in state0 file.", UserWarning)
                    continue
                assert state[k].shape == _state[k].shape, ValueError(
                    f"Shape mismatch for key {k} while loading initial state."
                )
                state[k] = _state[k]

        # the following arguments are needed for dataset generation
        args.dt, args.sequence_length = dt, sequence_length
        args.num_particles_max = num_particles
        args.periodic_boundary_conditions = [True, True, True]
        args.bounds = np.array([np.zeros_like(box_size), box_size]).T.tolist()

        state = self._boundary_conditions_fn(state)

        g_ext_fn = self._external_acceleration_fn
        bc_fn = self._boundary_conditions_fn

        return (
            args,
            box_size,
            state,
            g_ext_fn,
            bc_fn,
            eos,
            key,
            displacement_fn,
            shift_fn,
        )

    @abstractmethod
    def _box_size2D(self, args):
        pass

    @abstractmethod
    def _box_size3D(self, args):
        pass

    def _init_pos2D(self, box_size, dx):
        return pos_init_cartesian_2d(box_size, dx)

    def _init_pos3D(self, box_size, dx):
        return pos_init_cartesian_3d(box_size, dx)

    @abstractmethod
    def _tag2D(self, r):
        pass

    @abstractmethod
    def _tag3D(self, r):
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

        args = self.args
        name = "_".join([args.case.lower(), str(args.dim), str(dx), str(args.seed)])
        init_path = "data_relaxed/" + name + ".h5"

        if not os.path.isfile(init_path):
            message = (
                f"python main.py --case={args.case} --mode=rlx --solver=SPH --tvf=1.0 "
                f"--dim={str(args.dim)} --dx={str(dx)} --seed={str(args.seed)} "
                f"--write-h5 --r0-noise-factor=0.25 --data-path=data_relaxed"
            )
            raise FileNotFoundError(f"First execute this: \n{message}")

        state = read_h5(init_path)
        if self._load_only_fluid:
            return state["r"][state["tag"] == Tag.FLUID]
        else:
            return state["r"]

    def _set_default_rlx(self):
        """Set default values for relaxation case setup.

        These would only change if the domain is not full.
        """

        self._box_size2D_rlx = self._box_size2D
        self._box_size3D_rlx = self._box_size3D
        self._init_pos2D_rlx = self._init_pos2D
        self._init_pos3D_rlx = self._init_pos3D
        self._tag2D_rlx = self._tag2D
        self._tag3D_rlx = self._tag3D


def set_relaxation(Case, args):
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

        def __init__(self, args):
            super().__init__(args)

            # custom variables related only to this Simulation
            self.args.g_ext_magnitude = 0.0

            # use the relaxation setup from the main case
            self.args.is_bc_trick = self.args.is_bc_trick
            self.args.p_bg_factor = self.args.p_bg_factor
            self._init_pos2D = self._init_pos2D_rlx
            self._init_pos3D = self._init_pos3D_rlx
            self._box_size2D = self._box_size2D_rlx
            self._box_size3D = self._box_size3D_rlx
            self._tag2D = self._tag2D_rlx
            self._tag3D = self._tag3D_rlx

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

    return Rlx(args)
