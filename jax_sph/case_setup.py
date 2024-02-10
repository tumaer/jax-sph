"""Simulation setup"""

import warnings
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from jax_sph.eos import TaitEoS
from jax_sph.eos import RIEMANNEoS
from jax_sph.utils import noise_masked, pos_init_cartesian_2d, pos_init_cartesian_3d

EPS = jnp.finfo(float).eps


class SimulationSetup(ABC):
    """Parent class for all simulation setups."""

    def __init__(self, args):
        self.args = args

    def initialize(self):
        """Initialize and return everything needed for the numerical setup"""

        # check whether these variables were defined in the child class
        assert hasattr(self.args, "g_ext_magnitude"), AttributeError
        assert hasattr(self.args, "is_bc_trick"), AttributeError
        assert hasattr(self.args, "p_bg_factor"), AttributeError

        args = self.args

        key_prng = jax.random.PRNGKey(args.seed)

        # Primal: reference density, dynamic viscosity, and velocity
        rho_ref = 1.00
        eta_ref = args.viscosity
        u_ref = 1.0 if not hasattr(self, "u_ref") else self.u_ref
        gamma_eos = 1.0
        print(f"Using gamma_EoS={gamma_eos}.")
        # Derived: reference speed of sound, pressure
        c_ref = 10.0 * u_ref
        p_ref = rho_ref * c_ref**2 / gamma_eos
        # for free surface simulation p_background has to be 0
        p_bg = args.p_bg_factor * p_ref

        # calculate volume and mass
        h = args.dx
        volume_ref = h**args.dim
        mass_ref = volume_ref * rho_ref

        # time integration step dt
        CFL = 1.0
        dt_convective = CFL * h / (c_ref + u_ref)
        dt_viscous = CFL * h**2 * rho_ref / eta_ref if eta_ref != 0.0 else 999
        dt_body_force = CFL * (h / (self.args.g_ext_magnitude + EPS)) ** 0.5
        dt = np.amin([dt_convective, dt_viscous, dt_body_force])
        # TODO: this computation has to be applied at each time step as the
        # convective term changes with the current u (not u_ref)
        # Especially relevant for RPF

        print("dt_convective :", dt_convective)
        print("dt_viscous    :", dt_viscous)
        print("dt_body_force :", dt_body_force)
        print("dt_max        :", dt)

        # assert dt > args.dt, ValueError("Explicit dt has to comply with CFL.")
        if args.dt != 0.0:
            if args.dt > dt:
                warnings.warn("Explicit dt should comply with CFL.", UserWarning)
            dt = args.dt

        print("dt_final      :", dt)

        if args.case == "Rlx":
            # run a relaxation of randomly initialized state for 500 steps
            sequence_length = 5000
            args.t_end = dt * sequence_length
            # turn background pressure on for homogeneous particle distribution
        else:
            sequence_length = int(args.t_end / dt)

        # Equation of state
        if args.solver == "RIE":
            eos = RIEMANNEoS(rho_ref, p_bg, args.Vmax)
        elif args.solver == "RIE2":
            eos = RIEMANNEoS(rho_ref, p_bg, args.Vmax)
        else:
            eos = TaitEoS(p_ref, rho_ref, p_bg, gamma_eos)

        # initialize box and regular grid positions of particles
        # Tag particle: 0 - fluid, 1 - solid wall, 2 - moving wall
        if args.dim == 2:
            box_size = self._box_size2D()
            r = self._init_pos2D(box_size, args.dx)
            tag = self._tag2D(r)
        elif args.dim == 3:
            box_size = self._box_size3D()
            r = self._init_pos3D(box_size, args.dx)
            tag = self._tag3D(r)

        num_particles = len(r)
        print("Total number of particles = ", num_particles)

        # add noise to the fluid particles (tag=0) to break symmetry
        key, subkey = jax.random.split(key_prng)
        if args.r0_noise_factor != 0.0:
            noise_std = args.r0_noise_factor * args.dx
            r = noise_masked(r, tag == 0, subkey, std=noise_std)
            # PBC: move all particles to the box limits after noise addition
            r = r % jnp.array(box_size)

        # initialize the velocity given the coordinates r with the noise
        if args.dim == 2:
            v = vmap(self._init_velocity2D)(r)
        elif args.dim == 3:
            v = vmap(self._init_velocity3D)(r)

        # initialize all other field values
        rho = jnp.ones(num_particles) * rho_ref
        mass = jnp.ones(num_particles) * mass_ref
        eta = jnp.ones(num_particles) * eta_ref

        # initialize accelerations for TGV
        if jnp.logical_and(args.case == "TGV", args.dim == 2):
            dvdt = vmap(self._init_acceleration2D)(r)
            
            state = {
                "r": r,
                "tag": tag,
                "u": v,
                "v": v,
                "dudt": dvdt,
                "dvdt": dvdt,
                "drhodt": jnp.zeros_like(rho),
                "rho": rho,
                "p": eos.p_fn(rho),
                "mass": mass,
                "eta": eta,
            }

        else:
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
            }

        args.dt, args.sequence_length = dt, sequence_length
        args.num_particles_max = num_particles
        # TODO: embed this in the code - for dataset generation
        args.periodic_boundary_conditions = [True, True, True]
        args.bounds = np.array([np.zeros_like(box_size), box_size]).T.tolist()

        state = self._boundary_conditions_fn(state)

        g_ext_fn = self._external_acceleration_fn
        bc_fn = self._boundary_conditions_fn

        return args, box_size, state, g_ext_fn, bc_fn, eos, key

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
    def _init_acceleration2D(self, r):
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
