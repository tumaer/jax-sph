"""Utin test case setup"""

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import pos_init_cartesian_2d, pos_init_cartesian_3d


class UTSetup(SimulationSetup):
    """Unit Test: cube of water in periodic boundary box"""

    def __init__(self, args):
        super().__init__(args)

        self.L_wall = 1.0
        self.H_wall = 1.0
        self.L = 0.5
        self.H = 0.5

        self.args.g_ext_magnitude = 1.0
        self.args.is_bc_trick = False
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 1.0
        self.args.kappa = 0.0 
        self.args.Cp = 0.0

    def _box_size2D(self):
        return np.array([1.0, 1.0])

    def _box_size3D(self):
        return np.array([1.0, 1.0, 1.0])

    def _init_pos2D(self, box_size, dx):
        offset = np.array([[0.25, 0.25]])
        cube = np.array([self.L, self.H])
        return offset + pos_init_cartesian_2d(cube, dx)

    def _init_pos3D(self, box_size, dx):
        offset = np.array([[0.25, 0.25, 0.25]])
        cube = np.array([self.L, self.L, self.H])
        return offset + pos_init_cartesian_3d(cube, dx)

    def _tag2D(self, r):
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        return jnp.zeros(len(r), dtype=int)

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)
        res = res.at[:, 1].set(-self.args.g_ext_magnitude)
        return res

    def _boundary_conditions_fn(self, state):
        return state
