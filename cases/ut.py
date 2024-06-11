"""Utin test case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_init_cartesian_2d, pos_init_cartesian_3d


class UT(SimulationSetup):
    """Unit Test: cube of water in periodic boundary box"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.cube_offset = np.array(self.special.cube_offset)

        # relaxation configurations
        if self.case.mode == "rlx" or self.case.r0_type == "relaxed":
            raise NotImplementedError("Relaxation not implemented for CW")

    def _box_size2D(self, n_walls):
        return np.array([self.special.L_wall, self.special.H_wall])

    def _box_size3D(self, n_walls):
        return np.array([self.special.L_wall, self.special.L_wall, self.special.L_wall])

    def _init_walls_2d(self):
        pass

    def _init_walls_3d(self):
        pass

    def _init_pos2D(self, box_size, dx, n_walls):
        cube = np.array([self.special.L, self.special.H])
        r = self.cube_offset + pos_init_cartesian_2d(cube, dx)
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        cube = np.array([self.special.L, self.special.L, self.special.H])
        r = self.cube_offset + pos_init_cartesian_3d(cube, dx)
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)
        res = res.at[:, 1].set(-self.case.g_ext_magnitude)
        return res

    def _boundary_conditions_fn(self, state):
        return state
