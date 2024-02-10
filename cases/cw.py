"""Cube of water falling under gravity case setup"""

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import pos_init_cartesian_2d


class CW(SimulationSetup):
    """Cube of Water Experiment"""

    def __init__(self, args):
        super().__init__(args)

        self.L_wall = 1.0
        self.H_wall = 1.0
        self.L = 0.5
        self.H = 0.5

        # a trick to reduce computation using PBC in z-direction
        self.box_offset = 1.0

        self.args.g_ext_magnitude = 1.0
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 1.0

    def _box_size2D(self):
        return np.array([1.0, 3.0]) + 6 * self.args.dx

    def _box_size3D(self):
        dx6 = 6 * self.args.dx
        return np.array([1 + dx6, 3 + dx6, 0.5])

    def _init_pos2D(self, box_size, dx):
        dx3 = 3 * self.args.dx
        dx6 = 6 * self.args.dx
        L_wall, H_wall = self.L_wall, self.H_wall

        # horizontal and vertical blocks
        vertical = pos_init_cartesian_2d(np.array([dx3, H_wall + dx6]), dx)
        horiz = pos_init_cartesian_2d(np.array([L_wall, dx3]), dx)

        # wall: left, bottom, right, top
        wall_l = vertical.copy()
        wall_b = horiz.copy() + np.array([dx3, 0.0])
        wall_r = vertical.copy() + np.array([L_wall + dx3, 0.0])
        wall_t = horiz.copy() + np.array([dx3, H_wall + dx3])

        r_fluid = dx3 + pos_init_cartesian_2d(np.array([self.L, self.H]), dx)
        res = np.concatenate([wall_l, wall_b, wall_r, wall_t, r_fluid])
        return res

    def _tag2D(self, r):
        mask_left = jnp.where(r[:, 0] < 3 * self.args.dx, True, False)
        mask_bottom = jnp.where(r[:, 1] < 3 * self.args.dx, True, False)
        mask_right = jnp.where(r[:, 0] > 1 + 3 * self.args.dx, True, False)
        mask_wall = mask_left + mask_bottom + mask_right

        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.zeros(len(r), dtype=int)
        tag = jnp.where(mask_wall, 1, tag)
        return tag

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
        mask_wall = state["tag"][:, None] == 1
        mask_wall_1d = state["tag"] == 1

        state["u"] = jnp.where(mask_wall, 0.0, state["u"])
        state["v"] = jnp.where(mask_wall, 0.0, state["v"])
        state["dudt"] = jnp.where(mask_wall, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask_wall, 0.0, state["dvdt"])
        # pay attention to the shapes
        state["p"] = jnp.where(mask_wall_1d, 0.0, state["p"])
        return state

    def _init_acceleration2D(self, r):
        pass
