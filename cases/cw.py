"""Cube of water falling under gravity case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_box_2d, pos_init_cartesian_2d


class CW(SimulationSetup):
    """Cube of Water Experiment"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # relaxation configurations
        if cfg.case.mode == "rlx" or cfg.case.r0_type == "relaxed":
            raise NotImplementedError("Relaxation not implemented for CW.")

    def _box_size2D(self):
        return np.array([self.special.L_wall, self.special.H_wall]) + 6 * self.case.dx

    def _box_size3D(self):
        dx6 = 6 * self.case.dx
        return np.array([self.special.L_wall + dx6, self.special.H_wall + dx6, 0.5])

    def _init_pos2D(self, box_size, dx):
        dx3 = 3 * self.case.dx
        walls = pos_box_2d(self.special.L_wall, self.special.H_wall, dx)

        r_fluid = pos_init_cartesian_2d(np.array([self.special.L, self.special.H]), dx)
        r_fluid += dx3 + np.array(self.special.cube_offset)
        res = np.concatenate([walls, r_fluid])
        return res

    def _tag2D(self, r):
        dx3 = 3 * self.case.dx
        mask_left = jnp.where(r[:, 0] < dx3, True, False)
        mask_bottom = jnp.where(r[:, 1] < dx3, True, False)
        mask_right = jnp.where(r[:, 0] > self.special.L_wall + dx3, True, False)
        mask_top = jnp.where(r[:, 1] > self.special.H_wall + dx3, True, False)
        mask_wall = mask_left + mask_bottom + mask_right + mask_top

        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        tag = jnp.where(mask_wall, Tag.SOLID_WALL, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        res = jnp.array(self.special.u_init)
        return res

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)
        res = res.at[:, 1].set(-self.case.g_ext_magnitude)
        return res

    def _boundary_conditions_fn(self, state):
        mask_wall = state["tag"] == Tag.SOLID_WALL

        state["u"] = jnp.where(mask_wall[:, None], 0.0, state["u"])
        state["v"] = jnp.where(mask_wall[:, None], 0.0, state["v"])
        state["dudt"] = jnp.where(mask_wall[:, None], 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask_wall[:, None], 0.0, state["dvdt"])
        state["p"] = jnp.where(mask_wall, 0.0, state["p"])
        return state
