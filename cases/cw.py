"""Cube of water falling under gravity case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import (
    Tag,
    pos_box_2d,
    pos_box_3d,
    pos_init_cartesian_2d,
    pos_init_cartesian_3d,
)


class CW(SimulationSetup):
    """Cube of Water Experiment"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # define offset vector
        self.offset_vec = np.ones(cfg.case.dim) * cfg.solver.n_walls * cfg.case.dx

        # relaxation configurations
        if cfg.case.mode == "rlx" or cfg.case.r0_type == "relaxed":
            raise NotImplementedError("Relaxation not implemented for CW.")

    def _box_size2D(self, n_walls):
        sp = self.special
        return np.array([sp.L_wall, sp.H_wall]) + 2 * n_walls * self.case.dx

    def _box_size3D(self, n_walls):
        sp = self.special
        dx2n = 2 * n_walls * self.case.dx
        return np.array([sp.L_wall + dx2n, sp.H_wall + dx2n, 1.0 + dx2n])

    def _init_walls_2d(self, dx, n_walls):
        sp = self.special
        rw = pos_box_2d(np.array([sp.L_wall, sp.H_wall]), dx, n_walls)
        return rw

    def _init_walls_3d(self, dx, n_walls):
        sp = self.special
        rw = pos_box_3d(np.array([sp.L_wall, sp.H_wall, 1.0]), dx, n_walls, False)
        return rw

    def _init_pos2D(self, box_size, dx, n_walls):
        dxn = n_walls * self.case.dx

        # initialize walls
        r_w = self._init_walls_2d(dx, n_walls)

        # initialize fluid phase
        r_f = pos_init_cartesian_2d(np.array([self.special.L, self.special.H]), dx)
        r_f += dxn + np.array(self.special.cube_offset)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        dxn = n_walls * self.case.dx

        # initialize walls
        r_w = self._init_walls_3d(dx, n_walls)

        # initialize fluid phase
        r_f = pos_init_cartesian_3d(np.array([self.special.L, self.special.H, 0.3]), dx)
        r_f += dxn + np.array(self.special.cube_offset)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])
        return r, tag

    def _init_velocity2D(self, r):
        res = jnp.ones_like(r) * jnp.array(self.special.u_init)
        return res

    def _init_velocity3D(self, r):
        return jnp.ones_like(r) * jnp.array(self.special.u_init)

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
