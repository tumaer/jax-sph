"""Dam break case setup"""

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


class DB(SimulationSetup):
    """Dam Break"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # height and length are as presented in
        # A. Colagrossi, "Numerical simulation of interfacial flows by smoothed
        # particle hydrodynamics", J. Comput. Phys. 191 (2) (2003) 448–475.
        # |---------------------------|
        # |                           |
        # |-------------              |
        # |<     L    >| H            |  H_wall
        # | --------------------------|
        #  <          L_wall         >

        # define offset vector
        self.offset_vec = self._offset_vec()

        # relaxation configurations
        if self.case.mode == "rlx":
            self.special.L_wall = self.special.L
            self.special.H_wall = self.special.H
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = True

    def _box_size2D(self, n_walls):
        dx, bo = self.case.dx, self.special.box_offset
        return np.array(
            [
                self.special.L_wall + 2 * n_walls * dx + bo,
                self.special.H_wall + 2 * n_walls * dx + bo,
            ]
        )

    def _box_size3D(self, n_walls):
        dx, bo = self.case.dx, self.box_offset
        sp = self.special
        return np.array(
            [sp.L_wall + 2 * n_walls * dx + bo, sp.H_wall + 2 * n_walls * dx + bo, sp.W]
        )

    def _init_walls_2d(self, dx, n_walls):
        sp = self.special
        rw = pos_box_2d(np.array([sp.L_wall, sp.H_wall]), dx, n_walls)
        return rw

    def _init_walls_3d(self, dx, n_walls):
        sp = self.special
        rw = pos_box_3d(np.array([sp.L_wall, sp.H_wall, 1.0]), dx, n_walls)
        return rw

    def _init_pos2D(self, box_size, dx, n_walls):
        sp = self.special

        # initialize fluid phase
        if self.case.r0_type == "cartesian":
            r_f = n_walls * dx + pos_init_cartesian_2d(np.array([sp.L, sp.H]), dx)
        else:
            r_f = self._get_relaxed_r0(None, dx)

        # initialize walls
        r_w = self._init_walls_2d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        # TODO: not validated yet
        sp = self.special

        # initialize fluid phase
        if self.case.r0_type == "cartesian":
            r_f = np.array([1.0, 1.0, 0.0]) * n_walls * dx + pos_init_cartesian_3d(
                np.array([sp.L, sp.H, 1.0]), dx
            )
        else:
            r_f = self._get_relaxed_r0(None, dx)

        # initialize walls
        r_w = self._init_walls_3d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])

        return r, tag

    def _offset_vec(self):
        dim = self.cfg.case.dim
        if dim == 2:
            res = np.ones(dim) * self.cfg.solver.n_walls * self.cfg.case.dx
        elif dim == 3:
            res = np.array([1.0, 1.0, 0.0]) * self.cfg.solver.n_walls * self.cfg.case.dx
        return res

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

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
