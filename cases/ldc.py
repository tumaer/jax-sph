"""Lid-driven cavity case setup"""

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


class LDC(SimulationSetup):
    """Lid-Driven Cavity"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # custom variables related only to this Simulation
        if self.case.dim == 2:
            self.u_lid = jnp.array([self.special.u_x_lid, 0.0])
        elif self.case.dim == 3:
            self.u_lid = jnp.array([self.special.u_x_lid, 0.0, 0.0])

        # define offset vector
        self.offset_vec = self._offset_vec()

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        return np.ones((2,)) + 6 * self.case.dx

    def _box_size3D(self):
        dx6 = 6 * self.case.dx
        return np.array([1 + dx6, 1 + dx6, 0.5])

    def _box_size2D(self, n_walls):
        return np.ones((2,)) + 2 * n_walls * self.case.dx

    def _box_size3D(self, n_walls):
        dx2n = 2 * n_walls * self.case.dx
        return np.array([1 + dx2n, 1 + dx2n, 0.5])

    def _init_walls_2d(self, dx, n_walls):
        rw = pos_box_2d(np.ones(2), dx, n_walls)
        return rw

    def _init_walls_3d(self, dx, n_walls):
        rw = pos_box_3d(np.array([1.0, 1.0, 0.5]), dx, n_walls)
        return rw

    def _init_pos2D(self, box_size, dx, n_walls):
        # initialize fluid phase
        r_f = n_walls * dx + pos_init_cartesian_2d(np.ones(2), dx)

        # initialize walls
        r_w = self._init_walls_2d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])

        # set velocity wall tag
        box_size = self._box_size2D(n_walls)
        mask_lid = r[:, 1] > (box_size[1] - n_walls * self.case.dx)
        tag = jnp.where(mask_lid, Tag.MOVING_WALL, tag)
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        # initialize fluid phase
        r_f = n_walls * dx + pos_init_cartesian_3d(np.array([1.0, 1.0, 0.5]), dx)

        # initialize walls
        r_w = self._init_walls_3d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])

        # set velocity wall tag
        box_size = self._box_size3D(n_walls)
        mask_lid = r[:, 1] > (box_size[1] - n_walls * self.case.dx)
        tag = jnp.where(mask_lid, Tag.MOVING_WALL, tag)
        return r, tag

    def _offset_vec(self):
        dim = self.cfg.case.dim
        if dim == 2:
            res = jnp.ones(dim) * self.cfg.solver.n_walls * self.case.dx
        elif dim == 3:
            res = jnp.array([1.0, 1.0, 0.0]) * self.cfg.solver.n_walls * self.case.dx
        return res

    def _init_velocity2D(self, r):
        u = jnp.zeros_like(r)

        # # Somewhat better initial velocity field for 2D LDC. The idea is to
        # # mix up the particles and somewhat resemble the stationary solution
        # x, y = r[0], r[1]
        # dx_3 = self.case.dx * 3
        # u_x = - jnp.sin(jnp.pi * (x - dx_3)) * jnp.sin(2 * jnp.pi * (y - dx_3))
        # u_y = jnp.sin(2 * jnp.pi * (x - dx_3)) * jnp.sin(jnp.pi * (y - dx_3))
        # u = u.at[0].set(u_x)
        # u = 0.5 * u.at[1].set(u_y)
        return u

    def _init_velocity3D(self, r):
        return self._init_velocity2D(r)

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        mask1 = state["tag"][:, None] == Tag.SOLID_WALL
        mask2 = state["tag"][:, None] == Tag.MOVING_WALL

        state["u"] = jnp.where(mask1, 0.0, state["u"])
        state["v"] = jnp.where(mask1, 0.0, state["v"])
        state["u"] = jnp.where(mask2, self.u_lid, state["u"])
        state["v"] = jnp.where(mask2, self.u_lid, state["v"])

        state["dudt"] = jnp.where(mask1, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0.0, state["dvdt"])
        state["dudt"] = jnp.where(mask2, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask2, 0.0, state["dvdt"])

        return state
