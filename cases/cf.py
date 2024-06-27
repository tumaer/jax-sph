"""Couette flow case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_init_cartesian_2d, pos_init_cartesian_3d


class CF(SimulationSetup):
    """Couette Flow.

    Setup based on "Modeling Low Reynolds Number Incompressible [...], Morris 1997,
    and similar to PF case.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # custom variables related only to this Simulation
        if self.case.dim == 2:
            self.u_wall = jnp.array([self.special.u_x_wall, 0.0])
        elif self.case.dim == 3:
            self.u_wall = jnp.array([self.special.u_x_wall, 0.0, 0.0])

        # define offset vector
        self.offset_vec = self._offset_vec()

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self, n_walls):
        dx2n = self.case.dx * n_walls * 2
        sp = self.special
        return np.array([sp.L, sp.H + dx2n])

    def _box_size3D(self, n_walls):
        dx2n = self.case.dx * n_walls * 2
        sp = self.special
        return np.array([sp.L, sp.H + dx2n, 0.4])

    def _init_walls_2d(self, dx, n_walls):
        sp = self.special

        # thickness of wall particles
        dxn = dx * n_walls

        # horizontal and vertical blocks
        horiz = pos_init_cartesian_2d(np.array([sp.L, dxn]), dx)

        # wall: bottom, top
        wall_b = horiz.copy()
        wall_t = horiz.copy() + np.array([0.0, sp.H + dxn])

        rw = np.concatenate([wall_b, wall_t])
        return rw

    def _init_walls_3d(self, dx, n_walls):
        sp = self.special

        # thickness of wall particles
        dxn = dx * n_walls

        # horizontal and vertical blocks
        horiz = pos_init_cartesian_3d(np.array([sp.L, dxn, 0.4]), dx)

        # wall: bottom, top
        wall_b = horiz.copy()
        wall_t = horiz.copy() + np.array([0.0, sp.H + dxn, 0.0])

        rw = np.concatenate([wall_b, wall_t])
        return rw

    def _init_pos2D(self, box_size, dx, n_walls):
        sp = self.special

        # initialize fluid phase
        r_f = np.array([0.0, 1.0]) * n_walls * dx + pos_init_cartesian_2d(
            np.array([sp.L, sp.H]), dx
        )

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
        sp = self.special

        # initialize fluid phase
        r_f = np.array([0.0, 1.0, 0.0]) * n_walls * dx + pos_init_cartesian_3d(
            np.array([sp.L, sp.H, 0.4]), dx
        )

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
            res = np.array([0.0, 1.0]) * self.cfg.solver.n_walls * self.cfg.case.dx
        elif dim == 3:
            res = np.array([0.0, 1.0, 0.0]) * self.cfg.solver.n_walls * self.cfg.case.dx
        return res

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        mask1 = state["tag"][:, None] == Tag.SOLID_WALL
        mask2 = state["tag"][:, None] == Tag.MOVING_WALL

        state["u"] = jnp.where(mask1, 0.0, state["u"])
        state["v"] = jnp.where(mask1, 0.0, state["v"])
        state["u"] = jnp.where(mask2, self.u_wall, state["u"])
        state["v"] = jnp.where(mask2, self.u_wall, state["v"])

        state["dudt"] = jnp.where(mask1, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0.0, state["dvdt"])
        state["dudt"] = jnp.where(mask2, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask2, 0.0, state["dvdt"])

        return state
