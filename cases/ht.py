"Heat Transfer over a flat plate setup"


import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_init_cartesian_2d, pos_init_cartesian_3d


class HT(SimulationSetup):
    "Heat Transfer"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

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
        return np.array([sp.L, sp.H + dx2n, 0.5])

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
        horiz = pos_init_cartesian_3d(np.array([sp.L, dxn, 0.5]), dx)

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

        # set thermal tags
        _box_size = self._box_size2D(n_walls)
        mask_hot_wall = (
            (r[:, 1] < dx * n_walls)
            * (r[:, 0] < (_box_size[0] / 2) + self.special.hot_wall_half_width)
            * (r[:, 0] > (_box_size[0] / 2) - self.special.hot_wall_half_width)
        )
        tag = jnp.where(mask_hot_wall, Tag.DIRICHLET_WALL, tag)

        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        sp = self.special

        # initialize fluid phase
        r_f = np.array([0.0, 1.0, 0.0]) * n_walls * dx + pos_init_cartesian_3d(
            np.array([sp.L, sp.H, 0.5]), dx
        )

        # initialize walls
        r_w = self._init_walls_3d(dx, n_walls)

        # set tags
        tag_f = jnp.full(len(r_f), Tag.FLUID, dtype=int)
        tag_w = jnp.full(len(r_w), Tag.SOLID_WALL, dtype=int)

        r = np.concatenate([r_w, r_f])
        tag = np.concatenate([tag_w, tag_f])

        # set thermal tags
        _box_size = self._box_size3D(n_walls)
        mask_hot_wall = (
            (r[:, 1] < dx * n_walls)
            * (r[:, 0] < (_box_size[0] / 2) + self.special.hot_wall_half_width)
            * (r[:, 0] > (_box_size[0] / 2) - self.special.hot_wall_half_width)
        )
        tag = jnp.where(mask_hot_wall, Tag.DIRICHLET_WALL, tag)

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
        n_walls = self.cfg.solver.n_walls
        dxn = n_walls * self.case.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r)))
        box_size = self._box_size2D(n_walls)
        fluid_mask = (r[:, 1] < box_size[1] - dxn) * (r[:, 1] > dxn)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.case.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        n_walls = self.cfg.solver.n_walls
        mask_fluid = state["tag"] == Tag.FLUID

        # set incoming fluid temperature to reference_temperature
        mask_inflow = mask_fluid * (state["r"][:, 0] < n_walls * self.case.dx)
        state["T"] = jnp.where(mask_inflow, self.case.T_ref, state["T"])
        state["dTdt"] = jnp.where(mask_inflow, 0.0, state["dTdt"])

        # set the hot wall to hot_wall_temperature.
        mask_hot = state["tag"] == Tag.DIRICHLET_WALL  # hot wall
        state["T"] = jnp.where(mask_hot, self.special.hot_wall_temperature, state["T"])
        state["dTdt"] = jnp.where(mask_hot, 0.0, state["dTdt"])

        # set the fixed wall to reference_temperature.
        mask_solid = state["tag"] == Tag.SOLID_WALL  # fixed wall
        state["T"] = jnp.where(mask_solid, self.case.T_ref, state["T"])
        state["dTdt"] = jnp.where(mask_solid, 0, state["dTdt"])

        # ensure static walls have no velocity or acceleration
        mask_static = (mask_hot + mask_solid)[:, None]
        state["u"] = jnp.where(mask_static, 0.0, state["u"])
        state["v"] = jnp.where(mask_static, 0.0, state["v"])
        state["dudt"] = jnp.where(mask_static, 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask_static, 0.0, state["dvdt"])

        # set outlet temperature gradients to zero to avoid interaction with inflow
        # bounds[0][1] is the x-coordinate of the outlet
        mask_outflow = mask_fluid * (
            state["r"][:, 0] > self.case.bounds[0][1] - n_walls * self.case.dx
        )
        state["dTdt"] = jnp.where(mask_outflow, 0.0, state["dTdt"])

        return state
