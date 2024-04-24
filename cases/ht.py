"Heat Transfer over a flat plate setup"


import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class HT(SimulationSetup):
    "Heat Transfer"

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        dx = self.case.dx
        return np.array([1, 0.2 + 6 * dx])

    def _box_size3D(self):
        dx = self.case.dx
        return np.array([1, 0.2 + 6 * dx, 0.5])

    def _tag2D(self, r):
        dx3 = 3 * self.case.dx
        _box_size = self._box_size2D()
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)

        mask_no_slip_wall = (r[:, 1] < dx3) + (
            r[:, 1] > (_box_size[1] - 6 * self.case.dx) + dx3
        )
        mask_hot_wall = (
            (r[:, 1] < dx3)
            * (r[:, 0] < (_box_size[0] / 2) + self.special.hot_wall_half_width)
            * (r[:, 0] > (_box_size[0] / 2) - self.special.hot_wall_half_width)
        )
        tag = jnp.where(mask_no_slip_wall, Tag.SOLID_WALL, tag)
        tag = jnp.where(mask_hot_wall, Tag.DIRICHLET_WALL, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        dx3 = 3 * self.case.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r)))
        box_size = self._box_size2D()
        fluid_mask = (r[:, 1] < box_size[1] - dx3) * (r[:, 1] > dx3)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.case.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        mask_fluid = state["tag"] == Tag.FLUID

        # set incoming fluid temperature to reference_temperature
        mask_inflow = mask_fluid * (state["r"][:, 0] < 3 * self.case.dx)
        state["T"] = jnp.where(mask_inflow, self.cfg.T_ref, state["T"])
        state["dTdt"] = jnp.where(mask_inflow, 0.0, state["dTdt"])

        # set the hot wall to hot_wall_temperature.
        mask_hot = state["tag"] == Tag.DIRICHLET_WALL  # hot wall
        state["T"] = jnp.where(mask_hot, self.special.hot_wall_temperature, state["T"])
        state["dTdt"] = jnp.where(mask_hot, 0.0, state["dTdt"])

        # set the fixed wall to reference_temperature.
        mask_solid = state["tag"] == Tag.SOLID_WALL  # fixed wall
        state["T"] = jnp.where(mask_solid, self.cfg.T_ref, state["T"])
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
            state["r"][:, 0] > self.case.bounds[0][1] - 3 * self.case.dx
        )
        state["dTdt"] = jnp.where(mask_outflow, 0.0, state["dTdt"])

        return state

    def _init_density(self, r):
        return jnp.ones(jnp.shape(r)[0:1]) * self.case.rho_ref
