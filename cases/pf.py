"""Poiseuille flow case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class PF(SimulationSetup):
    """Poiseuille Flow.

    Setup from "A generalized wall boundary condition [...]", Adami 2012
    """

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
        return np.array([0.4, 1 + 6 * self.case.dx])

    def _box_size3D(self):
        return np.array([0.4, 1 + 6 * self.case.dx, 0.4])

    def _tag2D(self, r):
        dx3 = 3 * self.case.dx
        box_size = self._box_size2D()
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)

        mask_wall = (r[:, 1] < dx3) + (r[:, 1] > box_size[1] - dx3)
        tag = jnp.where(mask_wall, Tag.SOLID_WALL, tag)
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
        mask1 = state["tag"][:, None] == Tag.SOLID_WALL
        state["u"] = jnp.where(mask1, 0, state["u"])
        state["v"] = jnp.where(mask1, 0, state["v"])
        state["dudt"] = jnp.where(mask1, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])

        return state

    def _init_density(self, r):
        return jnp.ones(jnp.shape(r)[0:1]) * self.case.rho_ref
