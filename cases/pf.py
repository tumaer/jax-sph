"""Poiseuille flow case setup"""

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class PF(SimulationSetup):
    """Poiseuille Flow.

    Setup from "A generalized wall boundary condition [...]", Adami 2012
    """

    def __init__(self, args):
        super().__init__(args)

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1000
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.0

        # custom variables related only to this simulation
        self.args.u_ref = 1.25

        # relaxation configurations
        if self.args.mode == "rlx":
            self._set_default_rlx()

        if args.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        return np.array([0.4, 1 + 6 * self.args.dx])

    def _box_size3D(self):
        return np.array([0.4, 1 + 6 * self.args.dx, 0.4])

    def _tag2D(self, r):
        dx3 = 3 * self.args.dx
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
        dx3 = 3 * self.args.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r)))
        box_size = self._box_size2D()
        fluid_mask = (r[:, 1] < box_size[1] - dx3) * (r[:, 1] > dx3)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.args.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        mask1 = state["tag"][:, None] == Tag.SOLID_WALL
        state["u"] = jnp.where(mask1, 0, state["u"])
        state["v"] = jnp.where(mask1, 0, state["v"])
        state["dudt"] = jnp.where(mask1, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])

        return state
