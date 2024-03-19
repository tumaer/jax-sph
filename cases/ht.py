"Heat Transfer over a flat plate setup"


import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class HT(SimulationSetup):
    "Heat Transfer"

    def __init__(self, args):
        super().__init__(args)

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 2.3
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.05  # same as RPF

        # custom variables related only to this simulation
        self.args.kappa = 7.313  # thermal conductivity value at 50°C
        self.args.Cp = 305.27  # specific heat at constant pressure value at 50°C
        self.args.hot_wall_temperature = 1.23  # temp corresponding to 100°C
        self.args.reference_temperature = 1.0
        self.args.hot_wall_half_width = 0.25

        # relaxation configurations
        if self.args.mode == "rlx":
            self._set_default_rlx()

        if args.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        dx = self.args.dx
        return np.array([1, 0.2 + 6 * dx])

    def _box_size3D(self):
        dx = self.args.dx
        return np.array([1, 0.2 + 6 * dx, 0.5])

    def _tag2D(self, r):
        dx3 = 3 * self.args.dx
        _box_size = self._box_size2D()
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)

        mask_no_slip_wall = (r[:, 1] < dx3) + (
            r[:, 1] > (_box_size[1] - 6 * self.args.dx) + dx3
        )
        mask_hot_wall = (
            (r[:, 1] < dx3)
            * (r[:, 0] < (_box_size[0] / 2) + self.args.hot_wall_half_width)
            * (r[:, 0] > (_box_size[0] / 2) - self.args.hot_wall_half_width)
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
        dx3 = 3 * self.args.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r)))
        box_size = self._box_size2D()
        fluid_mask = (r[:, 1] < box_size[1] - dx3) * (r[:, 1] > dx3)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.args.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        mask_fluid = state["tag"] == Tag.FLUID

        # set incoming fluid temperature to reference_temperature
        mask_inflow = mask_fluid * (state["r"][:, 0] < 3 * self.args.dx)
        state["T"] = jnp.where(mask_inflow, self.args.reference_temperature, state["T"])
        state["dTdt"] = jnp.where(mask_inflow, 0.0, state["dTdt"])

        # set the hot wall to hot_wall_temperature.
        mask_hot = state["tag"] == Tag.DIRICHLET_WALL  # hot wall
        state["T"] = jnp.where(mask_hot, self.args.hot_wall_temperature, state["T"])
        state["dTdt"] = jnp.where(mask_hot, 0.0, state["dTdt"])

        # set the fixed wall to reference_temperature.
        mask_solid = state["tag"] == Tag.SOLID_WALL  # fixed wall
        state["T"] = jnp.where(mask_solid, self.args.reference_temperature, state["T"])
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
            state["r"][:, 0] > self.args.bounds[0][1] - 3 * self.args.dx
        )
        state["dTdt"] = jnp.where(mask_outflow, 0.0, state["dTdt"])

        return state
