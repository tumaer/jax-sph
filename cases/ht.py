"Heat Transfer over a flat plate setup"


import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup


class HT(SimulationSetup):
    "Heat Transfer"

    def __init__(self, args):
        super().__init__(args)

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1.0

        # for top and bottom walls
        self.args.is_bc_trick = True

        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.05  # same as RPF
        print("g_ext_force = ", self.args.g_ext_magnitude)

        self.args.kappa = 7.313  # thermal conductivity value at 50°C
        self.args.Cp = 305.27  # specific heat at constant pressure value at 50°C
        self.args.hot_wall_temperature = 1.23  # temp corresponding to 100°C
        self.args.reference_temperature = 1.0
        self.args.hot_wall_half_width = 0.25

    def _box_size2D(self):
        dx = self.args.dx
        return np.array([1, 0.2 + (6 * dx)])

    def _box_size3D(self):
        dx = self.args.dx
        return np.array([1, 0.2 + (6 * dx), 0.5])

    def _tag2D(self, r):
        dx3 = 3 * self.args.dx
        _box_size = self._box_size2D()
        # tags: {'0': water, '1': solid wall, '2': moving wall, '3': hot wall}
        tag = jnp.zeros(len(r), dtype=int)

        mask_no_slip_wall = (r[:, 1] < dx3) + (
            r[:, 1] > (_box_size[1] - 6 * self.args.dx) + dx3
        )
        mask_hot_wall = (
            (r[:, 1] < dx3)
            * (r[:, 0] < (_box_size[0] / 2) + self.args.hot_wall_half_width)
            * (r[:, 0] > (_box_size[0] / 2) - self.args.hot_wall_half_width)
        )
        tag = jnp.where(mask_no_slip_wall, 1, tag)
        tag = jnp.where(mask_hot_wall, 3, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        u = jnp.zeros_like(r)
        return u

    def _init_velocity3D(self, r):
        return self._init_velocity2D(r)

    def _external_acceleration_fn(self, r):
        dx3 = 3 * self.args.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r))) * 2.3  # magnitude of external force field
        fluid_mask = (r[:, 1] < 1 + dx3) * (r[:, 1] > dx3)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.args.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        # set incoming fluid temperature to reference_temperature
        mask_inflow = (state["tag"] == 0) * (state["r"][:, 0] < 3 * self.args.dx)
        state["T"] = jnp.where(mask_inflow, self.args.reference_temperature, state["T"])
        state["dTdt"] = jnp.where(mask_inflow, 0, state["dTdt"])

        # set the hot wall to hot_wall_temperature.
        mask3 = state["tag"] == 3  # hot wall
        state["T"] = jnp.where(mask3, self.args.hot_wall_temperature, state["T"])
        state["dTdt"] = jnp.where(mask3, 0, state["dTdt"])

        # set the fixed wall to reference_temperature.
        mask1 = state["tag"] == 1  # fixed wall
        state["T"] = jnp.where(mask1, self.args.reference_temperature, state["T"])
        state["dTdt"] = jnp.where(mask1, 0, state["dTdt"])

        # ensure static walls have no velocity or acceleration
        mask = state["tag"][:, None] > 0
        state["u"] = jnp.where(mask, 0, state["u"])
        state["v"] = jnp.where(mask, 0, state["v"])
        state["dudt"] = jnp.where(mask, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask, 0, state["dvdt"])

        # set outlet temperature gradients to zero to avoid interaction with inflow
        # bounds[0][1] is the x-coordinate of the outlet
        mask_outflow = (state["tag"] == 0) * (
            state["r"][:, 0] > self.args.bounds[0][1] - 3 * self.args.dx
        )
        state["dTdt"] = jnp.where(mask_outflow, 0, state["dTdt"])

        return state
