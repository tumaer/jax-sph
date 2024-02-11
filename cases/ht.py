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
        args.u_ref = 1.0
        args.Vmax = args.u_ref
        args.c0 = args.Vmax * 10
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
        # set incoming fluid temperature to T0 (reference temperature)
        mask0 = (state["tag"] == 0) * (state["r"][:, 0] < 3 * self.args.dx)
        state["T"] = jnp.where(mask0, self.args.reference_temperature, state["T"])
        # set the fixed wall at reference_temperature.
        state["dTdt"] = jnp.where(
            mask0, 0, state["dTdt"]
        )  # no change in temperature for the fixed wall

        mask3 = state["tag"] == 3  # hot wall
        # set the hot wall at hot_wall_temperature.
        state["T"] = jnp.where(mask3, self.args.hot_wall_temperature, state["T"])
        state["dTdt"] = jnp.where(
            mask3, 0, state["dTdt"]
        )  # no change in temperature for the hot wall

        mask1 = state["tag"] == 1  # fixed wall
        # set the fixed wall at reference_temperature.
        state["T"] = jnp.where(mask1, self.args.reference_temperature, state["T"])
        state["dTdt"] = jnp.where(
            mask1, 0, state["dTdt"]
        )  # no change in temperature for the fixed wall

        mask1 = (state["tag"][:, None] == 1) + (
            state["tag"][:, None] == 3
        )  # for paraview (fixed and hot)
        state["u"] = jnp.where(mask1, 0, state["u"])
        state["v"] = jnp.where(mask1, 0, state["v"])
        state["dudt"] = jnp.where(mask1, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])

        mask0 = (state["tag"] == 0) * (state["r"][:, 0] > self.args.bounds[0][1] - 3 * self.args.dx)
                # bounds[0][1] is the x-coordinate of the outlet
        state["dTdt"]  = jnp.where(mask0, 0, state["dTdt"] )


        return state