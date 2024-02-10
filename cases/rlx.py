"""Particle relaxation case setup"""

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup


class Rlx(SimulationSetup):
    """Relax particles in a box"""

    def __init__(
        self,
        args,
    ):
        super().__init__(args)

        self.nx, self.ny, self.nz = [int(i) for i in args.nxnynz.split("_")]
        # length, heigth, and width of fluid domain; box with 3dx not incl.
        dx = args.dx
        self.L, self.H, self.W = self.nx * dx, self.ny * dx, self.nz * dx
        self.relax_pbc = args.relax_pbc
        self.args.p_bg_factor = 0.0

        # custom variables related only to this Simulation
        self.args.g_ext_magnitude = 0.0
        self.args.is_bc_trick = True if args.relax_pbc else False

    def _box_size2D(self):
        wall = 0 if self.relax_pbc else 6
        return (np.array([self.nx, self.ny]) + wall) * self.args.dx

    def _box_size3D(self):
        wall = 0 if self.relax_pbc else 6
        return (np.array([self.nx, self.ny, self.nz]) + wall) * self.args.dx

    def _tag2D(self, r):
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.zeros(len(r), dtype=int)

        if not self.relax_pbc:
            dx3 = 3 * self.args.dx
            cond_x = (r[:, 0] < dx3) + (r[:, 0] > self.L + dx3)
            cond_y = (r[:, 1] < dx3) + (r[:, 1] > self.H + dx3)
            cond_z = jnp.array([False] * len(r))
            if r.shape[1] == 3:
                cond_z = (r[:, 2] < dx3) + (r[:, 2] > self.W + dx3)
            mask_wall = jnp.where(cond_x + cond_y + cond_z, True, False)

            tag = jnp.where(mask_wall, 1, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return self._init_velocity2D(r)

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        if not self.relax_pbc:
            mask1 = state["tag"][:, None] == 1

            state["u"] = jnp.where(mask1, 0, state["u"])
            state["v"] = jnp.where(mask1, 0, state["v"])

            state["dudt"] = jnp.where(mask1, 0, state["dudt"])
            state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])

        return state
    
    def _init_acceleration2D(self, r):
        pass
