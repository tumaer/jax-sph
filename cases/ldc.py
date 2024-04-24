"""Lid-driven cavity case setup"""


import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class LDC(SimulationSetup):
    """Lid-Driven Cavity"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # custom variables related only to this Simulation
        if self.case.dim == 2:
            self.u_lid = jnp.array([self.special.u_x_lid, 0.0])
        elif self.case.dim == 3:
            self.u_lid = jnp.array([self.special.u_x_lid, 0.0, 0.0])

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

    def _tag2D(self, r):
        box_size = self._box_size2D() if self.case.dim == 2 else self._box_size3D()

        mask_lid = r[:, 1] > (box_size[1] - 3 * self.case.dx)
        r_centered_abs = jnp.abs(r - r.mean(axis=0))
        mask_water = jnp.where(r_centered_abs.max(axis=1) < 0.5, True, False)
        tag = jnp.full(len(r), Tag.SOLID_WALL, dtype=int)
        tag = jnp.where(mask_water, Tag.FLUID, tag)
        tag = jnp.where(mask_lid, Tag.MOVING_WALL, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

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

    def _init_density(self, r):
        return jnp.ones(jnp.shape(r)[0:1]) * self.case.rho_ref
