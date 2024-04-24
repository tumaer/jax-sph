"""Rayleigh-Taylor Instability case setup"""


import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class RTI(SimulationSetup):
    """Rayleigh-Taylor Instability"""

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
        return np.array([1.0, 2.0])

    def _box_size3D(self):
        return np.array([1.0, 2.0, 1, 0])

    def _tag2D(self, r):
        box_size = self._box_size2D() if self.case.dim == 2 else self._box_size3D()

        dx3 = 3 * self.case.dx
        mask_bottom = jnp.where(r[:, 1] < dx3, True, False)
        mask_top = jnp.where(r[:, 1] > box_size[1] - dx3, True, False)

        mask_wall = mask_bottom + mask_top

        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        tag = jnp.where(mask_wall, Tag.SOLID_WALL, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)
        res = res.at[:, 1].set(-self.case.g_ext_magnitude)
        return res

    def _boundary_conditions_fn(self, state):
        mask_wall = state["tag"] == Tag.SOLID_WALL

        state["u"] = jnp.where(mask_wall[:, None], 0.0, state["u"])
        state["v"] = jnp.where(mask_wall[:, None], 0.0, state["v"])
        state["dudt"] = jnp.where(mask_wall[:, None], 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask_wall[:, None], 0.0, state["dvdt"])
        state["p"] = jnp.where(mask_wall, 0.0, state["p"])
        return state

    def _init_density(self, r):
        rho_ref = self.case.rho_ref
        rho = jnp.where(
            r[:, 1] > 1 - 0.15 * jnp.sin(2 * jnp.pi * r[:, 0]), rho_ref[1], rho_ref[0]
        )
        return rho
