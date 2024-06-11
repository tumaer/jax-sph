"""Reverse-Poiseuille flow case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup


class RPF(SimulationSetup):
    """Reverse Poiseuille Flow"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self, n_walls):
        return np.array([1.0, 2.0])

    def _box_size3D(self, n_walls):
        return np.array([1.0, 2.0, 0.5])

    def _init_walls_2d(self):
        pass

    def _init_walls_3d(self):
        pass

    def _init_velocity2D(self, r):
        u = jnp.zeros_like(r)
        return u

    def _init_velocity3D(self, r):
        return self._init_velocity2D(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)

        x_force = jnp.where(r[:, 1] > 1.0, -1.0, 1.0)
        res = res.at[:, 0].set(x_force)
        return res * self.case.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        return state
