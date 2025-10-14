"""Taylor-Green case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup


class TGV(SimulationSetup):
    """Taylor-Green Vortex"""

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
        return np.array([1.0, 1.0])

    def _box_size3D(self, n_walls):
        return 2 * np.pi * np.array([1.0, 1.0, 1.0])

    def _init_walls_2d(self):
        pass

    def _init_walls_3d(self):
        pass

    def _init_velocity2D(self, r):
        x, y = r
        # from Transport Veocity paper by Adami et al. 2013
        u = -1.0 * jnp.cos(2.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
        v = +1.0 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)

        return jnp.array([u, v])

    def _init_velocity3D(self, r):
        x, y, z = r
        z_term = jnp.cos(z)
        u = +jnp.sin(x) * jnp.cos(y) * z_term
        v = -jnp.cos(x) * jnp.sin(y) * z_term
        w = 0.0
        return jnp.array([u, v, w])

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        return state
