"""Taylor-Green case setup"""


import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_init_cartesian_2d, pos_init_cartesian_3d


class OD(SimulationSetup):
    """Oscillating Droplet"""

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
    
    def _init_pos2D(self, box_size, dx, n_walls):
        # initialize fluid phase
        r = np.array([0.25, 0.25]) + pos_init_cartesian_2d(
            np.array([0.5, 0.5]), dx
        )

        # set tags
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    def _init_pos3D(self, box_size, dx, n_walls):
        # initialize fluid phase
        r = np.array([0.25, 0.25, 0.25]) + pos_init_cartesian_3d(
            np.array([0.5, 0.5, 0.5]), dx
        )

        # set tags
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return r, tag

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        return jnp.zeros_like(r)

    def _boundary_conditions_fn(self, state):
        return state
