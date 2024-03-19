"""Taylor-Green case setup"""


import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag


class TGV(SimulationSetup):
    """Taylor-Green Vortex"""

    def __init__(self, args):
        super().__init__(args)

        self.args.g_ext_magnitude = 0.0
        self.args.is_bc_trick = False
        if self.args.p_bg_factor is None:
            # p_bg introduces oscillations at low speed
            self.args.p_bg_factor = 0.0

        # relaxation configurations
        if self.args.mode == "rlx":
            self._set_default_rlx()

        if args.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        return np.array([1.0, 1.0])

    def _box_size3D(self):
        return 2 * np.pi * np.array([1.0, 1.0, 1.0])

    def _tag2D(self, r):
        tag = jnp.full(len(r), Tag.FLUID, dtype=int)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

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
