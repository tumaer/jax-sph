"""Dam break case setup"""

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_box_2d, pos_init_cartesian_2d


class DB(SimulationSetup):
    """Dam Break"""

    def __init__(self, args):
        super().__init__(args)

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1.0
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.0

        # height and length are as presented in
        # A. Colagrossi, "Numerical simulation of interfacial flows by smoothed
        # particle hydrodynamics", J. Comput. Phys. 191 (2) (2003) 448–475.
        # |---------------------------|
        # |                           |
        # |-------------              |
        # |<     L    >| H            |  H_wall
        # | --------------------------|
        #  <          L_wall         >

        self.L_wall = 5.366
        self.H_wall = 2.0
        self.L = 2.0  # length
        self.H = 1.0  # height
        self.W = 0.2  # width

        # a trick to reduce computation using PBC in z-direction
        self.box_offset = 0.1

        # TODO: check behavior with 1, sqrt(2), and 2 for u_ref
        self.args.u_ref = (2 * self.args.g_ext_magnitude * self.H) ** 0.5
        self.args.Vmax = 2 * (self.args.g_ext_magnitude * self.H) ** 0.5

        # relaxation configurations
        if self.args.mode == "rlx":
            self.L_wall = self.L
            self.H_wall = self.H
            self._set_default_rlx()

        if args.r0_type == "relaxed":
            self._load_only_fluid = True

    def _box_size2D(self):
        dx, bo = self.args.dx, self.box_offset
        return np.array([self.L_wall + 6 * dx + bo, self.H_wall + 6 * dx + bo])

    def _box_size3D(self):
        dx, bo = self.args.dx, self.box_offset
        return np.array([self.L_wall + 6 * dx + bo, self.H_wall + 6 * dx + bo, self.W])

    def _init_pos2D(self, box_size, dx):
        if self.args.r0_type == "cartesian":
            r_fluid = 3 * dx + pos_init_cartesian_2d(np.array([self.L, self.H]), dx)
        else:
            r_fluid = self._get_relaxed_r0(None, dx)

        walls = pos_box_2d(self.L_wall, self.H_wall, dx)
        res = np.concatenate([walls, r_fluid])
        return res

    def _init_pos3D(self, box_size, dx):
        # cartesian coordinates in z
        Lz = box_size[2]
        zs = np.arange(0, Lz, dx) + 0.5 * dx

        # extend 2D points to 3D
        xy = self._init_pos2D(box_size, dx)
        xy_ext = np.hstack([xy, np.ones((len(xy), 1))])

        r_xyz = np.vstack([xy_ext * [1, 1, z] for z in zs])
        return r_xyz

    def _tag2D(self, r):
        dx3 = 3 * self.args.dx
        mask_left = jnp.where(r[:, 0] < dx3, True, False)
        mask_bottom = jnp.where(r[:, 1] < dx3, True, False)
        mask_right = jnp.where(r[:, 0] > self.L_wall + dx3, True, False)
        mask_top = jnp.where(r[:, 1] > self.H_wall + dx3, True, False)

        mask_wall = mask_left + mask_bottom + mask_right + mask_top

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
        res = res.at[:, 1].set(-self.args.g_ext_magnitude)
        return res

    def _boundary_conditions_fn(self, state):
        mask_wall = state["tag"] == Tag.SOLID_WALL

        state["u"] = jnp.where(mask_wall[:, None], 0.0, state["u"])
        state["v"] = jnp.where(mask_wall[:, None], 0.0, state["v"])
        state["dudt"] = jnp.where(mask_wall[:, None], 0.0, state["dudt"])
        state["dvdt"] = jnp.where(mask_wall[:, None], 0.0, state["dvdt"])
        state["p"] = jnp.where(mask_wall, 0.0, state["p"])
        return state
