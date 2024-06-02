"""Dam break case setup"""

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_sph.case_setup import SimulationSetup
from jax_sph.utils import Tag, pos_box_2d, pos_init_cartesian_2d


class DB(SimulationSetup):
    """Dam Break"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # height and length are as presented in
        # A. Colagrossi, "Numerical simulation of interfacial flows by smoothed
        # particle hydrodynamics", J. Comput. Phys. 191 (2) (2003) 448–475.
        # |---------------------------|
        # |                           |
        # |-------------              |
        # |<     L    >| H            |  H_wall
        # | --------------------------|
        #  <          L_wall         >

        # relaxation configurations
        if self.case.mode == "rlx":
            self.special.L_wall = self.special.L
            self.special.H_wall = self.special.H
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = True

    def _box_size2D(self):
        dx, bo = self.case.dx, self.special.box_offset
        return np.array(
            [self.special.L_wall + 6 * dx + bo, self.special.H_wall + 6 * dx + bo]
        )

    def _box_size3D(self):
        dx, bo = self.case.dx, self.box_offset
        sp = self.special
        return np.array([sp.L_wall + 6 * dx + bo, sp.H_wall + 6 * dx + bo, sp.W])

    def _init_pos2D(self, box_size, dx):
        sp = self.special
        if self.case.r0_type == "cartesian":
            r_fluid = 3 * dx + pos_init_cartesian_2d(jnp.array([sp.L, sp.H]), dx)
        elif self.solver.multiphase and self.case.r0_type == "cartesian":
            r_fluid = 3 * dx + pos_init_cartesian_2d(
                jnp.array([sp.L_wall, sp.H_wall]), dx
            )
        else:
            r_fluid = self._get_relaxed_r0(None, dx)

        walls = pos_box_2d(sp.L_wall, sp.H_wall, dx)
        res = jnp.concatenate([walls, r_fluid])
        return res

    def _init_pos3D(self, box_size, dx):
        # cartesian coordinates in z
        Lz = box_size[2]
        zs = jnp.arange(0, Lz, dx) + 0.5 * dx

        # extend 2D points to 3D
        xy = self._init_pos2D(box_size, dx)
        xy_ext = jnp.hstack([xy, jnp.ones((len(xy), 1))])

        r_xyz = jnp.vstack([xy_ext * [1, 1, z] for z in zs])
        return r_xyz

    def _tag2D(self, r):
        dx3 = 3 * self.case.dx
        mask_left = jnp.where(r[:, 0] < dx3, True, False)
        mask_bottom = jnp.where(r[:, 1] < dx3, True, False)
        mask_right = jnp.where(r[:, 0] > self.special.L_wall + dx3, True, False)
        mask_top = jnp.where(r[:, 1] > self.special.H_wall + dx3, True, False)

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
