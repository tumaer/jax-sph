"""Dam break case setup"""

import os

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.io_state import read_h5
from jax_sph.utils import Tag, pos_init_cartesian_2d


class DB(SimulationSetup):
    """Dam Break"""

    def __init__(self, args):
        super().__init__(args)

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
        self.L = 2.0
        self.H = 1.0

        # a trick to reduce computation using PBC in z-direction
        self.box_offset = 0.1

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1.0
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.0

        # TODO: check behavior with 1, sqrt(2), and 2 for u_ref
        self.args.u_ref = (2 * self.args.g_ext_magnitude * self.H) ** 0.5
        self.args.Vmax = 2 * (self.args.g_ext_magnitude * self.H) ** 0.5

    def _box_size2D(self):
        dx, bo = self.args.dx, self.box_offset
        L_wall, H_wall = self.L_wall, self.H_wall
        return np.array([L_wall + 6 * dx + bo, H_wall + 6 * dx + bo])

    def _box_size3D(self):
        dx, bo = self.args.dx, self.box_offset
        L_wall, H_wall = self.L_wall, self.H_wall
        return np.array([L_wall + 6 * dx + bo, H_wall + 6 * dx + bo, 0.2])

    def _init_pos2D(self, box_size, dx):
        dx3 = 3 * self.args.dx
        dx6 = 6 * self.args.dx
        L_wall, H_wall = self.L_wall, self.H_wall

        is_cartesian = True

        if is_cartesian:
            r_fluid = dx3 + pos_init_cartesian_2d(np.array([self.L, self.H]), dx)
        else:
            if len(box_size) == 2:
                nx, ny = np.array((np.array([self.L, self.H]) / dx).round(), dtype=int)
                nz = 0
            else:
                raise NotImplementedError

            nxnynz = "_".join([str(s) for s in [nx, ny, nz]])
            name = "_".join([str(s) for s in [nxnynz, dx, self.args.seed, ""]])
            init_path = "data_relaxed/" + name + ".h5"

            if not os.path.isfile(init_path):
                message = (
                    f"./venv/bin/python main.py --case=Rlx --solver=SPH "
                    f"--tvf=1.0 --dim={str(self.args.dim)} "
                    f"--dx={str(dx)} --nxnynz={nxnynz} "
                    f"--seed={str(self.args.seed)} --write-h5 "
                    f"--r0-noise-factor=0.25 --data-path=data_relaxed"
                )
                raise FileNotFoundError(f"First execute this: \n{message}")

            state = read_h5(init_path)
            r_fluid = state["r"][state["tag"] == Tag.FLUID]

        # horizontal and vertical blocks
        vertical = pos_init_cartesian_2d(np.array([dx3, H_wall + dx6]), dx)
        horiz = pos_init_cartesian_2d(np.array([L_wall, dx3]), dx)

        # wall: left, bottom, right, top
        wall_l = vertical.copy()
        wall_b = horiz.copy() + np.array([dx3, 0.0])
        wall_r = vertical.copy() + np.array([L_wall + dx3, 0.0])
        wall_t = horiz.copy() + np.array([dx3, H_wall + dx3])

        res = np.concatenate([wall_l, wall_b, wall_r, wall_t, r_fluid])
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
