"""Lid-driven cavity case setup"""

import os

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.io_state import read_h5


class LDC(SimulationSetup):
    """Lid-Driven Cavity"""

    def __init__(self, args):
        super().__init__(args)

        # custom variables related only to this Simulation
        if args.dim == 2:
            self.u_lid = jnp.array([1.0, 0.0])
        elif args.dim == 3:
            self.u_lid = jnp.array([1.0, 0.0, 0.0])

        args.Vmax = 1.0
        args.c0 = 10 * args.Vmax
        self.args.g_ext_magnitude = 0.0
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.0
        self.args.kappa = 0.0 
        self.args.Cp = 0.0

    def _box_size2D(self):
        return np.ones((2,)) + 6 * self.args.dx

    def _box_size3D(self):
        dx6 = 6 * self.args.dx
        return np.array([1 + dx6, 1 + dx6, 0.5])

    def _init_pos2D(self, box_size, dx):
        if len(box_size) == 2:
            nx, ny = np.array((box_size / dx).round(), dtype=int)
            nz = 0
        else:
            nx, ny, nz = np.array((box_size / dx).round(), dtype=int)
        nx -= 6
        ny -= 6

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
        res = state["r"]

        if len(box_size) == 3:
            # remove walls in z-direction from relaxation in a box
            res = res - np.array([0, 0, 3 * dx])
            mask_wall = state["tag"] == 1
            mask_wall_z = (res[:, 2] < 0) + (res[:, 2] > nz * dx)
            res = res[jnp.logical_not(mask_wall * mask_wall_z)]

        return res

    def _init_pos3D(self, box_size, dx):
        return self._init_pos2D(box_size, dx)

    def _tag2D(self, r):
        mask_lid = jnp.where(r[:, 1] > 1 + 3 * self.args.dx, True, False)
        r_centered_abs = jnp.abs(r - r.mean(axis=0))
        mask_water = jnp.where(r_centered_abs.max(axis=1) < 0.5, True, False)
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.ones(len(r), dtype=int)
        tag = jnp.where(mask_water, 0, tag)
        tag = jnp.where(mask_lid, 2, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        """Somewhat better initial velocity field for 2D LDC. The idea is to
        mix up the particles and somewhat resemble the stationary solution"""

        u = jnp.zeros_like(r)

        # x, y = r[0], r[1]
        # dx_3 = self.args.dx * 3
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
        mask1 = state["tag"][:, None] == 1
        mask2 = state["tag"][:, None] == 2

        state["u"] = jnp.where(mask1, 0, state["u"])
        state["v"] = jnp.where(mask1, 0, state["v"])
        state["u"] = jnp.where(mask2, self.u_lid, state["u"])
        state["v"] = jnp.where(mask2, self.u_lid, state["v"])

        state["dudt"] = jnp.where(mask1, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])
        state["dudt"] = jnp.where(mask2, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask2, 0, state["dvdt"])

        return state
