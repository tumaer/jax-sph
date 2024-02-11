"""Reverse-Poiseuille flow case setup"""

import os

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.io_state import read_h5


class RPF(SimulationSetup):
    """Reverse Poiseuille Flow"""

    def __init__(self, args):
        super().__init__(args)

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1.0
        self.args.is_bc_trick = False
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.05
        self.args.kappa = 0.0 
        self.args.Cp = 0.0

        print("g_ext_force = ", self.args.g_ext_magnitude)

    def _box_size2D(self):
        return np.array([1.0, 2.0])

    def _box_size3D(self):
        return np.array([1.0, 2.0, 0.5])

    def _init_pos2D(self, box_size, dx):
        if len(box_size) == 2:
            nx, ny = np.array((box_size / dx).round(), dtype=int)
            nz = 0
        else:
            nx, ny, nz = np.array((box_size / dx).round(), dtype=int)

        nxnynz = "_".join([str(s) for s in [nx, ny, nz]])
        name = "_".join([str(s) for s in [nxnynz, dx, self.args.seed, "pbc"]])
        init_path = "data_relaxed/" + name + ".h5"

        if not os.path.isfile(init_path):
            message = (
                f"./venv/bin/python main.py --case=Rlx --solver=SPH "
                f"--tvf=1.0 --dim={str(self.args.dim)} "
                f"--dx={str(self.args.dx)} --nxnynz={nxnynz} "
                f"--seed={str(self.args.seed)} --write-h5  --relax-pbc"
                f"--r0-noise-factor=0.25 --data-path=data_relaxed"
            )
            raise FileNotFoundError(f"First execute this: \n{message}")

        state = read_h5(init_path)
        return state["r"]

    def _init_pos3D(self, box_size, dx):
        return self._init_pos2D(box_size, dx)

    def _tag2D(self, r):
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.zeros(len(r), dtype=int)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        u = jnp.zeros_like(r)

        # # try to initialize with the analytical solution
        # u_x_lower = + 4 * (0.25 - (r[1] - 0.5) ** 2)
        # u_x_upper = - 4 * (0.25 - (r[1] - 1.5) ** 2)
        # u_x = 11.3 * jnp.where(r[1] > 1, u_x_upper, u_x_lower)
        # u = u.at[0].set(u_x)
        return u

    def _init_velocity3D(self, r):
        return self._init_velocity2D(r)

    def _external_acceleration_fn(self, r):
        res = jnp.zeros_like(r)

        x_force = jnp.where(r[:, 1] > 1.0, -1.0, 1.0)
        res = res.at[:, 0].set(x_force)
        return res * self.args.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        return state
