"""Taylor-Green case setup"""

import os

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.io_state import read_h5


class TGV(SimulationSetup):
    """Taylor-Green Vortex"""

    def __init__(self, args):
        super().__init__(args)

        self.args.g_ext_magnitude = 0.0
        self.args.is_bc_trick = False
        args.Vmax = 1.0
        #args.Vmax = 1.0 if not hasattr(self, "u_ref") else self.u_ref
        if self.args.p_bg_factor is None:
            # p_bg introduces oscillations at low speed
            self.args.p_bg_factor = 0.0
        if self.args.is_limiter is None:
            self.args.is_limiter = False

    def _box_size2D(self):
        return np.array([1.0, 1.0])

    def _box_size3D(self):
        return 2 * np.pi * np.array([1.0, 1.0, 1.0])

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
                f"--seed={str(self.args.seed)} --write-h5  --relax-pbc "
                f"--r0-noise-factor=0.25 --data-path=data_relaxed"
            )
            raise FileNotFoundError(f"First execute this: \n{message}")

        state = read_h5(init_path)
        return state["r"]

    def _init_pos3D(self, box_size, dx):
        # check if relaxed state for this seed exists
        # if no, run special relaxation in a box
        # independen of if, load the initial state
        return self._init_pos2D(box_size, dx)

    def _tag2D(self, r):
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.zeros(len(r), dtype=int)
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
    
    def _init_acceleration2D(self, r):
        x, y = r
        # from Transport Veocity paper by Adami et al. 2013, here Re = 100
        du = -1.0 * (-8) * jnp.pi ** 2 / 100 * jnp.cos(2.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
        dv = +1.0 * (-8) * jnp.pi ** 2 / 100 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)

        return jnp.array([du, dv])
