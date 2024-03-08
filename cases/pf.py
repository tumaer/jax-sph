"""Poiseuille flow case setup"""

import os

import jax.numpy as jnp
import numpy as np

from jax_sph.case_setup import SimulationSetup
from jax_sph.io_state import read_h5
from jax_sph.utils import pos_init_cartesian_2d


class PF(SimulationSetup):
    """Poiseuille Flow"""

    def __init__(self, args):
        super().__init__(args)

        # Setup from "A generalized wall boundary condition [...]", Adami 2012
        # custom variables related only to this Simulation
        self.args.args.u_ref = 1.25

        if self.args.g_ext_magnitude is None:
            self.args.g_ext_magnitude = 1000
        self.args.is_bc_trick = True
        if self.args.p_bg_factor is None:
            self.args.p_bg_factor = 0.0

    def _box_size2D(self):
        return np.array([0.4, 1 + 6 * self.args.dx])

    def _box_size3D(self):
        return np.array([0.4, 1 + 6 * self.args.dx, 0.4])

    def disable_init_pos2D(self, box_size, dx):
        if len(box_size) == 2:
            nx, ny = np.array((box_size / dx).round(), dtype=int)
            ny -= 6
            nz = 0
        else:
            nx, ny, nz = np.array((box_size / dx).round(), dtype=int)
            ny -= 6

        nxnynz = "_".join([str(s) for s in [nx, ny, nz]])
        # TODO: PBC
        # name = '_'.join([str(s) for s in [nxnynz, dx, self.args.seed, 'pbc']])
        name = "_".join([str(s) for s in [nxnynz, dx, self.args.seed, ""]])
        init_path = "data_relaxed/" + name + ".h5"

        if not os.path.isfile(init_path):
            message = (
                f"./venv/bin/python main.py --case=Rlx --solver=SPH "
                f"--tvf=1.0 --dim={str(self.args.dim)} "
                f"--dx={str(dx)} --nxnynz={nxnynz} "
                # TODO: pbc
                # f"--seed={str(self.args.seed)} --write-h5  --relax-pbc"
                f"--seed={str(self.args.seed)} --write-h5 "
                f"--r0-noise-factor=0.25 --data-path=data_relaxed"
            )
            raise FileNotFoundError(f"First execute this: \n{message}")
        # TODO: pbc
        state = read_h5(init_path)
        # r_fluid = state['r'] + np.array([0, 3 * dx])
        r_fluid = state["r"] + np.array([-3 * dx, 0])
        r_fluid = r_fluid[state["tag"] == 0]

        # wall: bottom and top
        wall_b = pos_init_cartesian_2d(np.array([box_size[0], 3 * dx]), dx)
        wall_t = wall_b.copy() + np.array([0, 1 + 3 * dx])

        res = np.concatenate([wall_b, wall_t, r_fluid])
        return res

    def _tag2D(self, r):
        # tags: {'0': water, '1': solid wall, '2': moving wall}
        tag = jnp.zeros(len(r), dtype=int)
        dx3 = 3 * self.args.dx

        mask_wall = (r[:, 1] < dx3) + (r[:, 1] > 1 + dx3)
        tag = jnp.where(mask_wall, 1, tag)
        return tag

    def _tag3D(self, r):
        return self._tag2D(r)

    def _init_velocity2D(self, r):
        return jnp.zeros_like(r)

    def _init_velocity3D(self, r):
        return jnp.zeros_like(r)

    def _external_acceleration_fn(self, r):
        dx3 = 3 * self.args.dx
        res = jnp.zeros_like(r)
        x_force = jnp.ones((len(r)))
        fluid_mask = (r[:, 1] < 1 + dx3) * (r[:, 1] > dx3)
        x_force = jnp.where(fluid_mask, x_force, 0)
        res = res.at[:, 0].set(x_force)
        return res * self.args.g_ext_magnitude

    def _boundary_conditions_fn(self, state):
        mask1 = state["tag"][:, None] == 1

        state["u"] = jnp.where(mask1, 0, state["u"])
        state["v"] = jnp.where(mask1, 0, state["v"])

        state["dudt"] = jnp.where(mask1, 0, state["dudt"])
        state["dvdt"] = jnp.where(mask1, 0, state["dvdt"])

        return state
