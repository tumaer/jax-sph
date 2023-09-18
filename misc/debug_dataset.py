"""Experiment with datasets"""

import json
import os

import jax.numpy as jnp
from jax import grad, jit
from jax.config import config
from jax_md import energy, space  # , partition, quantity, simulate,

from jax_sph.io_state import write_h5, write_vtk

# from argparse import Namespace
# from functools import partial


config.update("jax_enable_x64", True)


# # set up simulation
# args_dict = {
#     "dim": 3,
#     "dx": 1.4,  # 1.37820
#     "dt": 5e-3,
#     "t_end": 10.,
#     "data_path": "dataset_lj/3D_LJ_JAXMD_0",
#     "periodic_boundary_conditions": [True, True, True],
#     "bounds": [[0.0, 5.0], [0.0, 5.0], [0.0, 5.0]],
#     "seed": 0,
# }
# args_dict["sequence_length"] = int(args_dict["t_end"] / args_dict["dt"])

# # write configuration to file
# os.makedirs(args_dict["data_path"], exist_ok=True)
# with open(os.path.join(args_dict["data_path"], 'args.txt'), 'w') as f:
#     json.dump(args_dict, f)


# box_size = jnp.array(args_dict["bounds"])[:, 1]
# displacement, shift = space.periodic(box_size)

# R = jnp.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [2.5, 0.5, 0.5]]) * args_dict["dx"]
# tag = jnp.array([0, 0, 0], dtype=jnp.int32)

# N = R.shape[0]
# phi = 1 / args_dict["dx"] ** 3
# print(f'Created a system of {N} LJ particles with number density {phi:.3f}')

# neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
#     displacement,
#     box_size,
#     r_cutoff=3.0,
#     dr_threshold=1.,
#     capacity_multiplier=1.25,
#     format=partition.Sparse
# )

# init, apply = simulate.nvt_nose_hoover(energy_fn, shift, args_dict["dt"], kT=1.2, )
# key = random.PRNGKey(args_dict["seed"])
# nbrs = neighbor_fn.allocate(R)
# state = init(key, R, neighbor=nbrs)

# @jit
# def advance(state, nbrs):
#   nbrs = nbrs.update(state.position)
#   return apply(state, neighbor=nbrs), nbrs

# # Run once to make sure the JIT cache is occupied.
# for step in range(args_dict["sequence_length"] + 2):
#     state, nbrs = advance(state, nbrs)
#     if nbrs.did_buffer_overflow:
#         raise ValueError('Neighbor list overflowed. Try larger capacity_multiplier.')

#     digits = len(str(args_dict["sequence_length"]))
#     step_str = str(step).zfill(digits)
#     name = 'traj_' + step_str

#     state_dict = {
#         'r': state.position, 'v': state.velocity, 'f': state.force, 'tag': tag
#     }
#     # write_h5:
#     path = os.path.join(args_dict["data_path"], name + ".h5")
#     write_h5(state_dict, path)
#     # write_vtk:
#     path = os.path.join(args_dict["data_path"], name + ".vtk")
#     write_vtk(state_dict, path)

###############################################################################

# PARTICLE_COUNT = 3
# f32 = jnp.float32
# spatial_dimension = 3
# sy_steps = [1, 3, 5, 7]


# key = random.PRNGKey(0)

# box_size = quantity.box_size_at_number_density(PARTICLE_COUNT,
#                                                f32(1.2),
#                                                spatial_dimension)
# displacement_fn, shift_fn = space.periodic(box_size)

# bonds_i = jnp.arange(PARTICLE_COUNT)
# bonds_j = jnp.roll(bonds_i, 1)
# bonds = jnp.stack([bonds_i, bonds_j])

# E = energy.simple_spring_bond(displacement_fn, bonds)

# invariant = partial(simulate.nvt_nose_hoover_invariant, E)

# key, pos_key, vel_key, T_key, masses_key = random.split(key, 5)

# R = box_size * random.uniform(pos_key, (PARTICLE_COUNT, spatial_dimension), dtype=f32)
# T = random.uniform(T_key, (), minval=0.3, maxval=1.4, dtype=f32)
# mass = 1 + random.uniform(masses_key, (PARTICLE_COUNT,), dtype=f32)
# init_fn, apply_fn = simulate.nvt_nose_hoover(E, shift_fn, 1e-3, T)
# apply_fn = jit(apply_fn)

# state = init_fn(vel_key, R, mass=mass)

# initial = invariant(state, T)

# for i in range(10):
#     state = apply_fn(state)

#     print(i)


###############################################################################

# set up simulation
args_dict = {
    "dim": 3,
    "dx": 1.4,  # 1.37820
    "dt": 0.01,
    "t_end": 20.0,
    "data_path": "dataset_hook2/3D_Hook_JAXMD_0",
    "periodic_boundary_conditions": [True, True, True],
    "bounds": [[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]],
    "seed": 0,
}
args_dict["sequence_length"] = int(args_dict["t_end"] / args_dict["dt"])

# write configuration to file
os.makedirs(args_dict["data_path"], exist_ok=True)
with open(os.path.join(args_dict["data_path"], "args.txt"), "w") as f:
    json.dump(args_dict, f)

box_size = jnp.array(args_dict["bounds"])[:, 1]
displacement_fn, shift_fn = space.periodic(box_size)
bonds_i = jnp.arange(2)
bonds_j = jnp.roll(bonds_i, 1)
bonds = jnp.stack([bonds_i, bonds_j])
E = energy.simple_spring_bond(displacement_fn, bonds, length=0.8, epsilon=1.0)


# state = {
#     "r": jnp.array([[0.5, 1.5, 1.5], [1.5, 0.5, 1.5], [2.5, 0.5, 1.5]]),
#     "v": jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]]),
#     "f": jnp.zeros((3, 3)),
#     "tag": jnp.array([0, 0, 0], dtype=jnp.int32)
# }

state = {
    "r": jnp.array([[1.0, 1.5, 1.5], [2.0, 1.5, 1.5]]),
    "v": jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "f": jnp.zeros((2, 3)),
    "tag": jnp.array([0, 0], dtype=jnp.int32),
}


@jit
def advance(state):
    vel_half = state["v"] + 0.5 * state["f"] * args_dict["dt"]
    state["r"] = shift_fn(state["r"], vel_half * args_dict["dt"])

    state["f"] = -grad(E)(state["r"])

    state["v"] = vel_half + 0.5 * state["f"] * args_dict["dt"]
    return state


for step in range(args_dict["sequence_length"] + 2):
    state = advance(state)

    digits = len(str(args_dict["sequence_length"]))
    step_str = str(step).zfill(digits)
    name = "traj_" + step_str

    if step % 1 == 0:
        # write_h5:
        path = os.path.join(args_dict["data_path"], name + ".h5")
        write_h5(state, path)
        # write_vtk:
        path = os.path.join(args_dict["data_path"], name + ".vtk")
        write_vtk(state, path)
