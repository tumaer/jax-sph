import argparse

from jax.config import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import jit

from jax_sph import partition
from jax_sph.jax_md import space


def pos_init_cartesian_3d(box_size, dx, noise_std_factor=0.3333):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    np.random.seed(0)
    r += np.random.randn(*r.shape) * dx * noise_std_factor
    r = r % box_size  # project back into unit box
    return r


def update_wrapper(neighbors_old, r_new):
    neighbors_new = neighbors_old.update(r_new)
    return neighbors_new


def compute_neighbors(args):
    Nx = args.Nx
    mode = args.mode
    nl_backend = args.nl_backend
    num_partitions = args.num_partitions
    print(f"Start with Nx={Nx}, mode={mode}, backend={nl_backend}")

    dx = 1 / Nx
    box_size = np.array([1.0, 1.0, 1.0])
    r = pos_init_cartesian_3d(box_size, dx)

    displacement_fn, _ = space.periodic(side=box_size)
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=3 * dx,
        backend=nl_backend,
        dr_threshold=0.0,
        capacity_multiplier=1.25,
        mask_self=False,
        format=partition.NeighborListFormat.Sparse,
        num_particles_max=r.shape[0],
        num_partitions=num_partitions,
        pbc=np.array([True, True, True]),
    )
    current_num_particles = r.shape[0]
    neighbors = neighbor_fn.allocate(r, num_particles=current_num_particles)

    if mode == "update":
        updater = jit(update_wrapper)
        neighbors = updater(neighbors, r)

    print(f"Finish with {r.shape[0]} particles and {neighbors.idx.shape[1]} edges!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="update", choices=["allocate", "update"])
    parser.add_argument("--num-partitions", type=int, default=8)
    parser.add_argument("--Nx", type=int, default=30, help="alternative to --dx")
    parser.add_argument(
        "--nl-backend",
        default="jaxmd_scan",
        choices=["jaxmd_vmap", "jaxmd_scan", "matscipy"],
        help="Which backend to use for neighbor list",
    )
    args = parser.parse_args()

    compute_neighbors(args)
