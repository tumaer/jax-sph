"""Dirichelt Energy computation

From: 
"Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics", Toshev et al 2024
"""
import os

import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from jax_md import space
from jax_md.partition import Sparse

from jax_sph import partition
from jax_sph.io_state import read_args, read_h5
from jax_sph.kernels import QuinticKernel, WendlandC2Kernel
from jax_sph.utils import Tag, pos_init_cartesian_2d

EPS = jnp.finfo(float).eps


def set_kernel_function(kernel, dx, dim):
    """Define the kernel function and cut-off radius."""

    if kernel == "QSK":
        kernel_fn = QuinticKernel(h=dx, dim=dim)
    elif kernel == "WC2K":
        kernel_fn = WendlandC2Kernel(h=1.3 * dx, dim=dim)

    return kernel_fn, kernel_fn.cutoff


def density_gradient_wrapper(kernel_fn):
    def density_gradient(r_ij, d_ij, mass_j):
        """Compute drho/dr."""

        # Compute unit vector and kernel gradient
        e_ij = r_ij / (d_ij + EPS)
        kernel_part_diff = kernel_fn.grad_w(d_ij)
        kernel_grad = kernel_part_diff * e_ij

        # Compute spatial density gradient
        drhodr = mass_j * kernel_grad

        return drhodr

    return density_gradient


def squared_L2_norm(prop):
    """Squared L2-norm of a property"""

    erg = jnp.sum(prop**2)

    return erg


def get_dirichlet_energy_data(path, file_name):
    """Get quiantities for calculating the Dirichlet energy from a .h5 file."""

    dirs = os.listdir(path)
    dirs = [d for d in dirs if (file_name in d)]
    dirs = sorted(dirs)[0]
    args = read_args(os.path.join(path, dirs, "args.txt"))

    files = os.listdir(os.path.join(path, dirs))
    files = [f for f in files if (".h5" in f)]
    files = sorted(files)
    quant = {}
    for i in range(len(files)):
        state = read_h5(os.path.join(path, dirs, files[i]))
        quant[i] = {
            "r": state["r"],
            "tag": state["tag"],
            "mass": state["mass"],
            "rho": state["rho"],
        }

    dx = args.dx
    dim = args.dim
    box_size = jnp.array(args.bounds, float)[:, 1]

    return quant, dim, dx, box_size


def dirichlet_energy(r, tag, mass, dim, dx, box_size, kernel_arg="QSK", is_rho=False):
    kernel_fn, r_cut = set_kernel_function(kernel_arg, dx, dim)
    displacement_fn, _ = space.periodic(side=box_size)
    N = len(r)

    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=r_cut,
        backend="jaxmd_vmap",
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
    )
    num_particles = (tag != Tag.PAD_VALUE).sum()
    neighbors = neighbor_fn.allocate(r, num_particles=num_particles)

    i_s, j_s = neighbors.idx
    r_i_s, r_j_s = r[i_s], r[j_s]
    dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)

    # compute density gradient
    density_gradient = density_gradient_wrapper(kernel_fn)
    temp = vmap(density_gradient)(dr_i_j, dist, mass[j_s])
    drhodr = ops.segment_sum(temp, i_s, N)

    # compute squared L2-norm
    drhodr_norm = vmap(squared_L2_norm, in_axes=0)(drhodr)
    # drhodr_norm = vmap(jnp.linalg.norm, in_axes=0)(drhodr)

    # compute Dirichlet energy integral, see
    dir_energy = jnp.sum(drhodr_norm) * dx**dim / 2
    dir_energy_per_particle = 0.5 * drhodr_norm.mean()

    rho = None
    if is_rho:
        # compute density
        from jax_sph.solver import rho_summation_fn

        rho = rho_summation_fn(mass, i_s, kernel_fn.w(dist), N)
        print(f"Density deviation from mean: {np.abs(rho-1).mean():.4e}")

    return dir_energy, dir_energy_per_particle, rho, drhodr, drhodr_norm


def vis_density(rho, drhodr, drhodr_norm, suffix=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].scatter(r[:, 0], r[:, 1], c=np.asarray(drhodr[:, 0]))
    ax[0, 1].scatter(r[:, 0], r[:, 1], c=np.asarray(drhodr[:, 1]))
    ax[1, 0].scatter(r[:, 0], r[:, 1], c=np.asarray(rho))
    ax[1, 1].scatter(r[:, 0], r[:, 1], c=np.asarray(drhodr_norm))

    ax[0, 0].set_title(f"drhodr_x ({drhodr[:,0].min():.4e}, {drhodr[:,0].max():.4e})")
    ax[0, 1].set_title(f"drhodr_y ({drhodr[:,1].min():.4e}, {drhodr[:,1].max():.4e})")
    ax[1, 0].set_title(f"rho ({rho.min():.4e}, {rho.max():.4e})")
    ax[1, 1].set_title(
        f"drhodr_norm ({drhodr_norm.min():.4e}, {drhodr_norm.max():.4e})"
    )

    for ax in ax.flatten():
        ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs("vis_data", exist_ok=True)
    fig.savefig(f"vis_data/drhodr_{suffix}.png")


if __name__ == "__main__":
    # Dirichlet energy of a constant density field with 1.3% density fluctuation
    dim = 2
    dx = 0.02
    box_size = jnp.array([1, 1])
    r = pos_init_cartesian_2d(box_size, dx)
    p = 0.001
    # p=0.001 for 1.3% density fluctuation; 0.01 for 13% density fluctuation.
    # Corresponds to E_dirichlet=0.00079 and 0.079, respectively
    r = jnp.array(
        [
            r[:, 0] + p * jnp.sin(6.2831 * r[:, 0]),
            r[:, 1] + p * jnp.cos(6.2831 * r[:, 1]),
        ]
    ).T
    N = r.shape[0]
    tag = jnp.zeros(N, dtype=int)
    mass = jnp.ones(N) * dx ** (dim)  # mass = rho * dx**dim
    out = dirichlet_energy(r, tag, mass, dim, dx, box_size, "QSK", is_rho=True)
    edirichlet, _, rho, drhodr, drhodr_norm = out
    vis_density(rho, drhodr, drhodr_norm, suffix="0001")
    print("The Dirichlet energy of a constant density field is:", edirichlet)

    # Dirichlet energy of a constant density field with 13% density fluctuation
    dim = 2
    dx = 0.02
    box_size = jnp.array([1, 1])
    r = pos_init_cartesian_2d(box_size, dx)
    p = 0.01
    # p=0.001 for 1.3% density fluctuation; 0.01 for 13% density fluctuation.
    # Corresponds to E_dirichlet=0.00079 and 0.079, respectively
    r = jnp.array(
        [
            r[:, 0] + p * jnp.sin(6.2831 * r[:, 0]),
            r[:, 1] + p * jnp.cos(6.2831 * r[:, 1]),
        ]
    ).T
    N = r.shape[0]
    tag = jnp.zeros(N, dtype=int)
    mass = jnp.ones(N) * dx ** (dim)  # mass = rho * dx**dim
    out = dirichlet_energy(r, tag, mass, dim, dx, box_size, "QSK", is_rho=True)
    edirichlet, _, rho, drhodr, drhodr_norm = out
    vis_density(rho, drhodr, drhodr_norm, suffix="001")
    print("The Dirichlet energy of a constant density field is:", edirichlet)

    # Dirichlet energy of 2D TGV at time stamp 50
    quant, dim, dx, box_size = get_dirichlet_energy_data(
        "./data_valid/tgv2d_tvf", "2D_TGV_SPH"
    )
    quant = quant[49]
    r, tag, mass = quant["r"], quant["tag"], quant["mass"]
    out = dirichlet_energy(r, tag, mass, dim, dx, box_size, "QSK", is_rho=True)
    edirichlet, _, rho, drhodr, drhodr_norm = out
    vis_density(rho, drhodr, drhodr_norm, suffix="tgv2d")
    print("The Dirichlet energy of the 2D TGV at time stamp 50 is:", edirichlet)
