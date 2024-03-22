"""Dirichelt Energy computation

From: "Neural SPH: Improved Neural Modeling of Lagrangian Fluid Dynamics", Toshev et al 2024
"""
import os
import numpy as np
import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space
from jax_md.partition import Sparse
from jax_sph import partition
from jax_sph.io_state import read_args, read_h5
from jax_sph.kernels import QuinticKernel, WendlandC2Kernel
from jax_sph.utils import Tag, pos_init_cartesian_2d

EPS = jnp.finfo(float).eps

def kernel_function(kernel, dx, dim):
    "Define the kernel function and cut-off radius"

    if kernel == "QSK":
        r_cut = 3
        kernel_fn = QuinticKernel(h=dx, dim=dim)
    elif kernel == "WC2K":
        r_cut = 2.6
        kernel_fn = WendlandC2Kernel(h=1.3 * dx, dim=dim)

    return kernel_fn, r_cut

def density_gradient_wrapper(kernel_fn):
    def density_gradient(r_ij, d_ij, mass_j):
        "Compute drho/dr"

        # Compute unit vector and kernel gradient
        e_ij = r_ij / (d_ij + EPS)
        kernel_part_diff = kernel_fn.grad_w_analytical(d_ij)
        kernel_grad = kernel_part_diff * (e_ij)

        # Compute spatial density gradient
        drhodr = mass_j * kernel_grad

        return drhodr
    return density_gradient

def squared_L2_norm(prop):
    "Squared L2-norm of a property"

    erg = jnp.sum(prop**2)

    return erg

def get_dirichlet_energy_data(path, file_name):
    "A function for importing the quiantities necessary to calculate the Dirichlet energy from a .h5 file"

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
        quant[str(i)] = {"r": state["r"], "tag": state["tag"], "mass": state["mass"]}

    dx = args.dx
    dim = args.dim
    box_size = jnp.array(args.bounds, int)[:,1]

    return quant, dim, dx, box_size

def dirichlet_energy(r, tag, mass, dim, dx, box_size, kernel_arg="QSK"):

    kernel_fn, r_cut = kernel_function(kernel_arg, dx, dim)
    displacement_fn, shift_fn = space.periodic(side=box_size)
    N = len(r)

    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=r_cut * dx,
        backend='jaxmd_vmap',
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=r.shape[0],
        num_partitions=1,
        pbc=np.array([True, True, True]), # TODO: check whether this is generally true,
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

    # compute Dirichlet energy integral, see
    dir_energy = jnp.sum(drhodr_norm) * dx**2 /2

    return dir_energy

if __name__ == "__main__":

    # Dirichlet energy of 2D TGV at time stamp 50
    quant, dim, dx, box_size = get_dirichlet_energy_data("./data_valid/tgv2d_Rie", "2D_TGV_RIE")
    quant = quant[str(49)]
    r, tag, mass = quant["r"], quant["tag"], quant["mass"]
    edirichlet = dirichlet_energy(r, tag, mass, dim, dx, box_size, "QSK")
    print('The Dirichlet energy of the 2D TGV at time stamp 50 is:', edirichlet)

    # Dirichlet energy of a constant density field
    dim = 2
    dx = 0.01
    box_size = jnp.array([1, 1])
    r = pos_init_cartesian_2d(box_size, dx)
    tag = jnp.zeros(jnp.shape(r)[0])
    mass = jnp.ones_like(tag)
    edirichlet = dirichlet_energy(r, tag, mass, dim, dx, box_size, "QSK")
    print('The Dirichlet energy of a constant density field is:', edirichlet)










