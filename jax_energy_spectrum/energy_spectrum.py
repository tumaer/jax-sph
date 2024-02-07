"""energy spectrum implementation"""
import argparse
import os

import jax.numpy as jnp
import numpy as np
from jax import vmap
from jax_md import space
from jax_sph import partition
from jax_md.partition import Sparse
from jax_sph.args import Args
from jax_sph.io_state import read_args, read_h5
from jax_energy_spectrum.grid_setup import interpolationGrid
from jax_energy_spectrum.mls import MLS_2nd_order_3D, MLS_2nd_order_2D

EPS = jnp.finfo(float).eps




def energy_spectrum(args_energy):

    args_file = read_args(os.path.join(args_energy.source_dir, "args.txt"))

    files = os.listdir(args_energy.source_dir)
    files_h5 = [f for f in files if (".h5" in f)]
    files_h5 = sorted(files_h5)

    state = {}
 
    # TODO: needs change!!!
    for i, filename in enumerate(files_h5):
        state[str(i)] = read_h5(os.path.join(args_energy.source_dir, filename))
        

    r_file = state[str(args_energy.time_step)]["r"]
    u_file = state[str(args_energy.time_step)]["u"]
    r_mls, box_size = interpolationGrid(args_file).initialize_grid()
    # TODO: use periodic?
    displacement_fn, shift_fn = space.periodic(side=box_size)

    # specify cutoff radius according to used kernel
    if args_energy.kernel == "QSK":
        r_cutoff = 3 * args_file.dx
    elif args_energy.kernel == "WC2K":
        r_cutoff = 2.3 * args_file.dx
    elif args_energy.kernel == "M4K":
        r_cutoff = 2.85 * args_file.dx


    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=r_cutoff,
        backend=args_energy.nl_backend,
        dr_threshold=r_cutoff * 0.25,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=r_mls.shape[0],
        num_partitions=args_energy.num_partitions,
        #pbc=np.array(args.periodic_boundary_conditions), # function?
    )
    num_particles = jnp.shape(r_mls)[0]

    # TODO: How to Neighbors list between two point clouds?
    neighbors = neighbor_fn.allocate(r_mls, r_file, num_particles=num_particles) # NOT WORKING !!! - FIX THAT JONAS

    
    # precompute displacements `dr` and distances `dist`
    # the second vector is sorted
    i_s, j_s = neighbors.idx
    r_i_s, r_j_s = r_mls[i_s], r_file[j_s]
    dr_j_i = vmap(displacement_fn)(r_j_s, r_i_s)


    if args_file.dim == 3:
        u_file_x = u_file[:,0]
        u_file_y = u_file[:,1]
        u_file_z = u_file[:,2]

        u_interp_x = vmap(MLS_2nd_order_3D)(args_energy.kernel, u_file_x[j_s], dr_j_i, args_file.dim, args_file.dx)
        u_interp_y = vmap(MLS_2nd_order_3D)(args_energy.kernel, u_file_y[j_s], dr_j_i, args_file.dim, args_file.dx)
        u_interp_z = vmap(MLS_2nd_order_3D)(args_energy.kernel, u_file_z[j_s], dr_j_i, args_file.dim, args_file.dx)
    elif args_file.dim == 2:
        u_file_x = u_file[:,0]
        u_file_y = u_file[:,1]

        u_interp_x = vmap(MLS_2nd_order_2D)(args_energy.kernel, u_file_x[j_s], dr_j_i, args_file.dim, args_file.dx)
        u_interp_y = vmap(MLS_2nd_order_2D)(args_energy.kernel, u_file_y[j_s], dr_j_i, args_file.dim, args_file.dx)

    # TODO: FFT over interp. velocity field




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, help="Source directory")
    parser.add_argument("--kernel", type=str, default="QSK", choices=["QSK", "WC2K", "M4K"], help="choose kernel function")
    parser.add_argument("--time_step", type=int, default=1, help="choose timestep from h5 file")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use. -1 for CPU")
    parser.add_argument("--nl-backend", default="jaxmd_vmap", choices=["jaxmd_vmap", "jaxmd_scan", "matscipy"], help="Which backend to use for neighbor list")
    parser.add_argument("--relax-pbc", action="store_true", help="Relax particles in a PBC box oposed to wall box")
    parser.add_argument("--num-partitions", type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    energy_spectrum(args)


'''
def get_real_wavenumber_grid(N):
    Nf = N//2 + 1
    k = np.fft.fftfreq(N, 1./N)  # for y and z direction
    kx = k[:Nf].copy()
    kx[-1] *= -1
    k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    return k_field, k

def get_energy_spectrum(vel):
    Nx, Ny, Nz = vel.shape[1:]
    assert (Nx == Ny and Ny == Nz)

    N = Nx
    Nf = N//2 + 1

    k_field, k = get_real_wavenumber_grid(N=Nx)
    k_mag = np.sqrt(k_field[0]*k_field[0] + k_field[1]
                    * k_field[1] + k_field[2]*k_field[2])

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + \
        1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)

    # Fourier transform
    vel_hat = jnp.stack([jnp.fft.rfftn(vel[ii], axes=(2, 1, 0))
                        for ii in range(3)])

    ek = np.zeros(N)
    n_samples = np.zeros(N)

    uu = fact * 0.5 * (jnp.abs(vel_hat[0]*vel_hat[0]) + jnp.abs(
        vel_hat[1]*vel_hat[1]) + jnp.abs(vel_hat[2]*vel_hat[2]))

    np.add.at(ek, shell.flatten(), uu.flatten())
    np.add.at(n_samples, shell.flatten(), 1)
    ek *= 4 * np.pi * k**2 / (n_samples + 1e-10)
    ek *= 1/(N**3)

    return ek

def comp_energy_spectrum(dir_path):
    def mask_for_plot(masked_array):
        x = np.arange(len(masked_array))
        x = x[~masked_array.mask]
        y = masked_array.data[~masked_array.mask]
        mask2 = y > -17
        x = x[mask2]
        y = y[mask2]
        return x, y

    # read velocity tensor from h5py file (5s into TGV with Re=100 on 64^3 grid)
    vel = h5py.File(os.path.join(dir_path, "tgv_vel_64.h5"), "r")["velocity"][:]
    # compute reference energy spectrum
    ek_ref = get_energy_spectrum(vel)

    # plot results
    plt.figure(figsize=(7, 8))
    plt.plot(*mask_for_plot(np.ma.log(ek_ref)), 'x', label="reference")
    plt.legend()
    plt.ylabel("log(E(k))")
    plt.xlabel("Wavenumber k")
    plt.grid()
    plt.savefig(os.path.join(dir_path, "ek_plot.png"))
    plt.close()

'''
