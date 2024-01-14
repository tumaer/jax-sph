import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jax_sph.io_state import read_args, read_h5
from jax_sph.utils import get_ekin, get_val_max

EPS = jnp.finfo(float).eps

def val_TGV(val_root, dim=2, nxs=[10, 20], save_fig=False):
    """Validate the SPH implementation of the Taylor-Green Vortex

    Args:
        dim (int, optional): Dimension. Defaults to 2.
        nxs (list, optional): Set of 1/dx values. Defaults to [10, 20, 50].
        save_fig (bool, optional): Whether to save output. Defaults to False.
    """

    # get all existing directories with relevant statistics
    dirs = os.listdir(val_root)
    if dim == 2:
        dirs = [d for d in dirs if ("2D_TGV_RIE2" in d)]
    elif dim == 3:
        dirs = [d for d in dirs if ("3D_TGV_RIE2" in d)]
    dirs = sorted(dirs)

    # to each directory read the medatada file and store nx values
    nx_found = {}
    for dir in dirs:
        args = read_args(os.path.join(val_root, dir, "args.txt"))
        side_length = args.bounds[0][1] - args.bounds[0][0]
        nx_found[round(side_length / args.dx)] = [dir, args]

    # verify that all requested nx values are available
    for nx in nxs:
        assert nx in nx_found, FileNotFoundError(f"Simulation nx={nx} missing")

    # plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for nx in nxs:
        args = nx_found[nx][1]
        # dx = args.dx

        dir_path = nx_found[nx][0]
        files = os.listdir(os.path.join(val_root, dir_path))
        files_h5 = [f for f in files if (".h5" in f)]
        files_h5 = sorted(files_h5)

        u_max_vec = np.zeros((len(files_h5)))
        e_kin_vec = np.zeros((len(files_h5)))

        for i, filename in enumerate(files_h5):
            state = read_h5(os.path.join(val_root, dir_path, filename))
            u_max_vec[i] = get_val_max(state, "v")
            e_kin_vec[i] = get_ekin(state, args.dx)
        e_kin_vec /= side_length**dim
        dEdt = -(e_kin_vec[1:] - e_kin_vec[:-1]) / (args.dt * args.write_every)

        # plot
        t = np.linspace(0.0, args.t_end, len(u_max_vec))
        if args.is_limiter:
            if dim == 2:
                axs[0].plot(t, u_max_vec, label=f"Riemann SPH, dx={args.dx}, eta={args.eta_limiter}")
            elif dim == 3:
                axs[0].plot(t[:-1], dEdt, label=f"Riemann SPH, dx={args.dx}, eta={args.eta_limiter}")
            axs[1].plot(t, e_kin_vec, label=f"Riemann SPH, dx={args.dx}, eta={args.eta_limiter}")
        else:
            if dim == 2:
                axs[0].plot(t, u_max_vec, label=f"Riemann SPH, dx={args.dx}")
            elif dim == 3:
                axs[0].plot(t[:-1], dEdt, label=f"Riemann SPH, dx={args.dx}")
            axs[1].plot(t, e_kin_vec, label=f"Riemann SPH, dx={args.dx}")


    # reference solutions in 2D and 3D
    if dim == 2:
        # x-axis
        t = np.linspace(0.0, args.t_end, 100)

        rho = 1.0
        u_ref = 1.0
        L = 1.0
        eta = 0.01
        Re = rho * u_ref * L / eta
        slope_u_max = -8 * np.pi**2 / Re
        u_max_theory = np.exp(slope_u_max * t)
        # factor 0.25=int(int((sin(2pi x) cos(2pi y))^2, x from 0 to 1), y from 0 to 1)
        e_kin_theory = 0.25 * np.exp(2 * slope_u_max * t)

        axs[0].plot(t, u_max_theory, "k", label="Theory")
        axs[1].plot(t, e_kin_theory, "k", label="Theory")
    elif dim == 3:
        Re = 1 / args.viscosity
        ref = np.loadtxt(f"./validation/ref/tgv3d_ref_{int(Re)}.txt", delimiter=",")
        num_dots = 50
        every_n = max(len(ref) // num_dots, 1)
        axs[0].scatter(
            ref[::every_n, 0],
            ref[::every_n, 1],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )
        axs[1].scatter(
            ref[::every_n, 0],
            ref[::every_n, 2],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )

    # plot layout

    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$t$ [-]")
    axs[0].set_ylabel(r"$u_{max}$ [-]" if dim == 2 else r"$dE/dt$ [-]")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$t$ [-]")
    axs[1].set_ylabel(r"$E_{kin}$ [-]")
    axs[1].legend()
    axs[1].grid()

    fig.suptitle(f"{str(dim)}D Taylor-Green Vortex")
    fig.tight_layout()

    ###### save or visualize
    if save_fig:
        os.makedirs(f"{val_root}", exist_ok=True)
        nxs_str = "_".join([str(i) for i in nxs])
        plt.savefig(f"{val_root}/{str(dim)}D_TGV_{nxs_str}.png")

    plt.show()

def val_TGV_Riemann(val_root, dim=2, nxs=[10, 20], save_fig=False):
    """Validate the SPH implementation of the Taylor-Green Vortex

    Args:
        dim (int, optional): Dimension. Defaults to 2.
        nxs (list, optional): Set of 1/dx values. Defaults to [10, 20, 50].
        save_fig (bool, optional): Whether to save output. Defaults to False.
    """

    # get all existing directories with relevant statistics
    dirs = os.listdir(val_root)
    if dim == 2:
        dirs = [d for d in dirs if ("2D_TGV_RIE2" in d)]
    elif dim == 3:
        dirs = [d for d in dirs if ("3D_TGV_RIE2" in d)]
    dirs = sorted(dirs)

    # to each directory read the medatada file and store nx values
    nx_found = {}
    for dir in dirs:
        args = read_args(os.path.join(val_root, dir, "args.txt"))
        side_length = args.bounds[0][1] - args.bounds[0][0]
        nx_found[round(side_length / args.dx)] = [dir, args]

    # verify that all requested nx values are available
    for nx in nxs:
        assert nx in nx_found, FileNotFoundError(f"Simulation nx={nx} missing")

    # plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for nx in nxs:
        args = nx_found[nx][1]
        # dx = args.dx

        dir_path = nx_found[nx][0]
        files = os.listdir(os.path.join(val_root, dir_path))
        files_h5 = [f for f in files if (".h5" in f)]
        files_h5 = sorted(files_h5)

        u_max_vec = np.zeros((len(files_h5)))
        e_kin_vec = np.zeros((len(files_h5)))

        for i, filename in enumerate(files_h5):
            state = read_h5(os.path.join(val_root, dir_path, filename))
            u_max_vec[i] = get_val_max(state, "v")
            e_kin_vec[i] = get_ekin(state, args.dx)
        e_kin_vec /= side_length**dim
        dEdt = -(e_kin_vec[1:] - e_kin_vec[:-1]) / (args.dt * args.write_every)

        # plot
        t = np.linspace(0.0, args.t_end, len(u_max_vec))
        if dim == 2:
            axs[0].plot(t, u_max_vec, label=f"SPH, dx={args.dx}")
        elif dim == 3:
            axs[0].plot(t[:-1], dEdt, label=f"SPH, dx={args.dx}")
        axs[1].plot(t, e_kin_vec, label=f"SPH, dx={args.dx}")

    # reference solutions in 2D and 3D
    if dim == 2:
        # x-axis
        t = np.linspace(0.0, args.t_end, 100)

        rho = 1.0
        u_ref = 1.0
        L = 1.0
        eta = 0.01
        Re = rho * u_ref * L / eta
        slope_u_max = -8 * np.pi**2 / Re
        u_max_theory = np.exp(slope_u_max * t)
        # factor 0.25=int(int((sin(2pi x) cos(2pi y))^2, x from 0 to 1), y from 0 to 1)
        e_kin_theory = 0.25 * np.exp(2 * slope_u_max * t)

        axs[0].plot(t, u_max_theory, "k", label="Theory")
        axs[1].plot(t, e_kin_theory, "k", label="Theory")
    elif dim == 3:
        Re = 1 / args.viscosity
        ref = np.loadtxt(f"./validation/ref/tgv3d_ref_{int(Re)}.txt", delimiter=",")
        num_dots = 50
        every_n = max(len(ref) // num_dots, 1)
        axs[0].scatter(
            ref[::every_n, 0],
            ref[::every_n, 1],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )
        axs[1].scatter(
            ref[::every_n, 0],
            ref[::every_n, 2],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )

    # plot layout

    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$t$ [-]")
    axs[0].set_ylabel(r"$u_{max}$ [-]" if dim == 2 else r"$dE/dt$ [-]")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylim(bottom = 10**(-2))
    axs[0].set_xlim([0, 4])

    axs[1].set_yscale("log")
    axs[1].set_xlabel(r"$t$ [-]")
    axs[1].set_ylabel(r"$E_{kin}$ [-]")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_ylim(bottom = 10**(-4))
    axs[1].set_xlim([0, 4])

    fig.suptitle(f"{str(dim)}D Taylor-Green Vortex")
    fig.tight_layout()

    ###### save or visualize
    if save_fig:
        os.makedirs(f"{val_root}", exist_ok=True)
        nxs_str = "_".join([str(i) for i in nxs])
        plt.savefig(f"{val_root}/{str(dim)}D_TGV_RIE_{nxs_str}.png")

    plt.show()   



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, help="One of the above cases")
    parser.add_argument("--src_dir", type=str, help="Source directory")
    args = parser.parse_args()

    if args.case == "2D_TGV":
        val_TGV(args.src_dir, 2, [100], True)
    elif args.case == "3D_TGV":
        val_TGV(args.src_dir, 3, [20, 40], True)
    elif args.case == "2D_TGV_RIE":
        val_TGV_Riemann(args.src_dir, 2, [20, 50, 100], True)
    elif args.case == "3D_TGV_RIE":
        val_TGV_Riemann(args.src_dir, 3, [20, 40], True)

