"""Collection of validation scripts"""

import argparse
import os

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jax_sph.io_state import read_args, read_h5
from jax_sph.utils import get_ekin, get_val_max, sph_interpolator

EPS = jnp.finfo(float).eps


def val_TGV(val_root, dim=2, nxs=[50, 100], save_fig=False):
    """Validate the SPH implementation of the Taylor-Green Vortex

    Args:
        dim (int, optional): Dimension. Defaults to 2.
        nxs (list, optional): Set of 1/dx values. Defaults to [10, 20, 50].
        save_fig (bool, optional): Whether to save output. Defaults to False.
    """

    # get all existing directories with relevant statistics
    if dim == 2:
        notvf = "tgv2d_notvf/"
        tvf = "tgv2d_tvf/"
        Rie = "tgv2d_Rie/"
    elif dim == 3:
        notvf = "tgv3d_notvf/"
        tvf = "tgv3d_tvf/"
        Rie = "tgv3d_Rie/"

    dirs_notvf = os.listdir(os.path.join(val_root, notvf))
    dirs_tvf = os.listdir(os.path.join(val_root, tvf))
    dirs_Rie = os.listdir(os.path.join(val_root, Rie))
    if dim == 2:
        dirs_notvf = [d for d in dirs_notvf if ("2D_TGV_SPH" in d)]
        dirs_tvf = [d for d in dirs_tvf if ("2D_TGV_SPH" in d)]
        dirs_Rie = [d for d in dirs_Rie if ("2D_TGV_RIE" in d)]
    elif dim == 3:
        dirs_notvf = [d for d in dirs_notvf if ("3D_TGV_SPH" in d)]
        dirs_tvf = [d for d in dirs_tvf if ("3D_TGV_SPH" in d)]
        dirs_Rie = [d for d in dirs_Rie if ("3D_TGV_RIE" in d)]
    dirs_notvf = sorted(dirs_notvf)
    dirs_tvf = sorted(dirs_tvf)
    dirs_Rie = sorted(dirs_Rie)

    # to each directory read the medatada file and store nx values
    nx_found_notvf = {}
    nx_found_tvf = {}
    nx_found_Rie = {}
    for dir in dirs_notvf:
        args = read_args(os.path.join(val_root, notvf, dir, "args.txt"))
        side_length = args.bounds[0][1] - args.bounds[0][0]
        nx_found_notvf[round(side_length / args.dx)] = [dir, args]
    for dir in dirs_tvf:
        args = read_args(os.path.join(val_root, tvf, dir, "args.txt"))
        side_length = args.bounds[0][1] - args.bounds[0][0]
        nx_found_tvf[round(side_length / args.dx)] = [dir, args]
    for dir in dirs_Rie:
        args = read_args(os.path.join(val_root, Rie, dir, "args.txt"))
        side_length = args.bounds[0][1] - args.bounds[0][0]
        nx_found_Rie[round(side_length / args.dx)] = [dir, args]

    # verify that all requested nx values are available
    for nx in nxs:
        assert nx in nx_found_notvf, FileNotFoundError(f"Simulation nx={nx} missing")
        assert nx in nx_found_tvf, FileNotFoundError(f"Simulation nx={nx} missing")
        assert nx in nx_found_Rie, FileNotFoundError(f"Simulation nx={nx} missing")

    # plots
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
    temp = 0

    for nx in nxs:
        temp = temp + 1

        args_tvf = nx_found_tvf[nx][1]
        args_notvf = nx_found_tvf[nx][1]
        args_Rie = nx_found_Rie[nx][1]
        # dx = args.dx

        dir_path_tvf = nx_found_tvf[nx][0]
        dir_path_notvf = nx_found_notvf[nx][0]
        dir_path_Rie = nx_found_Rie[nx][0]
        files_tvf = os.listdir(os.path.join(val_root, tvf, dir_path_tvf))
        files_notvf = os.listdir(os.path.join(val_root, notvf, dir_path_notvf))
        files_Rie = os.listdir(os.path.join(val_root, Rie, dir_path_Rie))
        files_h5_tvf = [f for f in files_tvf if (".h5" in f)]
        files_h5_tvf = sorted(files_h5_tvf)
        files_h5_notvf = [f for f in files_notvf if (".h5" in f)]
        files_h5_notvf = sorted(files_h5_notvf)
        files_h5_Rie = [f for f in files_Rie if (".h5" in f)]
        files_h5_Rie = sorted(files_h5_Rie)

        u_max_vec_tvf = np.zeros((len(files_h5_tvf)))
        e_kin_vec_tvf = np.zeros((len(files_h5_tvf)))
        u_max_vec_notvf = np.zeros((len(files_h5_notvf)))
        e_kin_vec_notvf = np.zeros((len(files_h5_notvf)))
        u_max_vec_Rie = np.zeros((len(files_h5_Rie)))
        e_kin_vec_Rie = np.zeros((len(files_h5_Rie)))

        for i, filename in enumerate(files_h5_tvf):
            state = read_h5(os.path.join(val_root, tvf, dir_path_tvf, filename))
            if i == 1:
                r0 = state["r"]
                u0 = state["u"]
            elif i == 440:
                rend = state["r"]
                uend = state["u"]
            u_max_vec_tvf[i] = get_val_max(state, "v")
            e_kin_vec_tvf[i] = get_ekin(state, args_tvf.dx)
        e_kin_vec_tvf /= side_length**dim
        dEdt_tvf = -(e_kin_vec_tvf[1:] - e_kin_vec_tvf[:-1]) / (
            args_tvf.dt * args_tvf.write_every
        )

        for i, filename in enumerate(files_h5_notvf):
            state = read_h5(os.path.join(val_root, notvf, dir_path_notvf, filename))
            u_max_vec_notvf[i] = get_val_max(state, "v")
            e_kin_vec_notvf[i] = get_ekin(state, args_notvf.dx)
        e_kin_vec_notvf /= side_length**dim
        dEdt_notvf = -(e_kin_vec_notvf[1:] - e_kin_vec_notvf[:-1]) / (
            args_notvf.dt * args_notvf.write_every
        )

        for i, filename in enumerate(files_h5_Rie):
            state = read_h5(os.path.join(val_root, Rie, dir_path_Rie, filename))
            u_max_vec_Rie[i] = get_val_max(state, "v")
            e_kin_vec_Rie[i] = get_ekin(state, args_Rie.dx)
        e_kin_vec_Rie /= side_length**dim
        dEdt_Rie = -(e_kin_vec_Rie[1:] - e_kin_vec_Rie[:-1]) / (
            args_Rie.dt * args_Rie.write_every
        )

        if dim == 2:
            if temp == 1:
                x = "--"
            elif temp == 2:
                x = "-"
        elif dim == 3:
            if temp == 1:
                x = "-."
            elif temp == 2:
                x = "--"
            elif temp == 3:
                x = "-"

        cmap = matplotlib.colormaps["turbo"]

        # plot
        t_tvf = np.linspace(0.0, args_tvf.t_end, len(u_max_vec_tvf))
        t_notvf = np.linspace(0.0, args_notvf.t_end, len(u_max_vec_notvf))
        t_Rie = np.linspace(0.0, args_Rie.t_end, len(u_max_vec_Rie))
        if dim == 2:
            lbl1 = f"SPH + tvf, dx={args_tvf.dx}"
            lbl2 = f"SPH, dx={args_notvf.dx}"
            lbl3 = f"Riemann, dx={args_Rie.dx}"
            axs1[0].plot(t_tvf, u_max_vec_tvf, x, color=cmap(0.65), label=lbl1)
            axs1[0].plot(t_notvf, u_max_vec_notvf, x, color=cmap(0.1), label=lbl2)
            axs1[0].plot(t_Rie, u_max_vec_Rie, x, color=cmap(0.9), label=lbl3)
            axs1[1].plot(t_tvf, e_kin_vec_tvf, x, color=cmap(0.65), label=lbl1)
            axs1[1].plot(t_notvf, e_kin_vec_notvf, x, color=cmap(0.1), label=lbl2)
            axs1[1].plot(t_Rie, e_kin_vec_Rie, x, color=cmap(0.9), label=lbl3)
        elif dim == 3:
            lbl1 = f"SPH + tvf, dx={np.round(args_tvf.dx, decimals=3)}"
            lbl2 = f"SPH, dx={np.round(args_notvf.dx, decimals=3)}"
            lbl3 = f"Riemann, dx={np.round(args_Rie.dx, decimals=3)}"
            axs1[0].plot(t_tvf[:-1], dEdt_tvf, x, color=cmap(0.65), label=lbl1)
            axs1[0].plot(t_notvf[:-1], dEdt_notvf, x, color=cmap(0.1), label=lbl2)
            axs1[0].plot(t_Rie[:-1], dEdt_Rie, x, color=cmap(0.9), label=lbl3)
            axs1[1].plot(t_tvf, e_kin_vec_tvf, x, color=cmap(0.65), label=lbl1)
            axs1[1].plot(t_notvf, e_kin_vec_notvf, x, color=cmap(0.1), label=lbl2)
            axs1[1].plot(t_Rie, e_kin_vec_Rie, x, color=cmap(0.9), label=lbl3)

    # reference solutions in 2D and 3D
    if dim == 2:
        # x-axis
        t = np.linspace(0.0, args_tvf.t_end, 100)

        rho = 1.0
        u_ref = 1.0
        L = 1.0
        eta = 0.01
        Re = rho * u_ref * L / eta
        slope_u_max = -8 * np.pi**2 / Re
        u_max_theory = np.exp(slope_u_max * t)
        e_kin_theory = 0.25 * np.exp(2 * slope_u_max * t)

        axs1[0].plot(t, u_max_theory, "k", label="Theory")
        axs1[1].plot(t, e_kin_theory, "k", label="Theory")
    elif dim == 3:
        Re = 1 / args.viscosity
        ref = np.loadtxt(
            f"./validation_paper/ref/tgv3d_ref_{int(Re)}.txt", delimiter=","
        )
        num_dots = 50
        every_n = max(len(ref) // num_dots, 1)
        axs1[0].scatter(
            ref[::every_n, 0],
            ref[::every_n, 1],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )
        axs1[1].scatter(
            ref[::every_n, 0],
            ref[::every_n, 2],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )

    # plot layout

    axs1[0].set_yscale("log")
    axs1[0].set_xlabel(r"$t$ [-]")
    axs1[0].set_ylabel(r"$u_{max}$ [-]" if dim == 2 else r"$dE/dt$ [-]")
    axs1[0].set_xlim([0, args_Rie.t_end])
    axs1[0].legend()
    axs1[0].grid()

    axs1[1].set_yscale("log")
    axs1[1].set_xlabel(r"$t$ [-]")
    axs1[1].set_ylabel(r"$E_{kin}$ [-]")
    axs1[1].set_xlim([0, args_Rie.t_end])
    axs1[1].legend()
    axs1[1].grid()

    fig1.suptitle(f"{str(dim)}D Taylor-Green Vortex")
    fig1.tight_layout()

    ###### save or visualize
    if save_fig:
        os.makedirs(f"{val_root}", exist_ok=True)
        nxs_str = "_".join([str(i) for i in nxs])
        plt.savefig(f"{val_root}/{str(dim)}D_TGV_{nxs_str}.pdf")

    plt.show()

    if dim == 2:
        vel0 = jnp.linalg.norm(u0, axis=1)
        velend = jnp.linalg.norm(uend, axis=1)

        fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))

        axs2[0].scatter(
            r0[:, 0], r0[:, 1], c=vel0, cmap="turbo", s=4, vmin=-0.0, vmax=1
        )
        axs2[0].set_xlim([0, 1])
        axs2[0].set_ylim([0, 1])
        axs2[0].tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

        axs2[1].scatter(
            rend[:, 0], rend[:, 1], c=velend, cmap="turbo", s=4, vmin=-0.0, vmax=1
        )
        axs2[1].set_xlim([0, 1])
        axs2[1].set_ylim([0, 1])
        axs2[1].tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

        fig2.tight_layout()

        if save_fig:
            os.makedirs(val_root, exist_ok=True)
            plt.savefig(f"{val_root}/2D_TGV_Scatter.pdf", dpi=300)

        plt.show()


def val_2D_LDC(
    val_root_tvf, val_root_notvf, val_root_Rie, dim=2, nxs=[50], save_fig=False
):
    # get all existing directories with relevant statistics
    dirs_Rie = os.listdir(val_root_Rie)
    dirs_tvf = os.listdir(val_root_tvf)
    dirs_notvf = os.listdir(val_root_notvf)
    if dim == 2:
        dirs_Rie = [d for d in dirs_Rie if ("2D_LDC_RIE" in d)]
        dirs_tvf = [d for d in dirs_tvf if ("2D_LDC_SPH" in d)]
        dirs_notvf = [d for d in dirs_notvf if ("2D_LDC_SPH" in d)]
    elif dim == 3:
        raise NotImplementedError

    if len(dirs_tvf) > 1:
        raise ValueError(f"More than one directory found in {val_root_tvf}")
    elif len(dirs_notvf) > 1:
        raise ValueError(f"More than one directory found in {val_root_notvf}")
    elif len(dirs_Rie) > 1:
        raise ValueError(f"More than one directory found in {val_root_Rie}")
        # TODO: implement influence of discretization?

    val_dir_path_Rie = os.path.join(val_root_Rie, dirs_Rie[0])
    files_Rie = os.listdir(val_dir_path_Rie)
    files_Rie = [f for f in files_Rie if (".h5" in f)]
    files_Rie = sorted(files_Rie, key=lambda x: int(x.split("_")[1][:-3]))

    val_dir_path_tvf = os.path.join(val_root_tvf, dirs_tvf[0])
    files_tvf = os.listdir(val_dir_path_tvf)
    files_tvf = [f for f in files_tvf if (".h5" in f)]
    files_tvf = sorted(files_tvf, key=lambda x: int(x.split("_")[1][:-3]))

    val_dir_path_notvf = os.path.join(val_root_notvf, dirs_notvf[0])
    files_notvf = os.listdir(val_dir_path_notvf)
    files_notvf = [f for f in files_notvf if (".h5" in f)]
    files_notvf = sorted(files_notvf, key=lambda x: int(x.split("_")[1][:-3]))

    args = read_args(os.path.join(val_dir_path_Rie, "args.txt"))
    Re = 1 / args.viscosity  # Reynolds number

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    N = round(1 / args.dx) + 1
    for i, src_path in enumerate(files_Rie[-1:]):
        # for i, src_path in enumerate(files[-1:]):
        src_path_Rie = os.path.join(val_dir_path_Rie, src_path)
        src_path_tvf = os.path.join(val_dir_path_tvf, src_path)
        src_path_notvf = os.path.join(val_dir_path_notvf, src_path)
        if i == 0:
            interp_vel_fn = sph_interpolator(args, src_path_Rie)

        # velocity in Y
        x_axis = jnp.array([args.dx * (3 + i) for i in range(N)])  # (101, )
        rs = (0.5 + 3 * args.dx) * jnp.ones([x_axis.shape[0], 2])  # (101, 2)
        rs = rs.at[:, 0].set(x_axis)
        v_val_Rie = interp_vel_fn(src_path_Rie, rs, "u", dim_ind=1)
        v_val_tvf = interp_vel_fn(src_path_tvf, rs, "u", dim_ind=1)
        v_val_notvf = interp_vel_fn(src_path_notvf, rs, "u", dim_ind=1)

        state = read_h5(src_path_Rie)
        rs = state["r"]
        mask = np.where(
            (rs[:, 1] > 0.5 + args.dx * 2) * (rs[:, 1] < 0.5 + args.dx * 4), True, False
        )
        x_axis_dots = rs[mask, 0]
        v_val_dots = state["v"][mask, 1]
        sorted_indices = np.argsort(x_axis_dots)
        x_axis_dots = x_axis_dots[sorted_indices]
        v_val_dots = v_val_dots[sorted_indices]

        # velocity in x
        y_axis = jnp.array([args.dx * (3 + i) for i in range(N)])  # (101,)
        rs = (0.5 + 3 * args.dx) * jnp.ones([y_axis.shape[0], 2])  # (101, 2)
        rs = rs.at[:, 1].set(y_axis)
        u_val_Rie = interp_vel_fn(src_path_Rie, rs, "u", dim_ind=0)
        u_val_tvf = interp_vel_fn(src_path_tvf, rs, "u", dim_ind=0)
        u_val_notvf = interp_vel_fn(src_path_notvf, rs, "u", dim_ind=0)

        cmap = matplotlib.colormaps["turbo"]

        ax2.plot(
            np.asarray(x_axis - 3 * args.dx),
            np.asarray(v_val_tvf),
            color=cmap(0.65),
            label=f"SPH + tvf, dx={args.dx}",
        )
        ax2.plot(
            np.asarray(x_axis - 3 * args.dx),
            np.asarray(v_val_notvf),
            color=cmap(0.1),
            label=f"SPH, dx={args.dx}",
        )
        ax2.plot(
            np.asarray(x_axis - 3 * args.dx),
            np.asarray(v_val_Rie),
            color=cmap(0.9),
            label=f"Riemann, dx={args.dx}",
        )
        ax2.set_xlim([0, 1])
        ax2.set_ylim([-0.6, 0.5])
        ax2.set_yticks(jnp.linspace(-0.6, 0.5, 12))
        ax2.set_ylabel("V(x)")
        ax2.set_xlabel("x")

        ax3 = ax2.twinx().twiny()
        ax4 = ax2.twinx()
        ax3.set_xlim([-0.4, 1])
        ax3.set_ylim([0, 1])
        ax3.set_xlabel("U(y)")
        ax4.set_ylabel("y")

        ax3.plot(
            np.asarray(u_val_Rie), np.asarray(y_axis - 3 * args.dx), color=cmap(0.9)
        )
        ax3.plot(
            np.asarray(u_val_tvf), np.asarray(y_axis - 3 * args.dx), color=cmap(0.65)
        )
        ax3.plot(
            np.asarray(u_val_notvf), np.asarray(y_axis - 3 * args.dx), color=cmap(0.1)
        )

    # getting the reference data
    u_vel = pd.read_csv("validation_paper/ref/ldc_data_u_vel.csv")
    u_vel.columns = u_vel.iloc[0]

    u_vel = u_vel.drop(labels=0)
    u_vel["100"] = u_vel["100"].replace(
        ["0,00000"], "0.00000"
    )  # had to make some corrections in the input data

    y = (u_vel.loc[:, "y"].values).astype(float)
    u_Re_100, u_Re_1000, u_Re_10000 = (
        (u_vel.loc[:, "100"].values).astype(float),
        (u_vel.loc[:, 1000.0].values).astype(float),
        (u_vel.loc[:, "10,000"].values).astype(float),
    )

    v_vel = pd.read_csv("validation_paper/ref/ldc_data_v_vel.csv")
    v_vel.columns = v_vel.iloc[0]

    v_vel = v_vel.drop(labels=0)  # had to make some corrections in the input data

    x = (v_vel.loc[:, "x"].values).astype(float)
    v_Re_100, v_Re_1000, v_Re_10000 = (
        (v_vel.loc[:, 100.0].values).astype(float),
        (v_vel.loc[:, 1000.0].values).astype(float),
        (v_vel.loc[:, "10,000"].values).astype(float),
    )

    if Re == 100.0:
        ax2.scatter(x, v_Re_100, color="k", marker="s")
        ax3.scatter(u_Re_100, y, color="k", marker="o")
    elif Re == 1000.0:
        ax2.scatter(x, v_Re_1000, color="C1", marker="s")
        ax3.scatter(u_Re_1000, y, color="C0", marker="o")
    elif Re == 10000.0:
        ax2.scatter(x, v_Re_10000, color="C1", marker="s")
        ax3.scatter(u_Re_10000, y, color="C0", marker="o")

    ax2.grid()
    ax2.legend(loc="lower right")

    velend = jnp.linalg.norm(state["u"], axis=1)

    ax.scatter(
        state["r"][:, 0],
        state["r"][:, 1],
        c=velend,
        cmap="turbo",
        s=12,
        vmin=-0.0,
        vmax=1,
    )
    ax.set_xlim([0, 1.12])
    ax.set_ylim([0, 1.12])
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    fig.tight_layout()

    if save_fig:
        os.makedirs(val_root_Rie, exist_ok=True)
        plt.savefig(f"{val_root_Rie}/2D_LCD.pdf", dpi=300)

    plt.show()


def val_DB(val_root, save_fig=False):
    dirs = os.listdir(val_root)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(val_root, d))]
    assert len(dirs) == 1, f"Expected only one directory in {val_root}"
    args = read_args(os.path.join(val_root, dirs[0], "args.txt"))
    assert args.dt == 0.0002 or args.dt == 0.0001 or args.dt == 0.0003
    assert args.dx == 0.02 or args.dx == 0.01

    files = os.listdir(os.path.join(val_root, dirs[0]))
    files_h5 = [f for f in files if (".h5" in f)]
    files_h5 = sorted(files_h5)
    val_dir_path = os.path.join(val_root, dirs[0])
    files = os.listdir(val_dir_path)
    files = [f for f in files if (".h5" in f)]
    files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))

    L_wall = 5.366

    step_max = np.array(np.rint(args.t_end / args.dt), dtype=int)
    digits = len(str(step_max))

    time_stamps = [1.62, 2.38, 4.0, 5.21, 6.02, 7.23]
    steps = [8100, 11900, 20000, 26050, 30100, 36150]
    for i, step in enumerate(steps[:6]):
        file_name = "traj_" + str(step).zfill(digits) + ".h5"
        src_path = os.path.join(val_root, dirs[0], file_name)

        state = read_h5(src_path)
        r = state["r"]
        tags = state["tag"]
        p = state["p"]

        # pressure at wall to zero
        p = np.where(tags > 0, 0, p)

        mask_lower = r[:, 1] < 2 + 3 * args.dx
        r = r[mask_lower]
        p = p[mask_lower]

        _ = plt.figure(figsize=(10, 4))
        plt.scatter(r[:, 0], r[:, 1], c=p, cmap="turbo", s=1, vmin=-0.0, vmax=1)
        plt.xlim([0, L_wall + 6 * args.dx])
        plt.ylim([0, 2 + 3 * args.dx])
        plt.yticks([0, 0.5, 1, 1.5, 2], fontsize=20)
        plt.xticks([0, 1, 2, 3, 4, 5], fontsize=20)
        plt.tight_layout()

        if save_fig:
            os.makedirs(val_root, exist_ok=True)
            plt.savefig(
                f"{val_root}/{str(args.dim)}D_DAM_{str(time_stamps[i])}.pdf", dpi=300
            )

    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, help="One of the above cases")
    parser.add_argument("--src_dir", type=str, help="Source directory")
    parser.add_argument("--src_dir_tvf", type=str, help="Source directory tvf")
    parser.add_argument("--src_dir_notvf", type=str, help="Source directory notvf")
    parser.add_argument("--src_dir_Rie", type=str, help="Source directory Riemann")
    args = parser.parse_args()

    if args.case == "2D_LDC":
        val_2D_LDC(
            args.src_dir_tvf, args.src_dir_notvf, args.src_dir_Rie, save_fig=True
        )

    elif args.case == "2D_TGV":
        val_TGV(args.src_dir, 2, [50, 100], True)

    elif args.case == "3D_TGV":
        val_TGV(args.src_dir, 3, [20, 32, 50], True)

    elif args.case == "2D_DB":
        val_DB(args.src_dir, save_fig=True)
