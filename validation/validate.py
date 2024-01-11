"""Collection of validation scripts"""

import argparse
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jax_sph.io_state import read_args, read_h5
from jax_sph.utils import get_ekin, get_val_max, sph_interpolator

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
        dirs = [d for d in dirs if ("2D_TGV_SPH" in d)]
    elif dim == 3:
        dirs = [d for d in dirs if ("3D_TGV_SPH" in d)]
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


def val_2D_LDC(val_root, dim=2, nxs=[50, 100, 200], save_fig=False):
    # get all existing directories with relevant statistics
    dirs = os.listdir(val_root)
    if dim == 2:
        dirs = [d for d in dirs if ("2D_LDC_SPH" in d)]
    elif dim == 3:
        raise NotImplementedError

    if len(dirs) > 1:
        raise ValueError(f"More than one directory found in {val_root}")
        # TODO: implement influence of discretization?

    val_dir_path = os.path.join(val_root, dirs[0])
    files = os.listdir(val_dir_path)
    files = [f for f in files if (".h5" in f)]
    files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))

    args = read_args(os.path.join(val_dir_path, "args.txt"))
    Re = 1 / args.viscosity  # Reynolds number

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # ax=fig.add_subplot(111, label="1")
    # ax2=fig.add_subplot(111, label="2", frame_on=False)

    N = round(1 / args.dx) + 1
    for i, src_path in enumerate(files[-5:]):
        # for i, src_path in enumerate(files[-1:]):
        src_path = os.path.join(val_dir_path, src_path)
        if i == 0:
            interp_vel_fn = sph_interpolator(args, src_path)

        # velocity in Y
        x_axis = jnp.array([args.dx * (3 + i) for i in range(N)])  # (101, )
        rs = (0.5 + 3 * args.dx) * jnp.ones([x_axis.shape[0], 2])  # (101, 2)
        rs = rs.at[:, 0].set(x_axis)
        v_val = interp_vel_fn(src_path, rs, "u", dim_ind=1)

        state = read_h5(src_path)
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
        u_val = interp_vel_fn(src_path, rs, "u", dim_ind=0)

        ax.plot(np.asarray(x_axis - 3 * args.dx), np.asarray(v_val), color="C1")
        # ax.plot(np.asarray(x_axis_dots - 3 * args.dx), np.asarray(v_val_dots),
        #   color = "C2")
        ax2.plot(np.asarray(y_axis - 3 * args.dx), np.asarray(u_val), color="C0")

    # getting the reference data
    u_vel = pd.read_csv("validation/ref/ldc_data_u_vel.csv")
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

    v_vel = pd.read_csv("validation/ref/ldc_data_v_vel.csv")
    v_vel.columns = v_vel.iloc[0]

    v_vel = v_vel.drop(labels=0)  # had to make some corrections in the input data

    x = (v_vel.loc[:, "x"].values).astype(float)
    v_Re_100, v_Re_1000, v_Re_10000 = (
        (v_vel.loc[:, 100.0].values).astype(float),
        (v_vel.loc[:, 1000.0].values).astype(float),
        (v_vel.loc[:, "10,000"].values).astype(float),
    )
    # Re=100, y-velocity, min at x=0.8 with -0.245, max at x=0.3 with 0.175

    # for Re=100 the reference is:
    # ref_x = np.array([0.0, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
    #   0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
    # ref_v_100 = np.array([0.0, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507,
    #   0.17527, 0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391,
    #   -0.05906, 0.0])

    if Re == 100.0:
        ax.scatter(x, v_Re_100, color="k", marker="s")
        ax2.scatter(y, u_Re_100, color="k", marker="o")
    elif Re == 1000.0:
        ax.scatter(x, v_Re_1000, color="C1", marker="s")
        ax2.scatter(u_Re_1000, y, color="C0", marker="o")
    elif Re == 10000.0:
        ax.scatter(x, v_Re_10000, color="C1", marker="s")
        ax2.scatter(u_Re_10000, y, color="C0", marker="o")

    # x_uni_sorted, v_x_sorted = zip(*sorted(zip(x_uni, v_x)))
    ax.set_xlabel("x", color="C0")
    ax.set_ylabel("$V_y(x)$", color="C1")
    ax.tick_params(axis="x", colors="C1")
    ax.tick_params(axis="y", colors="C1")
    ax.grid()
    # ax.set_xlim([0., 1.])
    # ax.set_ylim([-0.6, 0.5])

    ax2.grid()
    # y_uni_sorted, u_y_sorted = zip(*sorted(zip(y_uni, u_y)))
    # ax2.xaxis.tick_top()
    # ax2.yaxis.tick_right()
    # ax2.set_xlabel('$V_x(y)$', color="C0")
    # ax2.set_ylabel('y', color="C1")
    # ax2.xaxis.set_label_position('top')
    # ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C0")
    # ax2.tick_params(axis='y', colors="C0")
    # ax2.set_xlim([-0.4, 1.])
    # ax2.set_ylim([0., 1.])

    if save_fig:
        plt.savefig(f"{val_root}/2D_LDC_Re_{str(Re)}.png")

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

    #### Front and height evolution for case with water box of 1x1

    # x_front = np.zeros((len(files_h5)))
    # height = np.zeros((len(files_h5)))

    # x_axis = jnp.array([0 + args.dt * args.write_every * i for i in range(
    #   len(files_h5)
    # )])
    # for i, filename in enumerate(files_h5):
    #     state = read_h5(os.path.join(val_root, filename))
    #     r = state['r']
    #     tag = state['tag']
    #     x_front[i] = r[np.where(tag == 0)][:, 0].max() #to find x_front/H
    #     height[i] = r[np.where(tag == 0)][np.where(r[:, 0] < 0.5)][:, 1].max()
    #     #to find h(at x = 0)/H

    # #Ref data
    # #Time evolution of the front (a) and the height (b) of a colapsing water column

    # time_front = np.array([0.43, 0.62, 0.80, 0.97, 1.14, 1.29, 1.45, 1.62, 1.76, 1.93,
    #   2.07, 2.24, 2.40, 2.54, 2.71, 2.87, 3.04, 3.21, 3.29, 3.32])
    # y_front = np.array([1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00,
    #   3.22, 3.44, 3.67, 3.89, 4.11, 4.33, 4.56, 4.78, 4.89, 5.0])

    # time_height = np.array([0., 0.80, 1.29, 1.74, 2.15, 2.57, 3.08, 4.27, 6.29])
    # y_height = np.append(np.arange(1.0, 0.55, -0.11), np.arange(0.44, 0.0, -0.11))

    # fig, axs =plt.subplots(2, 1, constrained_layout=True, figsize = (10, 10))
    # axs[0].scatter(time_front, y_front, marker = "^")
    # axs[0].plot(x_axis, x_front)
    # axs[0].set_xlim([0., 3.5])
    # axs[0].set_ylim([0, 6])
    # axs[0].set_xlabel("t[-]")
    # axs[0].set_ylabel("x_front / H")
    # axs[0].set_title("Time evolution of the front")

    # axs[1].scatter(time_height, y_height, marker = "^")
    # axs[1].plot(x_axis, height)
    # axs[1].set_xlim([0., 3.5])
    # axs[1].set_ylim([0., 1.4])
    # axs[1].set_xlabel("t[-]")
    # axs[1].set_ylabel("h(x = 0) / H")
    # axs[1].set_title("Time evolution of the height")

    L_wall = 5.366
    # H_wall = 2.0
    #### Pressure distribution

    step_max = np.array(np.rint(args.t_end / args.dt), dtype=int)
    digits = len(str(step_max))

    time_stamps = [2, 4.8, 5.7, 6.2, 7.4, 9, 10, 11, 12]
    steps = [10000, 24000, 28500, 31000, 37000, 45000, 50000, 55000, 60000]
    # time_stamps = [2.0, 5.7, 6.2, 7.4]
    # steps = [i * args.write_every for i in [200, 570, 620, 740]]
    for i, step in enumerate(steps[:5]):
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
        plt.scatter(r[:, 0], r[:, 1], c=p, cmap="rainbow", s=1, vmin=-0.06, vmax=1)
        plt.xlim([0, L_wall + 6 * args.dx])
        plt.ylim([0, 2 + 3 * args.dx])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.yticks([0, 0.5, 1, 1.5, 2])
        plt.tight_layout()

        if save_fig:
            os.makedirs(val_root, exist_ok=True)
            plt.savefig(
                f"{val_root}/{str(args.dim)}D_DAM_{str(time_stamps[i])}.png", dpi=100
            )

    plt.show()
    plt.close()

    #### Pressure on right wall

    # extract point from our solution for the case with water box of 2x1
    rs = jnp.array([[L_wall + 3 * args.dx, 0.2 + 3 * args.dx]])

    time_axis = np.arange(0, args.t_end / args.dt, args.write_every, dtype=int)
    time_axis = time_axis * args.dt
    # time_axis = time_axis[::5]
    pressure_list = []
    for i, t_val in enumerate(time_axis):
        step = np.array(np.rint(t_val / args.dt), dtype=int)
        file_name = "traj_" + str(step).zfill(digits) + ".h5"
        src_path = os.path.join(val_root, dirs[0], file_name)

        if i == 0:
            interp_vel_fn = sph_interpolator(args, src_path, prop_type="scalar")

        p_val = interp_vel_fn(src_path, rs, prop="p")
        pressure_list.append(p_val)

    pressure_sph = np.asarray(pressure_list).squeeze()
    # pressures = np.convolve(pressure_sph[2:], np.ones(5)/5, mode='valid')
    # pressures = pressures[4:-2:5]
    pressures = pressure_sph[3:-2:5]
    times = time_axis[3:-2:5]

    # reference data
    pressure_profile = pd.read_csv("validation/ref/Buchner_pressureProfile.csv")
    x_val = pressure_profile["x"].to_numpy()
    y_val = pressure_profile[" y"].to_numpy()

    _ = plt.figure(figsize=(5, 4))
    # plt.plot(time_axis, pressure_sph, 'k', mfc='none', label=r'SPH, H/$\Delta$x=100')
    plt.plot(times, pressures, "k", mfc="none", label=r"SPH, H/$\Delta$x=100")
    plt.scatter(x_val, y_val, marker="^", c="C1", label="Exp. Data")
    plt.xlim([0, 8])
    # plt.ylim([-0.2, 1.2])
    plt.xlabel(r"t(g/H)$^{1/2}$")
    plt.ylabel(r"p/$\rho$gH")
    plt.legend()
    plt.tight_layout()

    if save_fig:
        nx = str(round(1 / args.dx))
        os.makedirs(val_root, exist_ok=True)
        plt.savefig(f"{val_root}/{str(args.dim)}D_DAM_{nx}.png", dpi=300)

    plt.show()
    plt.close()


def val_2D_PF(
    val_dir_path,
    dim=2,
    nxs=[
        60,
    ],
    save_fig=False,
):
    def u_series_exp(y, t, n_max=10):
        """Analytical solution to unsteady Poiseuille flow (low Re)

        Based on Series expansion as shown in:
        "Modeling Low Reynolds Number Incompressible Flows Using SPH"
        ba Morris et al. 1997
        """

        eta = 100.0  # dynamic viscosity
        rho = 1.0  # denstiy
        nu = eta / rho  # kinematic viscosity
        u_max = 1.25  # max velocity in middle of channel
        d = 1.0  # channel width

        Re = u_max * d / nu
        print(f"Poiseuille flow at Re={Re}")

        fx = -8 * nu * u_max / d**2
        offset = fx / (2 * nu) * y * (y - d)

        def term(n):
            base = np.pi * (2 * n + 1) / d

            prefactor = 4 * fx / (nu * base**3 * d)
            sin_term = np.sin(base * y)
            exp_term = np.exp(-(base**2) * nu * t)
            return prefactor * sin_term * exp_term

        res = offset
        for i in range(n_max):
            res += term(i)

        return res

    # analytical solution

    y_axis = np.linspace(0, 1, 100)
    t_axis = [
        r"$0.02\times 10^{-2}$",
        r"$0.10\times 10^{-2}$",
        r"$0.20\times 10^{-2}$",
        r"$1.00\times 10^{-2}$",
    ]
    t_dimless = [0.0002, 0.001, 0.002, 0.01]

    for t_val, t_label in zip(t_dimless, t_axis):
        plt.plot(y_axis, u_series_exp(y_axis, t_val), label=f"t={t_label}")

    # extract points from our solution
    dirs = os.listdir(val_dir_path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(val_dir_path, d))]
    assert len(dirs) == 1, f"Expected only one directory in {val_dir_path}"
    args = read_args(os.path.join(val_dir_path, dirs[0], "args.txt"))
    assert args.dt == 0.0000005
    assert args.dx == 0.0166666

    num_points = 21
    dx_plot = 0.05
    y_axis = jnp.array([dx_plot * i for i in range(num_points)]) + 3 * args.dx
    rs = 0.2 * jnp.ones([y_axis.shape[0], 2])
    rs = rs.at[:, 1].set(y_axis)

    step_max = np.array(np.rint(args.t_end / args.dt), dtype=int)
    digits = len(str(step_max))

    for i, t_val in enumerate(t_dimless):
        step = np.array(np.rint(t_val / args.dt), dtype=int)
        file_name = "traj_" + str(step).zfill(digits) + ".h5"
        src_path = os.path.join(val_dir_path, dirs[0], file_name)

        if i == 0:
            interp_vel_fn = sph_interpolator(args, src_path)

        u_val = interp_vel_fn(src_path, rs, prop="u", dim_ind=0)

        if i == 0:
            plt.plot(
                y_axis - 3 * args.dx, u_val, "ko", mfc="none", label=r"SPH, $r_c$=0.05"
            )
        else:
            plt.plot(y_axis - 3 * args.dx, u_val, "ko", mfc="none")

    # plot layout

    plt.legend()
    plt.ylim([0, 1.4])
    plt.xlim([0, 1])
    plt.xlabel(r"y [-]")
    plt.ylabel(r"$u_x$ [-]")
    # plt.title(f"{str(dim)}D Poiseuille Flow")
    plt.grid()
    plt.tight_layout()

    ###### save or visualize

    if save_fig:
        os.makedirs(val_dir_path, exist_ok=True)
        nxs_str = "_".join([str(i) for i in nxs])
        plt.savefig(f"{val_dir_path}/{str(dim)}D_PF_{nxs_str}_new.png")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, help="One of the above cases")
    parser.add_argument("--src_dir", type=str, help="Source directory")
    args = parser.parse_args()

    if args.case == "2D_PF":
        val_2D_PF(args.src_dir, 2, [60], True)

    elif args.case == "2D_LDC":
        val_2D_LDC(args.src_dir, save_fig=True)

    elif args.case == "2D_TGV":
        val_TGV(args.src_dir, 2, [20, 50, 100, 200], True)
    elif args.case == "3D_TGV":
        val_TGV(args.src_dir, 3, [20, 40], True)

    elif args.case == "2D_DB":
        val_DB(args.src_dir, save_fig=True)
