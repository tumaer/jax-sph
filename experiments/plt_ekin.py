"""Utility to plot kinetic enery from h5 simulation trajectories"""

import enum
import json
import os

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# copy-paste functions to avoid importing JAX
# from utils import SPH, TGV, PF, RPF, LDC, CW, DB, Rlx, get_ekin, get_val_max
# from src.io_state import read_h5, render_data_dict, _plot, write_vtk


class Tag(enum.IntEnum):
    """Particle types."""

    PAD_VALUE = -1
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    DIRICHLET_WALL = 3


def get_ekin(state, dx):
    v = state["v"]
    v_water = np.where(state["tag"][:, None] == Tag.FLUID, v, 0)
    ekin = np.square(v_water).sum().item()
    return 0.5 * ekin * dx ** v.shape[1]


def get_val_max(state, var="u"):
    "Extract the largest velocity magnitude, needed for TGV"
    return np.sqrt(np.square(state[var]).sum(axis=1)).max()


def read_h5(file_name, array_type="jax"):
    hf = h5py.File(file_name, "r")

    data_dict = {}
    for k, v in hf.items():
        if array_type == "jax":
            data_dict[k] = jnp.array(v)
        elif array_type == "numpy":
            data_dict[k] = np.array(v)
        else:
            raise ValueError('array_type must be either "jax" or "numpy"')

    hf.close()

    return data_dict


def main(src_dir):
    # metadata
    with open(os.path.join(src_dir, "args.txt"), "r") as f:
        args_dict = json.load(f)

    run_name = src_dir.split("/")[-1]
    fig_root = os.path.join(*src_dir.split("/")[:-1])
    # split one long trajectory into train, valid, and test
    files = os.listdir(src_dir)
    files = [f for f in files if (".h5" in f)]
    files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
    if len(files) > 10000:
        files = files[::10]

    e_kin = np.zeros(len(files))
    u_max = np.zeros(len(files))
    u_min = np.zeros(len(files))
    for j, filename in enumerate(files):
        file_path_h5 = os.path.join(src_dir, filename)
        state = read_h5(file_path_h5, array_type="numpy")
        e_kin[j] = get_ekin(state, args_dict["dx"])
        u_max[j] = state["u"].max()
        u_min[j] = state["u"].min()

    if "3D_TGV" in src_dir:
        e_kin /= (2 * np.pi) ** 3
    dEdt = -(e_kin[1:] - e_kin[:-1]) / (args_dict["dt"] * args_dict["write_every"])
    Nx = round(
        (args_dict["bounds"][0][1] - args_dict["bounds"][0][0]) / args_dict["dx"]
    )

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    ax[0].plot(e_kin, label=f"SPH Nx={Nx}")
    ax[0].set_title("E_kin")
    ax[1].plot(u_max, label="u_max")
    ax[1].plot(-u_min, label="-u_min")
    ax[1].legend()
    ax[1].set_title("u_max")
    ax[2].plot(dEdt, label=f"SPH Nx={Nx}")
    ax[2].set_title("d(E_kin)/dt")
    try:
        assert "tgv" in src_dir.lower()
        ax[0].set_yscale("log")

        Re = 1 / args_dict["viscosity"]
        dEdt_ref = np.loadtxt(
            f"./validation/ref/tgv3d_ref_{int(Re)}.txt", delimiter=","
        )
        every_n = max(len(dEdt_ref) // 50, 1)
        # TODO: rescale x axis alltogether
        x_ref = (
            dEdt_ref[::every_n, 0]
            / dEdt_ref[::every_n, 0].max()
            * len(dEdt)
            / (args_dict["t_end"] / 10)
        )
        ax[2].scatter(
            x_ref,
            dEdt_ref[::every_n, 1],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )
        ax[0].scatter(
            x_ref,
            dEdt_ref[::every_n, 2],
            s=20,
            edgecolors="k",
            lw=1,
            facecolors="none",
            label="Jax-fluids Nx=64",
        )
    except FileNotFoundError:
        print("Not working with TGV or no reference data available")

    ax[0].legend()
    ax[2].legend()

    for a in ax:
        a.grid()
    plt.savefig(os.path.join(fig_root, f"{run_name}_plots.png"))
    plt.close()

    print(f"Finished {src_dir}!")


if __name__ == "__main__":
    main("datasets/2D_RPF_800_100kevery10/2D_RPF_SPH_0_20230525-233534")
