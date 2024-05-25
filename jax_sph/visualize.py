"""Visualization tools for SPH simulations."""

import os

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from jax_sph.io_state import read_h5
from jax_sph.utils import get_ekin


def plt_scatter(r, x, ttl, vmin=None, vmax=None):
    plt.figure()
    plt.scatter(r[:, 0], r[:, 1], c=np.array(x), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(ttl)
    plt.show()


def plt_ekin(src_dir):
    # metadata
    cfg = OmegaConf.load(os.path.join(src_dir, "config.yaml"))

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
        e_kin[j] = get_ekin(state, cfg.case.dx)
        u_max[j] = state["u"].max()
        u_min[j] = state["u"].min()

    if "3D_TGV" in src_dir:
        e_kin /= (2 * np.pi) ** 3
    dEdt = (e_kin[1:] - e_kin[:-1]) / (cfg.solver.dt * cfg.io.write_every)
    Nx = round((cfg.case.bounds[0][1] - cfg.case.bounds[0][0]) / cfg.case.dx)

    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].plot(e_kin, label=f"SPH Nx={Nx}")
    ax[0].set_title(r"$E_{kin}$")
    ax[1].plot(u_max, label=r"$u_{max}$")
    ax[1].plot(-u_min, label=r"$-u_{min}$")
    ax[1].legend()
    ax[1].set_title(r"$u_{max}$")
    ax[2].plot(dEdt, label=f"SPH Nx={Nx}")
    ax[2].set_title(r"$d(E_{kin})/dt$")
    if "tgv" in src_dir.lower():
        try:
            ax[0].set_yscale("log")

            Re = 1 / cfg.case.viscosity
            dEdt_ref = np.loadtxt(
                f"./validation/ref/tgv3d_ref_{int(Re)}.txt", delimiter=","
            )
            every_n = max(len(dEdt_ref) // 50, 1)
            x_ref = (
                dEdt_ref[::every_n, 0]
                / dEdt_ref[::every_n, 0].max()
                * len(dEdt)
                / (cfg.solver.t_end / 10)
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
    plt.tight_layout()

    plt.savefig(os.path.join(fig_root, f"{run_name}_plots.png"))
    plt.show()
    plt.close()

    print(f"Finished {src_dir}!")


if __name__ == "__main__":
    plt_ekin("notebooks/data/debug/2D_HT_SPH_123_20240525-020025")
