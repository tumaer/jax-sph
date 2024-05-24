"""Utility to reproduce the plot of the thermal diffusion problem from the paper."""

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from jax_sph.io_state import read_h5
from jax_sph.utils import Tag


def paper_plots(data_dir, plt_dir, steps):
    """Create the three plots from the paper appendix."""

    metadata = OmegaConf.load(os.path.join(data_dir, "config.yaml"))

    for idx in steps:
        filename = f"{data_dir}/traj_{idx:04d}.h5"
        data_dict = read_h5(filename, array_type="numpy")
        r = data_dict["r"]
        v = data_dict["T"]
        tag = data_dict["tag"]
        wall_mask = tag == Tag.SOLID_WALL
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(
            r[~wall_mask][:, 0],
            r[~wall_mask][:, 1],
            vmin=0.93,
            vmax=1.27,
            c=v[~wall_mask],
            cmap="turbo",
            s=20,
        )
        ax.scatter(r[wall_mask][:, 0], r[wall_mask][:, 1], c="black", s=20)
        ax.set_xlim(metadata.case.bounds[0])
        ax.set_ylim(metadata.case.bounds[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid()

        # add colorbar right of the image
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        cax = fig.add_axes([0.87, 0.2, 0.02, 0.75])

        cbar = fig.colorbar(
            ax.collections[0],
            cax=cax,
            orientation="vertical",
            ticks=[1.0, 1.1, 1.2],
            boundaries=np.linspace(1.0, 1.2, 100),
            values=np.linspace(1.0, 1.2, 99),
        )

        cbar.ax.tick_params(
            labelsize=15,
            length=6,
            width=2,
        )

        os.makedirs(plt_dir, exist_ok=True)
        fig.savefig(f"{plt_dir}/HT_T_{idx}.pdf", bbox_inches="tight")
        plt.show()
        plt.close()


def docs_gif(data_dir, plt_dir):
    """Create a gif over all 3000 steps of the simulation."""

    metadata = OmegaConf.load(os.path.join(data_dir, "config.yaml"))
    bounds = metadata.case.bounds

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".h5")]
    # sort files by index
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    files = files[::2]

    fig = plt.figure(figsize=(bounds[0][1] * 10, bounds[1][1] * 10))
    ax = plt.axes(xlim=(bounds[0]), ylim=(bounds[1]))

    # Function to plot a single frame
    def update_plot(i):
        # get data
        filename = os.path.join(data_dir, files[i])
        data_dict = read_h5(filename, array_type="numpy")
        r = data_dict["r"]
        v = data_dict["T"]  # temperature
        tag = data_dict["tag"]
        wall_mask = tag == Tag.SOLID_WALL

        # plot data
        ax.clear()
        ax.scatter(
            r[~wall_mask][:, 0],
            r[~wall_mask][:, 1],
            vmin=0.93,
            vmax=1.27,
            c=v[~wall_mask],
            cmap="turbo",
            s=20,
        )
        ax.scatter(r[wall_mask][:, 0], r[wall_mask][:, 1], c="black", s=20)
        ax.set_xlim(metadata.case.bounds[0])
        ax.set_ylim(metadata.case.bounds[1])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid()

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    anim = animation.FuncAnimation(fig, update_plot, frames=len(files), interval=20)

    os.makedirs(plt_dir, exist_ok=True)
    anim.save(f"{plt_dir}/anim.gif", writer="imagemagick", fps=30, dpi=50)
    plt.show()
    plt.close()


if __name__ == "__main__":
    data_dir = "data/2D_HT_SPH_123_20240308-154911"
    plt_dir = "data/2D_TH_plots"

    paper_plots(data_dir, plt_dir, steps=[0, 1000, 3000])
    docs_gif(data_dir, plt_dir)
