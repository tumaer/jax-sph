"""advect 2D TGV particles with ground truth velocity field"""

import matplotlib.pyplot as plt
import numpy as np


def pos_init_cartesian_2d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r


def vel_fn(r, i):
    x = r[:, 0]
    y = r[:, 1]
    res_x = -1.0 * np.cos(2.0 * np.pi * x) * np.sin(2.0 * np.pi * y)
    res_y = +1.0 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    return i / 10000 * np.stack([res_x, res_y]).T


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()

    dx_mesh = 0.005
    nx = round(1 / dx_mesh)
    x_axis, y_axis = np.arange(0.0, 1.0, dx_mesh), np.arange(0.0, 1.0, dx_mesh)
    X, Y = np.meshgrid(x_axis, y_axis, indexing="xy")
    Z = vel_fn(np.stack([X, Y], axis=-1).reshape(-1, 2), 0)
    Z = np.linalg.norm(Z, axis=1).reshape(nx, nx)
    for ax in axs:
        ax.pcolormesh(X, Y, Z, alpha=0.5)
        # ax.colorbar()

    r = pos_init_cartesian_2d(np.array([1.0, 1.0]), 0.01)  # shape (2500, 2)
    counter = 0
    for i in range(80000):
        v = vel_fn(r, i)
        r += 0.0001 * v
        r = np.mod(r, 1.0)
        if i % 8000 == 0:
            axs[counter].scatter(r[:, 0], r[:, 1], s=2.0)
            counter += 1

    dx_quiver = 0.05
    r_quiver = pos_init_cartesian_2d(np.array([1.0, 1.0]), dx_quiver)  # shape (2500, 2)
    v_quiver = vel_fn(r_quiver, 0)
    # v_mag = np.linalg.norm(v_quiver, axis=1)
    for ax in axs:
        ax.quiver(
            r_quiver[:, 0], r_quiver[:, 1], v_quiver[:, 0], v_quiver[:, 1], width=0.01
        )  # , headwidth=12, headlength=20)
        # plt.scatter(r[:, 0], r[:, 1], c=v_mag, cmap='jet', s=2.0)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()
