import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_sph.io_state import read_args
from jax_sph.utils import sph_interpolator


# function to plot the temperature at 6*dx from the outlet of the channel
def plot_outlet_temperature(
    val_dir_path,
    dim=2,
    save_fig=False,
):
    # extract points from our solution
    dirs = os.listdir(val_dir_path)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(val_dir_path, d))]
    assert len(dirs) == 1, f"Expected only one directory in {val_dir_path}"
    args = read_args(os.path.join(val_dir_path, dirs[0], "args.txt"))

    dx_plot = 0.002
    y_length = args.bounds[1][1] - 6 * args.dx
    num_points = int(y_length / dx_plot) + 1
    y_axis = jnp.array([dx_plot * i for i in range(num_points)]) + 3 * args.dx
    rs = (args.bounds[0][1] - 6 * args.dx) * jnp.ones([y_axis.shape[0], 2])
    rs = rs.at[:, 1].set(y_axis)

    step_max = np.array(np.rint(args.t_end / args.dt), dtype=int)
    digits = len(str(step_max))

    t_val = args.t_end  # enter the end time

    step = np.array(np.rint(t_val / args.dt), dtype=int)

    file_name = "traj_" + str(step).zfill(digits) + ".h5"
    src_path = os.path.join(val_dir_path, dirs[0], file_name)

    interp_scalar_fn = sph_interpolator(args, src_path, prop_type="scalar")

    temp_val = interp_scalar_fn(src_path, rs, prop="T")

    plt.plot(y_axis - 3 * args.dx, temp_val, mfc="none")
    plt.ylim([1, 1.3])
    plt.xlim([0, 0.2])
    plt.xlabel(r"y [-]")
    plt.ylabel(r"$Temperature$ [-]")
    plt.title(f"{str(dim)}D Channel Flow with Heat Transfer")
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    val_dir_path = "results"
    plot_outlet_temperature(val_dir_path, dim=2, save_fig=False)
