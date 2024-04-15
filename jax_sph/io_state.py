"""Input-output utilities."""

import json
import os
import time
from argparse import Namespace
from typing import Dict

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista


def write_args(args: Namespace, file_path: str):
    """Write argparse arguments to a .txt file using json.

    Args:
        args: full set of input arguments incl. defaults
        file_path: e.g. "./data/3D_TGV_<...>/args.txt"
    """

    with open(file_path, "w") as f:
        json.dump(vars(args), f)


def read_args(file_path: str):
    """Read argparse arguments from a .txt file using json."""
    with open(file_path, "r") as f:
        args_dict = json.load(f)
    args = Namespace(**(args_dict))
    return args


def io_setup(args: Namespace):
    """Setup the output directory and write the arguments to a .txt file."""
    case, solver, dim = args.case, args.solver, args.dim
    dir = args.data_path
    if not dir.endswith("/"):
        dir += "/"
    if (args.write_h5 or args.write_vtk) and args.mode != "rlx":
        dir += str(dim) + "D_" + case + "_" + solver + "_" + str(args.seed)
        dir += "_" + time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(dir, exist_ok=True)
    write_args(args, os.path.join(dir, "args.txt"))

    return dir


def write_h5(data_dict: Dict, path: str):
    """Write a dict of numpy or jax arrays to a .h5 file."""
    hf = h5py.File(path, "w")
    for k, v in data_dict.items():
        hf.create_dataset(k, data=np.array(v))
    hf.close()


def write_vtk(data_dict: Dict, path: str):
    """Store a .vtk file for ParaView."""
    data_pv = dict2pyvista(data_dict)
    data_pv.save(path)


def write_state(step: int, step_max: int, state: Dict, dir: str, args: Namespace):
    """Write state to .h5 or .vtk file while simulation is running."""
    write_normal = (
        (args.mode != "rlx") and ((step % args.write_every) == 0) and (step >= 0)
    )
    write_relax = (args.mode == "rlx") and (step == (step_max - 1))

    if write_normal or write_relax:
        digits = len(str(step_max))
        step_str = str(step).zfill(digits)

        if args.mode == "rlx":
            name = [args.case.lower(), str(args.dim), str(args.dx), str(args.seed)]
            name = "_".join(name)
        else:
            name = "traj_" + step_str

        if args.write_h5:
            path = os.path.join(dir, name + ".h5")
            write_h5(state, path)
        if args.write_vtk:
            path = os.path.join(dir, name + ".vtk")
            write_vtk(state, path)


def read_h5(file_name: str, array_type: str = "jax"):
    """Read an .h5 file and return a dict of numpy or jax arrays."""
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


def write_vtks_from_h5s(dir_path: str, keep_h5: bool = True):
    """Transform a set of .h5 files to .vtk files.

    Args:
        path: path to directory with .h5 files
        keep_h5: Whether to keep or delete the original .h5 files.
    """

    files = os.listdir(dir_path)
    # only regard h5 files
    files = [f for f in files if (".h5" in f)]
    files = sorted(files)

    for i, filename in enumerate(files):
        file_path_h5 = os.path.join(dir_path, filename)
        file_path_vtk = file_path_h5[:-2] + "vtk"
        state = read_h5(file_path_h5)
        write_vtk(state, file_path_vtk)

        if not keep_h5:
            os.remove(file_path_h5)


def render_data_dict(data_dict):
    pl = pyvista.Plotter()
    pl.show_grid()
    pl.add_bounding_box()
    pl.add_axes()
    pl.camera_position = "xy"
    pl.show()


def dict2pyvista(data_dict):
    # PyVista works only with 3D objects, thus we check whether the inputs
    # are 2D and then increase the degrees of freedom of the second dimension.
    # N is the number of points and dim the dimension
    r = np.asarray(data_dict["r"])
    N, dim = r.shape

    # PyVista treats the position information differently than the rest
    if dim == 2:
        r = np.hstack([r, np.zeros((N, 1))])
    data_pv = pyvista.PolyData(r)

    # copy all the other information also to pyvista, using plain numpy arrays
    for k, v in data_dict.items():
        # skip r because we already considered it above
        if k == "r":
            continue

        # working in 3D or scalar features do not require special care
        if dim == 2 and v.ndim == 2:
            v = np.hstack([v, np.zeros((N, 1))])

        data_pv[k] = np.asarray(v)

    return data_pv


def _plot(r, x, ttl, vmin=None, vmax=None):
    plt.figure()
    plt.scatter(r[:, 0], r[:, 1], c=np.array(x), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(ttl)
    plt.show()


if __name__ == "__main__":
    # write_vtks_from_h5s("./3D_TGV_SPH_17_20221224-084051", False)
    pass
