"""Input-output utilities."""

import os
import time
from typing import Dict

import h5py
import jax.numpy as jnp
import numpy as np
import pyvista
from omegaconf import DictConfig, OmegaConf


def io_setup(cfg: DictConfig):
    """Setup the output directory and write the arguments to a .txt file."""
    case, solver, dim = cfg.case.name.upper(), cfg.solver.name, cfg.case.dim
    dir = cfg.io.data_path
    if not dir.endswith("/"):
        dir += "/"
    if len(cfg.io.write_type) > 0 and cfg.case.mode == "sim":
        dir += str(dim) + "D_" + case + "_" + solver + "_" + str(cfg.seed)
        dir += "_" + time.strftime("%Y%m%d-%H%M%S")

    os.makedirs(dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(dir, "config.yaml"))

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


def write_state(step: int, state: Dict, dir: str, cfg: DictConfig):
    """Write state to .h5 or .vtk file while simulation is running."""
    step_max = cfg.solver.sequence_length
    write_normal = (
        (cfg.case.mode == "sim") and ((step % cfg.io.write_every) == 0) and (step >= 0)
    )
    write_relax = (cfg.case.mode == "rlx") and (step == (step_max - 1))

    if write_normal or write_relax:
        digits = len(str(step_max))
        step_str = str(step).zfill(digits)

        if cfg.case.mode == "rlx":
            name = [cfg.case.name, str(cfg.case.dim), str(cfg.case.dx), str(cfg.seed)]
            name = "_".join(name)  # e.g. "tgv_3_0.02_42"
        else:
            name = "traj_" + step_str

        if "h5" in cfg.io.write_type:
            path = os.path.join(dir, name + ".h5")
            write_h5(state, path)
        if "vtk" in cfg.io.write_type:
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


if __name__ == "__main__":
    # write_vtks_from_h5s("./3D_TGV_SPH_17_20221224-084051", False)
    pass
