"""Extract information from Water 3D Deep Mind dataset"""

# place to work with the .tfrecord files from the DM "Learning to Simulate
# [...] paper"

# np.savez('GNS/data/Water-3D/details_water_3D.npz',
# avg_dist = np.array([np.abs(r[0][i+1] - r[0][i]) for i in np.arange(len(r[0]) - 1)]),
#     x_dim = np.array([r[:, :, 0].min(), r[:, :, 0].max()]),
#     y_dim = np.array([r[:, :, 1].min(), r[:, :, 1].max()]),
#     z_dim = np.array([r[:, :, 2].min(), r[:, :, 2].max()]),
#     velocities = velocities)

import argparse

# import functools
# import json
import os

import h5py
import numpy as np

# import tensorflow.compat.v1 as tf
# import tensorflow_datasets as tfds


def read_h5_demo(args):
    """Simple demonstation how to read the previously converted h5 file"""

    file_path = os.path.join(args.dataset_path, args.file_name)
    hf = h5py.File(f"{file_path[:-9]}.h5", "r")

    velo_init = {}
    velo_max = {}
    init_lim_min, init_lim_max = {}, {}
    n_particles_per_dim = {}
    x_offset, y_offset, z_offset = {}, {}, {}
    box = {}

    for key in hf:
        r = hf[f"{key}/r"][:]
        # print(key, r.shape, r.mean())
        velo_init[int(key[-2:])] = (r[1] - r[0]) / 0.005
        velo_max[int(key[-2:])] = np.amax(velo_init[int(key[-2:])], axis=0)
        init_lim_min[int(key[-2:])] = np.amin(r[0], axis=0)
        init_lim_max[int(key[-2:])] = np.amax(r[0], axis=0)
        n_particles_per_dim[int(key[-2:])] = np.round(
            (init_lim_max[int(key[-2:])] - init_lim_min[int(key[-2:])]) / 0.024
        )
        x_offset[int(key[-2:])] = np.unique(r[0][:, 0])
        y_offset[int(key[-2:])] = np.unique(r[0][:, 1])
        z_offset[int(key[-2:])] = np.unique(r[0][:, 2])
        box[int(key[-2:])] = r[0]

    np.savez(
        "details_water_3D.npz",
        x_dim=np.array([r[:, :, 0].min(), r[:, :, 0].max()]),
        y_dim=np.array([r[:, :, 1].min(), r[:, :, 1].max()]),
        z_dim=np.array([r[:, :, 2].min(), r[:, :, 2].max()]),
        velo_init=velo_init,
        velo_max=velo_max,
        init_lim_min=init_lim_min,
        init_lim_max=init_lim_max,
        n_particles_per_dim=n_particles_per_dim,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        box=box,
    )

    #  use .item() method to access dictionary from inside a numpy array when loading
    # the .npz file
    hf.close()


if __name__ == "__main__":
    """Environment setup:

    python3 -m venv venv_tf
    source venv_tf/bin/activate
    pip install tensorflow tensorflow-datasets

    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data_relaxed/GNS/data/Water-3D",
        help="Location of the dataset",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="valid.tfrecord",
        help="Which file to convert from .tfrecord to .h5",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="None",
        choices=["None", "gzip"],
        help='"gzip" takes 8x longer, but reduces size by 15%',
    )
    args = parser.parse_args()
    args.compression = "gzip" if args.compression == "gzip" else None

    # convert_tfrecord_to_h5(args)
    read_h5_demo(args)
