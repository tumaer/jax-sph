"""Script for generating ML datasets from h5 simulation frames"""

import argparse
import json
import os

import h5py
import numpy as np
from jax import vmap
from jax_md import space

from jax_sph.io_state import read_args, read_h5, write_h5


def write_h5_frame_for_visualization(state_dict, file_path_h5):
    path_file_vis = os.path.join(file_path_h5[:-3] + "_vis.h5")
    print("writing to", path_file_vis)
    write_h5(state_dict, path_file_vis)
    print("done")


def single_h5_files_to_h5_dataset(args):
    """Transform a set of .h5 files to a single .h5 dataset file

    Args:
        src_dir: source directory containing other directories, each with .h5 files
            corresponding to a trajectory
        dst_dir: destination directory where three files will be written: train.h5,
            valid.h5, and test.h5
        split: string of three integers separated by underscores, e.g. "80_10_10"
    """

    os.makedirs(args.dst_dir, exist_ok=True)

    # list only directories in a root with files and directories
    dirs = os.listdir(args.src_dir)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(args.src_dir, d))]
    # order by seed value
    dirs = sorted(dirs, key=lambda x: int(x.split("_")[3]))

    splits_array = np.array([int(s) for s in args.split.split("_")])
    splits_sum = splits_array.sum()

    if len(dirs) == 1:
        # split one long trajectory into train, valid, and test
        files = os.listdir(os.path.join(args.src_dir, dirs[0]))
        files = [f for f in files if (".h5" in f)]
        files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
        files = files[args.skip_first_n_frames :: args.slice_every_nth_frame]

        num_eval = np.ceil(splits_array[1] / splits_sum * len(files)).astype(int)
        # at least one validation and one testing trajectory
        splits_trajs = np.cumsum([0, len(files) - 2 * num_eval, num_eval, num_eval])
    else:
        num_eval = np.ceil(splits_array[1] / splits_sum * len(dirs)).astype(int)
        # at least one validation and one testing trajectory
        splits_trajs = np.cumsum([0, len(dirs) - 2 * num_eval, num_eval, num_eval])

    for i, split in enumerate(["train", "valid", "test"]):
        hf = h5py.File(f"{args.dst_dir}/{split}.h5", "w")

        if len(dirs) == 1:  # one long trajectory
            position = []
            traj_path = os.path.join(args.src_dir, dirs[0])

            for j, filename in enumerate(files[splits_trajs[i] : splits_trajs[i + 1]]):
                file_path_h5 = os.path.join(traj_path, filename)
                state = read_h5(file_path_h5, array_type="numpy")
                r = state["r"]
                tag = state["tag"]

                if "ldc" in args.src_dir.lower():
                    L, H = 1.0, 1.0
                    args_sim = read_args(os.path.join(traj_path, "args.txt"))
                    mask_bottom = np.where(r[:, 1] < 2 * args_sim.dx, False, True)
                    mask_lid = np.where(r[:, 1] > H + 4 * args_sim.dx, False, True)
                    mask_left = np.where(
                        ((r[:, 0] < 2 * args_sim.dx) * (tag == 1)), False, True
                    )
                    mask_right = np.where(
                        (r[:, 0] > L + 4 * args_sim.dx) * (tag == 1), False, True
                    )
                    mask = mask_bottom * mask_lid * mask_left * mask_right

                    r = r[mask]
                    tag = tag[mask]

                if args.is_visualize:
                    write_h5_frame_for_visualization({"r": r, "tag": tag}, file_path_h5)
                position.append(r)

            position = np.stack(position)  # (time steps, particles, dim)
            particle_type = tag  # (particles,)

            traj_str = "00000"
            hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
            hf.create_dataset(
                f"{traj_str}/position",
                data=position,
                dtype=np.float32,
                compression="gzip",
            )

        else:  # multiple trajectories
            for j, dir in enumerate(dirs[splits_trajs[i] : splits_trajs[i + 1]]):
                traj_path = os.path.join(args.src_dir, dir)
                files = os.listdir(traj_path)
                files = [f for f in files if (".h5" in f)]
                files = sorted(files, key=lambda x: int(x.split("_")[1][:-3]))
                files = files[args.skip_first_n_frames :: args.slice_every_nth_frame]

                position = []
                for k, filename in enumerate(files):
                    file_path_h5 = os.path.join(traj_path, filename)
                    state = read_h5(file_path_h5, array_type="numpy")
                    r = state["r"]
                    tag = state["tag"]

                    if "db" in args.src_dir.lower():
                        L, H = 5.366, 2.0
                        args_sim = read_args(os.path.join(traj_path, "args.txt"))
                        mask_bottom = np.where(r[:, 1] < 2 * args_sim.dx, False, True)
                        mask_lid = np.where(r[:, 1] > H + 4 * args_sim.dx, False, True)
                        mask_left = np.where(
                            ((r[:, 0] < 2 * args_sim.dx) * (tag == 1)), False, True
                        )
                        mask_right = np.where(
                            (r[:, 0] > L + 4 * args_sim.dx) * (tag == 1), False, True
                        )
                        mask = mask_bottom * mask_lid * mask_left * mask_right

                        r = r[mask]
                        tag = tag[mask]

                    if args.is_visualize:
                        write_h5_frame_for_visualization(
                            {"r": r, "tag": tag}, file_path_h5
                        )
                    position.append(r)
                position = np.stack(position)  # (time steps, particles, dim)
                particle_type = tag  # (particles,)

                traj_str = str(j).zfill(5)
                hf.create_dataset(f"{traj_str}/particle_type", data=particle_type)
                hf.create_dataset(
                    f"{traj_str}/position",
                    data=position,
                    dtype=np.float32,
                    compression="gzip",
                )

        hf.close()
        print(f"Finished {args.src_dir} {split} with {j+1} entries!")
        print(f"Sample positions shape {position.shape}")

    # metadata
    with open(os.path.join(traj_path, "args.txt"), "r") as f:
        args_dict = json.load(f)

    x = 1.45 * args_dict["dx"]
    x = float(
        np.format_float_positional(
            x, precision=2, unique=False, fractional=False, trim="k"
        )
    )
    args_dict["default_connectivity_radius"] = x
    # seqience_length should be after subsampling every nth trajectory
    # and "-1" for the last target position (see GNS dataset format)
    args_dict["sequence_length"] = position.shape[0] - 1

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(args_dict, f)


def compute_statistics_h5(args):
    """Compute the mean and std of a h5 dataset files"""

    # metadata
    with open(os.path.join(args.dst_dir, "metadata.json"), "r") as f:
        args_dict = json.load(f)

    # apply PBC in all directions or not at all
    if np.array(args_dict["periodic_boundary_conditions"]).any():
        box = np.array(args_dict["bounds"])
        box = box[:, 1] - box[:, 0]
        displacement_fn, _ = space.periodic(side=box)
    else:
        displacement_fn, _ = space.free()

    displacement_fn_sets = vmap(vmap(displacement_fn, in_axes=(0, 0)))

    vels, accs = [], []
    vels_sq, accs_sq = [], []
    for loop in ["mean", "std"]:
        for split in ["train", "valid", "test"]:
            hf = h5py.File(f"{args.dst_dir}/{split}.h5", "r")

            for _, v in hf.items():
                tag = v.get("particle_type")[:]
                r = v.get("position")[:][:, tag == 0]

                # The velocity and acceleration computation is based on an
                # inversion of Semi-Implicit Euler
                vel = displacement_fn_sets(r[1:], r[:-1])
                if loop == "mean":
                    vels.append(vel.mean((0, 1)))
                    accs.append((vel[1:] - vel[:-1]).mean((0, 1)))
                elif loop == "std":
                    centered_vel = vel - vel_mean
                    vels_sq.append(np.square(centered_vel).mean((0, 1)))
                    centered_acc = vel[1:] - vel[:-1] - acc_mean
                    accs_sq.append(np.square(centered_acc).mean((0, 1)))

            hf.close()

        if loop == "mean":
            vel_mean = np.stack(vels).mean(0)
            acc_mean = np.stack(accs).mean(0)
            print(f"vel_mean={vel_mean}, acc_mean={acc_mean}")
        elif loop == "std":
            vel_std = np.stack(vels_sq).mean(0) ** 0.5
            acc_std = np.stack(accs_sq).mean(0) ** 0.5
            print(f"vel_std={vel_std}, acc_std={acc_std}")

    # stds should not be 0. If they are, set them to 1.
    vel_std = np.where(vel_std < 1e-7, 1, vel_std)
    acc_std = np.where(acc_std < 1e-7, 1, acc_std)

    args_dict["vel_mean"] = vel_mean.tolist()
    args_dict["vel_std"] = vel_std.tolist()

    args_dict["acc_mean"] = acc_mean.tolist()
    args_dict["acc_std"] = acc_std.tolist()

    with open(os.path.join(args.dst_dir, "metadata.json"), "w") as f:
        json.dump(args_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--dst_dir", type=str)
    parser.add_argument("--split", type=str, help="E.g. 3_1_1")
    parser.add_argument("--skip_first_n_frames", type=int, default=0)
    parser.add_argument("--slice_every_nth_frame", type=int, default=1)
    parser.add_argument("--is_visualize", action="store_true")
    args = parser.parse_args()

    single_h5_files_to_h5_dataset(args)
    compute_statistics_h5(args)
