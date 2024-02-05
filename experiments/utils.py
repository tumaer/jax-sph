import os
import os.path as osp
from argparse import Namespace
from typing import Tuple

from lagrangebench.data import H5Dataset


def setup_data(args: Namespace) -> Tuple[H5Dataset, H5Dataset, Namespace]:
    if not osp.isabs(args.config.data_dir):
        args.config.data_dir = osp.join(os.getcwd(), args.config.data_dir)

    args.info.dataset_name = osp.basename(args.config.data_dir.split("/")[-1])
    if args.config.ckp_dir is not None:
        os.makedirs(args.config.ckp_dir, exist_ok=True)
    if args.config.rollout_dir is not None:
        os.makedirs(args.config.rollout_dir, exist_ok=True)

    # dataloader
    data_train = H5Dataset(
        "train",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        nl_backend=args.config.neighbor_list_backend,
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        extra_seq_length=args.config.n_rollout_steps,
        nl_backend=args.config.neighbor_list_backend,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=args.config.data_dir,
        input_seq_length=args.config.input_seq_length,
        extra_seq_length=args.config.n_rollout_steps,
        nl_backend=args.config.neighbor_list_backend,
    )
    if args.config.eval_n_trajs == -1:
        args.config.eval_n_trajs = data_valid.num_samples
    if args.config.eval_n_trajs_infer == -1:
        args.config.eval_n_trajs_infer = data_valid.num_samples
    assert data_valid.num_samples >= args.config.eval_n_trajs, (
        f"Number of available evaluation trajectories ({data_valid.num_samples}) "
        f"exceeds eval_n_trajs ({args.config.eval_n_trajs})"
    )

    args.info.has_external_force = bool(data_train.external_force_fn is not None)

    return data_train, data_valid, data_test, args
