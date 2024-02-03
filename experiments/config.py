import argparse
import os
from typing import Dict

import yaml


def cli_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)

    # config arguments
    group.add_argument("-c", "--config", type=str, help="Path to the config yaml.")
    group.add_argument("--model_dir", type=str, help="Path to the model checkpoint.")

    # run arguments
    parser.add_argument(
        "--mode", type=str, choices=["train", "infer", "all"], help="Train or evaluate."
    )
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size.")
    parser.add_argument(
        "--lr_start", type=float, required=False, help="Starting learning rate."
    )
    parser.add_argument(
        "--lr_final", type=float, required=False, help="Learning rate after decay."
    )
    parser.add_argument(
        "--lr_decay_rate", type=float, required=False, help="Learning rate decay."
    )
    parser.add_argument(
        "--lr_decay_steps", type=int, required=False, help="Learning rate decay steps."
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        required=False,
        help="Additive noise standard deviation.",
    )
    parser.add_argument(
        "--test",
        action=argparse.BooleanOptionalAction,
        help="Run test mode instead of validation.",
    )
    parser.add_argument("--seed", type=int, required=False, help="Random seed.")
    parser.add_argument(
        "--data_dir", type=str, help="Absolute/relative path to the dataset."
    )
    parser.add_argument("--ckp_dir", type=str, help="Path for checkpoints.")

    # model arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model name.",
    )
    parser.add_argument(
        "--input_seq_length",
        type=int,
        required=False,
        help="Input position sequence length.",
    )
    parser.add_argument(
        "--num_mp_steps",
        type=int,
        required=False,
        help="Number of message passing layers.",
    )
    parser.add_argument(
        "--num_mlp_layers", type=int, required=False, help="Number of MLP layers."
    )
    parser.add_argument(
        "--latent_dim", type=int, required=False, help="Hidden layer dimension."
    )
    parser.add_argument(
        "--magnitude_features",
        action=argparse.BooleanOptionalAction,
        help="Whether to include velocity magnitudes in node features.",
    )
    parser.add_argument(
        "--isotropic_norm",
        action=argparse.BooleanOptionalAction,
        help="Use isotropic normalization.",
    )

    # output arguments
    parser.add_argument(
        "--out_type",
        type=str,
        required=False,
        choices=["vtk", "pkl", "none"],
        help="Output type to store rollouts during validation.",
    )
    parser.add_argument(
        "--out_type_infer",
        type=str,
        required=False,
        choices=["vtk", "pkl", "none"],
        help="Output type to store rollouts during inference.",
    )
    parser.add_argument(
        "--rollout_dir", type=str, required=False, help="Directory to write rollouts."
    )

    # segnn-specific arguments
    parser.add_argument(
        "--lmax_attributes",
        type=int,
        required=False,
        help="Maximum degree of attributes.",
    )
    parser.add_argument(
        "--lmax_hidden",
        type=int,
        required=False,
        help="Maximum degree of hidden layers.",
    )
    parser.add_argument(
        "--segnn_norm",
        type=str,
        required=False,
        choices=["instance", "batch", "none"],
        help="Normalisation type.",
    )
    parser.add_argument(
        "--velocity_aggregate",
        type=str,
        required=False,
        choices=["avg", "sum", "last", "all"],
        help="Velocity aggregation function for node attributes.",
    )
    parser.add_argument(
        "--attribute_mode",
        type=str,
        required=False,
        choices=["add", "concat", "velocity"],
        help="How to combine node attributes.",
    )
    # HAE-specific arguments
    parser.add_argument(
        "--right_attribute",
        required=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use last velocity to steer the attribute embedding.",
    )
    parser.add_argument(
        "--attribute_embedding_blocks",
        required=False,
        type=int,
        help="Number of embedding layers for the attributes.",
    )

    # misc arguments
    parser.add_argument(
        "--gpu", type=int, required=False, help="CUDA device ID to use."
    )
    parser.add_argument(
        "--f64",
        required=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use double precision.",
    )

    parser.add_argument(
        "--eval_n_trajs",
        required=False,
        type=int,
        help="Number of trajectories to evaluate during validation.",
    )
    parser.add_argument(
        "--eval_n_trajs_infer",
        required=False,
        type=int,
        help="Number of trajectories to evaluate during inference.",
    )

    parser.add_argument(
        "--metrics",
        required=False,
        nargs="+",
        help="Validation metrics to evaluate. Choose from: mse, mae, sinkhorn, e_kin.",
    )
    parser.add_argument(
        "--metrics_infer",
        required=False,
        nargs="+",
        help="Inference metrics to evaluate during inference.",
    )
    parser.add_argument(
        "--metrics_stride",
        required=False,
        type=int,
        help="Stride for Sinkhorn and e_kin during validation",
    )
    parser.add_argument(
        "--metrics_stride_infer",
        required=False,
        type=int,
        help="Stride for Sinkhorn and e_kin during inference.",
    )
    parser.add_argument(
        "--n_rollout_steps",
        required=False,
        type=int,
        help="Number of rollout steps during validation/testing.",
    )
    # only keep passed arguments to avoid overwriting config
    return {k: v for k, v in vars(parser.parse_args()).items() if v is not None}


class NestedLoader(yaml.SafeLoader):
    """Load yaml files with nested configs."""

    def get_single_data(self):
        parent = {}
        config = super().get_single_data()
        if "extends" in config and (included := config["extends"]):
            del config["extends"]
            with open(os.path.join("configs", included), "r") as f:
                parent = yaml.load(f, NestedLoader)
        return {**parent, **config}
