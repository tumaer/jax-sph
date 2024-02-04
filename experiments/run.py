import copy
import os
import os.path as osp
import pprint
from argparse import Namespace
from datetime import datetime

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
import yaml
from jax_md import space
from jax_md.partition import Sparse
from lagrangebench import Trainer, infer
from lagrangebench.case_setup import case_builder
from lagrangebench.data.utils import get_dataset_stats
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.utils import PushforwardConfig

from experiments.config import NestedLoader, cli_arguments
from experiments.sitl import SolverInTheLoop
from experiments.utils import setup_data
from jax_sph import partition


def train_or_infer(args: Namespace):
    data_train, data_valid, data_test, args = setup_data(args)

    # neighbors search
    bounds = np.array(data_train.metadata["bounds"])
    args.box = bounds[:, 1] - bounds[:, 0]

    args.info.len_train = len(data_train)
    args.info.len_eval = len(data_valid)

    # setup core functions
    case = case_builder(
        box=args.box,
        metadata=data_train.metadata,
        input_seq_length=args.config.input_seq_length,
        isotropic_norm=args.config.isotropic_norm,
        noise_std=args.config.noise_std,
        magnitude_features=args.config.magnitude_features,
        external_force_fn=data_train.external_force_fn,
        neighbor_list_backend=args.config.neighbor_list_backend,
        neighbor_list_multiplier=args.config.neighbor_list_multiplier,
        dtype=(jnp.float64 if args.config.f64 else jnp.float32),
    )

    features, particle_type = data_train[0]

    args.info.homogeneous_particles = particle_type.max() == particle_type.min()
    args.metadata = data_train.metadata
    args.normalization_stats = case.normalization_stats
    args.config.has_external_force = data_train.external_force_fn is not None

    pbc = jnp.array(args.metadata["periodic_boundary_conditions"])

    if pbc.any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(args.box))
    else:
        displacement_fn, shift_fn = space.free()

    # TODO pretty ugly

    num_particles = args.metadata["num_particles_max"]
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        jnp.array(args.box),
        r_cutoff=3.0 * args.metadata["dx"],
        capacity_multiplier=args.config.neighbor_list_multiplier,
        mask_self=False,
        format=Sparse,
        num_particles_max=num_particles,
        pbc=pbc,
    )
    neighbors = neighbor_fn.allocate(
        features[:, 0, :], num_particles=num_particles, extra_capacity=10
    )
    neighbors_update_fn = jax.jit(neighbors.update)

    # setup model from configs
    def model(x):
        return SolverInTheLoop(
            latent_size=args.config.latent_dim,
            blocks_per_step=args.config.num_mlp_layers,
            num_mp_steps=args.config.num_mp_steps,
            dim=args.metadata["dim"],
            num_sitl_steps=args.config.num_sitl_steps,
            dt=args.metadata["dt"] * args.metadata["write_every"],
            p_bg_factor=0.0,
            base_viscosity=args.metadata["viscosity"],
            dx=args.metadata["dx"],
            shift_fn=shift_fn,
            ext_force_fn=data_train.external_force_fn,
            displacement_fn=displacement_fn,
            neighbors_update_fn=neighbors_update_fn,
            normalization_stats=get_dataset_stats(
                args.metadata, args.config.isotropic_norm, args.config.noise_std
            ),
        )(x)

    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(SolverInTheLoop, policy)

    if args.config.mode == "train" or args.config.mode == "all":
        print("Start training...")
        # save config file
        run_prefix = f"{args.config.model}_{data_train.name}"
        data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
        args.info.run_name = f"{run_prefix}_{data_and_time}"

        args.config.new_checkpoint = os.path.join(
            args.config.ckp_dir, args.info.run_name
        )
        os.makedirs(args.config.new_checkpoint, exist_ok=True)
        os.makedirs(os.path.join(args.config.new_checkpoint, "best"), exist_ok=True)
        with open(os.path.join(args.config.new_checkpoint, "config.yaml"), "w") as f:
            yaml.dump(vars(args.config), f)
        with open(
            os.path.join(args.config.new_checkpoint, "best", "config.yaml"), "w"
        ) as f:
            yaml.dump(vars(args.config), f)

        if args.config.wandb:
            # wandb doesn't like Namespace objects
            args_dict = copy.copy(args)
            args_dict.config = vars(args.config)
            args_dict.info = vars(args.info)

            wandb_run = wandb.init(
                project=args.config.wandb_project,
                entity=args.config.wandb_entity,
                name=args.info.run_name,
                config=args_dict,
                save_code=True,
            )
        else:
            wandb_run = None

        pf_config = PushforwardConfig(
            steps=args.config.pushforward["steps"],
            unrolls=args.config.pushforward["unrolls"],
            probs=args.config.pushforward["probs"],
        )

        trainer = Trainer(
            model,
            case,
            data_train,
            data_valid,
            pushforward=pf_config,
            metrics=args.config.metrics,
            seed=args.config.seed,
            batch_size=args.config.batch_size,
            input_seq_length=args.config.input_seq_length,
            noise_std=args.config.noise_std,
            lr_start=args.config.lr_start,
            lr_final=args.config.lr_final,
            lr_decay_steps=args.config.lr_decay_steps,
            lr_decay_rate=args.config.lr_decay_rate,
            loss_weight=args.config.loss_weight,
            n_rollout_steps=args.config.n_rollout_steps,
            eval_n_trajs=args.config.eval_n_trajs,
            rollout_dir=args.config.rollout_dir,
            out_type=args.config.out_type,
            log_steps=args.config.log_steps,
            eval_steps=args.config.eval_steps,
            metrics_stride=args.config.metrics_stride,
            num_workers=args.config.num_workers,
            batch_size_infer=args.config.batch_size_infer,
        )
        _, _, _ = trainer(
            step_max=args.config.step_max,
            load_checkpoint=args.config.model_dir,
            store_checkpoint=args.config.new_checkpoint,
            wandb_run=wandb_run,
        )

        if args.config.wandb:
            wandb.finish()

    if args.config.mode == "infer" or args.config.mode == "all":
        print("Start inference...")
        if args.config.mode == "all":
            args.config.model_dir = os.path.join(args.config.new_checkpoint, "best")
            assert osp.isfile(os.path.join(args.config.model_dir, "params_tree.pkl"))

            args.config.rollout_dir = args.config.model_dir.replace("ckp", "rollout")
            os.makedirs(args.config.rollout_dir, exist_ok=True)

            if args.config.eval_n_trajs_infer is None:
                args.config.eval_n_trajs_infer = args.config.eval_n_trajs

        assert args.config.model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if args.config.test else data_valid,
            load_checkpoint=args.config.model_dir,
            metrics=args.config.metrics_infer,
            rollout_dir=args.config.rollout_dir,
            eval_n_trajs=args.config.eval_n_trajs_infer,
            n_rollout_steps=args.config.n_rollout_steps,
            out_type=args.config.out_type_infer,
            n_extrap_steps=args.config.n_extrap_steps,
            seed=args.config.seed,
            metrics_stride=args.config.metrics_stride_infer,
            batch_size=args.config.batch_size_infer,
        )

        split = "test" if args.config.test else "valid"
        print(f"Metrics of {args.config.model_dir} on {split} split:")
        print(averaged_metrics(metrics))


if __name__ == "__main__":
    cli_args = cli_arguments()
    if "config" in cli_args:  # to (re)start training
        config_path = cli_args["config"]
    elif "model_dir" in cli_args:  # to run inference
        config_path = os.path.join(cli_args["model_dir"], "config.yaml")

    with open(config_path, "r") as f:
        args = yaml.load(f, NestedLoader)

    # priority to command line arguments
    args.update(cli_args)
    args = Namespace(config=Namespace(**args), info=Namespace())
    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    pprint.pprint(vars(args.config))
    print("#" * 79)

    # specify cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.config.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.config.xla_mem_fraction)

    if args.config.f64:
        from jax import config

        config.update("jax_enable_x64", True)

    train_or_infer(args)
