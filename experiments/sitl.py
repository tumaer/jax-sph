import argparse
import os
import os.path as osp
import pprint
from datetime import datetime
from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import wandb
import yaml
from jax_md import space
from jax_md.partition import Sparse
from lagrangebench import GNS, H5Dataset, Trainer, case_builder, infer
from lagrangebench.evaluate import averaged_metrics

from jax_sph import partition
from jax_sph.eos import TaitEoS
from jax_sph.kernel import QuinticKernel
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag


class SolverInTheLoop(hk.Module):
    def __init__(
        self,
        latent_size: int,
        blocks_per_step: int,
        num_mp_steps: int,
        dim: int,
        num_sitl_steps: int,
        dt: float,
        gnn_radius: float,
        base_viscosity: float,
        p_bg_factor: float,
        dx: float,
        ext_force_fn,
        shift_fn,
        displacement_fn,
        sph_nbrs_upd_fn,
        gnn_nbrs_upd_fn,
        normalization_stats,
    ):
        super().__init__()

        self.model = GNS(dim, latent_size, blocks_per_step, num_mp_steps, 16, 9)

        self.num_sitl_steps = num_sitl_steps
        self.effective_dt = dt
        self.stil_dt = dt / num_sitl_steps
        self.dx = dx
        self.gnn_radius = gnn_radius
        self.dim = dim
        self.base_viscosity = base_viscosity
        self.normalization_stats = normalization_stats

        if ext_force_fn is None:

            def ext_force_fn(r):
                return jnp.zeros_like(r)

        rho_ref = 1.0
        c_ref = 10.0
        eta_limiter = 3
        solver = "SPH"
        kernel = "QSK"

        u_ref = 1.0
        c_ref = 10.0 * u_ref
        gamma_eos = 1.0
        p_ref = rho_ref * c_ref**2 / gamma_eos

        self.eos = TaitEoS(
            p_ref=p_ref,
            rho_ref=rho_ref,
            p_background=p_bg_factor * p_ref,
            gamma=gamma_eos,
        )
        self.solver = WCSPH(
            displacement_fn,
            self.eos,
            ext_force_fn,
            dx,
            dim,
            dt,
            c_ref,
            eta_limiter,
            solver,
            kernel,
        )

        self.shift_fn = shift_fn
        self.disp_fn_vmap = jax.vmap(displacement_fn)
        self.kernel_fn = QuinticKernel(dx, dim=dim)
        self.kernel_vmap = jax.vmap(self.kernel_fn.w)

        self.sph_nbrs_upd_fn = sph_nbrs_upd_fn
        self.gnn_nbrs_upd_fn = gnn_nbrs_upd_fn

    def acceleration_fn(
        self,
        r_ij,
        d_ij,
        rho_i,
        rho_j,
        u_i,
        u_j,
        m_i,
        m_j,
        eta_i,
        eta_j,
        p_i,
        p_j,
    ):
        # (Eq. 6) - inter-particle-averaged shear viscosity (harmonic mean)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + 1e-8)
        # (Eq. 7) - density-weighted pressure (weighted arithmetic mean)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = self.kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + 1e-8)

        _u_ij = u_i - u_j
        a_eq_8 = _c * (-p_ij * r_ij + eta_ij * _u_ij)

        return a_eq_8

    def to_physical(self, x, key="velocity"):
        stats = self.normalization_stats[key]
        x = x * stats["std"] + stats["mean"]
        x = x / self.effective_dt
        if key == "acceleration":
            x = x / self.effective_dt
        return x

    def to_effective(self, x, key="velocity"):
        stats = self.normalization_stats[key]
        x = x * self.effective_dt
        if key == "acceleration":
            x = x * self.effective_dt
        x = (x - stats["mean"]) / stats["std"]
        return x

    def _transform(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, tag = sample

        num_particles = features["abs_pos"].shape[0]

        u = self.to_physical(features["vel_hist"])

        r = features["abs_pos"][:, -1, :]

        neighbors = self.sph_nbrs_upd_fn(r, num_particles=num_particles)
        i_s, j_s = neighbors.idx

        eta = jnp.ones(num_particles) * self.base_viscosity
        mass = jnp.ones(num_particles) * self.dx**self.dim

        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = self.disp_fn_vmap(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = self.kernel_vmap(dist)

        rho = mass * jax.ops.segment_sum(w_dist, i_s, num_particles)

        p = self.eos.p_fn(rho)

        dudt = jax.vmap(self.acceleration_fn)(
            dr_i_j,
            dist,
            rho[i_s],
            rho[j_s],
            u[i_s],
            u[j_s],
            mass[i_s],
            mass[j_s],
            eta[i_s],
            eta[j_s],
            p[i_s],
            p[j_s],
        )
        dudt = jax.ops.segment_sum(dudt, i_s, num_particles)

        state = {
            "r": r,
            "tag": tag,
            "u": u,
            "v": u,
            "dudt": dudt,
            "dvdt": dudt,
            "drhodt": None,
            "rho": rho,
            "p": p,
            "mass": mass,
            "eta": eta,
        }

        return state

    def __call__(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        state = self._transform(sample)
        features, tag = sample
        r0 = state["r"].copy()
        u0 = state["u"].copy()
        N = (state["tag"] != Tag.PAD_VALUE).sum()

        for _ in range(self.num_sitl_steps):
            # solver step and neighbors list
            sph_neighbors = self.sph_nbrs_upd_fn(state["r"], num_particles=N)
            acc_sph = self.solver(state, sph_neighbors)["dudt"]

            # correction
            gnn_neighbors = self.gnn_nbrs_upd_fn(state["r"], num_particles=N)
            features["receivers"], features["senders"] = gnn_neighbors.idx
            features["vel_hist"] = self.to_effective(state["u"])
            features["rel_disp"] = (
                self.disp_fn_vmap(
                    state["r"][features["receivers"]], state["r"][features["senders"]]
                )
                / self.gnn_radius
            )
            features["rel_dist"] = space.distance(features["rel_disp"])[:, None]
            acc_gns = self.model((features, tag))["acc"]
            acc_gns = self.to_physical(acc_gns, key="acceleration")

            # integrate
            # 1. Twice 1/2dt integration of u and v
            state["u"] += self.stil_dt * (acc_sph + acc_gns)
            state["v"] = state["u"]

            # 2. Integrate position with velocity v
            state["r"] = self.shift_fn(state["r"], self.stil_dt * state["u"])

        # to effective units
        vel = self.disp_fn_vmap(state["r"], r0)
        acc = vel - u0 * self.effective_dt
        acc = self.to_effective(acc, "acceleration") / self.effective_dt**2

        return {"acc": acc}


def train_or_infer(args: argparse.Namespace):
    if not osp.isabs(args.data_dir):
        args.data_dir = osp.join(os.getcwd(), args.data_dir)

    os.makedirs("ckp", exist_ok=True)
    os.makedirs("rollouts", exist_ok=True)

    # dataloader
    data_train = H5Dataset("train", dataset_path=args.data_dir, input_seq_length=2)
    data_valid = H5Dataset(
        "valid", dataset_path=args.data_dir, input_seq_length=2, extra_seq_length=20
    )
    data_test = H5Dataset(
        "test", dataset_path=args.data_dir, input_seq_length=2, extra_seq_length=20
    )

    metadata = data_train.metadata
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    # setup core functions
    case = case_builder(
        box=box,
        metadata=metadata,
        input_seq_length=2,  # single velocity input
        noise_std=args.noise_std,
        external_force_fn=data_train.external_force_fn,
        neighbor_list_multiplier=1.5,
        dtype=jnp.float32,
    )

    features, _ = data_train[0]

    pbc = jnp.array(metadata["periodic_boundary_conditions"])

    if pbc.any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()

    # setup model from configs
    if args.model == "sitl":
        # neighbor lists for SitL SPH and GNN part
        num_particles = metadata["num_particles_max"]
        sph_neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(box),
            r_cutoff=3.0 * metadata["dx"],
            capacity_multiplier=2.5,
            mask_self=False,
            format=Sparse,
            num_particles_max=num_particles,
            pbc=pbc,
        )
        gnn_neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(box),
            r_cutoff=metadata["default_connectivity_radius"],
            capacity_multiplier=2.5,
            mask_self=False,
            format=Sparse,
            num_particles_max=num_particles,
            pbc=pbc,
        )
        sph_neighbors = sph_neighbor_fn.allocate(
            features[:, 0, :], num_particles=num_particles, extra_capacity=10
        )
        sph_neighbors_update_fn = jax.jit(sph_neighbors.update)
        gnn_neighbors = gnn_neighbor_fn.allocate(
            features[:, 0, :], num_particles=num_particles, extra_capacity=10
        )
        gnn_neighbors_update_fn = jax.jit(gnn_neighbors.update)

        def model(x):
            return SolverInTheLoop(
                latent_size=args.latent_dim,
                blocks_per_step=args.num_mlp_layers,
                num_mp_steps=args.num_mp_steps,
                dim=metadata["dim"],
                num_sitl_steps=args.num_sitl_steps,
                dt=metadata["dt"] * metadata["write_every"],
                p_bg_factor=0.0,
                base_viscosity=metadata["viscosity"],
                dx=metadata["dx"],
                gnn_radius=metadata["default_connectivity_radius"],
                shift_fn=shift_fn,
                ext_force_fn=data_train.external_force_fn,
                displacement_fn=displacement_fn,
                sph_nbrs_upd_fn=sph_neighbors_update_fn,
                gnn_nbrs_upd_fn=gnn_neighbors_update_fn,
                normalization_stats=case.normalization_stats,
            )(x)

    elif args.model == "gns":

        def model(x):
            return GNS(
                metadata["dim"],
                latent_size=args.latent_dim,
                blocks_per_step=args.num_mlp_layers,
                num_mp_steps=args.num_mp_steps,
                particle_type_embedding_size=16,
            )(x)

    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(SolverInTheLoop, policy)

    if args.mode == "train" or args.mode == "all":
        print("Start training...")
        # save config file
        run_prefix = f"{args.model}_{data_train.name}"
        data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
        run_name = f"{run_prefix}_{data_and_time}"

        args.new_checkpoint = os.path.join("ckp/", run_name)
        os.makedirs(args.new_checkpoint, exist_ok=True)
        os.makedirs(os.path.join(args.new_checkpoint, "best"), exist_ok=True)
        with open(os.path.join(args.new_checkpoint, "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)
        with open(os.path.join(args.new_checkpoint, "best", "config.yaml"), "w") as f:
            yaml.dump(vars(args), f)

        if args.wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
                save_code=True,
            )
        else:
            wandb_run = None

        trainer = Trainer(
            model,
            case,
            data_train,
            data_valid,
            seed=args.seed,
            batch_size=args.batch_size,
            input_seq_length=2,  # single velocity input
            noise_std=args.noise_std,
            lr_start=args.lr,
            eval_n_trajs=10,
        )

        _, _, _ = trainer(
            step_max=args.step_max,
            load_checkpoint=args.model_dir,
            store_checkpoint=args.new_checkpoint,
            wandb_run=wandb_run,
        )

        if args.wandb:
            wandb.finish()

    if args.mode == "infer" or args.mode == "all":
        print("Start inference...")
        if args.mode == "all":
            args.model_dir = os.path.join(args.new_checkpoint, "best")
            assert osp.isfile(os.path.join(args.model_dir, "params_tree.pkl"))

        assert args.model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if args.test else data_valid,
            load_checkpoint=args.model_dir,
            metrics=["mse", "ekin", "sinkhorn"],
            rollout_dir=osp.join("rollouts", run_name),
            eval_n_trajs=data_valid.num_samples,
            out_type="vtk",
            n_extrap_steps=args.n_extrap_steps,
            seed=args.seed,
        )

        split = "test" if args.test else "valid"
        print(f"Metrics of {args.model_dir} on {split} split:")
        print(averaged_metrics(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LagrangeBench")
    parser.add_argument("--model_dir", type=str, help="Path to the model directory")

    # training
    parser.add_argument(
        "--data_dir",
        type=str,
        # NOTE: the simpler dataset "2D_RPF_3200_20kevery20" is not in lagrangebench
        default="datasets/2D_RPF_3200_20kevery100",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--mode", type=str, default="train", help="train, infer, or all"
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to use the test set"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--step_max", type=int, default=10000, help="Max number of steps"
    )
    parser.add_argument("--noise_std", type=float, default=1e-4, help="Noise std")
    parser.add_argument(
        "--n_extrap_steps",
        type=int,
        default=0,
        help="Number of inference extrapolation steps",
    )

    # model
    parser.add_argument("--model", type=str, default="sitl", help="Model name")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument(
        "--num_mp_steps", type=int, default=10, help="Number of message passing steps"
    )
    parser.add_argument(
        "--num_mlp_layers", type=int, default=2, help="Number of MLP layers"
    )
    parser.add_argument(
        "--num_sitl_steps", type=int, default=3, help="Number of SitL steps"
    )

    # wandb
    parser.add_argument(
        "--wandb", action="store_true", help="Whether to use wandb for logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, help="Name of the wandb project to log to"
    )
    parser.add_argument(
        "--wandb_entity", type=str, help="Name of the wandb entity to log to"
    )
    args = parser.parse_args()

    if args.model_dir is not None:  # to run inference
        config_path = os.path.join(args.model_dir, "config.yaml")
        with open(config_path, "r") as f:
            loaded_args = yaml.safe_load(f)
        loaded_args.update(vars(args))
        args = argparse.Namespace(**loaded_args)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    pprint.pprint(vars(args))
    print("#" * 79)

    train_or_infer(args)
