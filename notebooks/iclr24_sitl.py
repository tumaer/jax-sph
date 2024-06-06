import os
import os.path as osp
from datetime import datetime
from typing import Dict, Tuple

# specify cuda device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"

import haiku as hk
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from jax import config
from jax_md import space
from jax_md.partition import Sparse
from lagrangebench import GNS, Trainer, case_builder, infer
from lagrangebench.defaults import defaults
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.runner import setup_data, setup_model
from omegaconf import DictConfig, OmegaConf

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
        sph_prefactor: float,
        gnn_prefactor: float,
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
        self.sph_prefactor = sph_prefactor
        self.gnn_prefactor = gnn_prefactor

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
        ).forward_wrapper()

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
            "kappa": None,
            "Cp": None,
            "T": None,
            "dTdt": None,
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
            state["u"] += self.stil_dt * (
                self.sph_prefactor * acc_sph + self.gnn_prefactor * acc_gns
            )
            state["v"] = state["u"]

            # 2. Integrate position with velocity v
            state["r"] = self.shift_fn(state["r"], self.stil_dt * state["u"])

        # to effective units
        vel = self.disp_fn_vmap(state["r"], r0)
        acc = vel - u0 * self.effective_dt
        acc = self.to_effective(acc, "acceleration") / self.effective_dt**2

        return {"acc": acc}


def train_or_infer(cfg: DictConfig):
    mode = cfg.mode
    load_ckp = cfg.load_ckp
    is_test = cfg.eval.test

    if cfg.dtype == "float64":
        config.update("jax_enable_x64", True)

    data_train, data_valid, data_test = setup_data(cfg)

    metadata = data_train.metadata
    # neighbors search
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    # setup core functions
    case = case_builder(
        box=box,
        metadata=metadata,
        input_seq_length=cfg.model.input_seq_length,
        cfg_neighbors=cfg.neighbors,
        cfg_model=cfg.model,
        noise_std=cfg.train.noise_std,
        external_force_fn=data_train.external_force_fn,
        dtype=cfg.dtype,
    )

    features, particle_type = data_train[0]

    pbc = jnp.array(metadata["periodic_boundary_conditions"])

    if pbc.any():
        displacement_fn, shift_fn = space.periodic(side=jnp.array(box))
    else:
        displacement_fn, shift_fn = space.free()

    # setup model from configs
    if cfg.model.name == "sitl":
        # neighbor lists for SitL SPH and GNN part
        num_particles = metadata["num_particles_max"]
        sph_neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(box),
            r_cutoff=3.0 * metadata["dx"],
            capacity_multiplier=cfg.neighbors.multiplier,
            mask_self=False,
            format=Sparse,
            num_particles_max=num_particles,
            pbc=pbc,
        )
        gnn_neighbor_fn = partition.neighbor_list(
            displacement_fn,
            jnp.array(box),
            r_cutoff=metadata["default_connectivity_radius"],
            capacity_multiplier=cfg.neighbors.multiplier,
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
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_mp_steps=cfg.model.num_mp_steps,
                dim=metadata["dim"],
                num_sitl_steps=cfg.model.num_sitl_steps,
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
                sph_prefactor=cfg.model.sph_prefactor,
                gnn_prefactor=cfg.model.gnn_prefactor,
            )(x)

        MODEL = SolverInTheLoop
    else:
        # setup model from configs
        model, MODEL = setup_model(
            cfg,
            metadata=metadata,
            homogeneous_particles=particle_type.max() == particle_type.min(),
            has_external_force=data_train.external_force_fn is not None,
            normalization_stats=case.normalization_stats,
        )

    model = hk.without_apply_rng(hk.transform_with_state(model))

    # mixed precision training based on this reference:
    # https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    if mode == "train" or mode == "all":
        print("Start training...")

        if cfg.logging.run_name is None:
            run_prefix = f"{cfg.model.name}_{data_train.name}"
            data_and_time = datetime.today().strftime("%Y%m%d-%H%M%S")
            cfg.logging.run_name = f"{run_prefix}_{data_and_time}"

        store_ckp = os.path.join(cfg.logging.ckp_dir, cfg.logging.run_name)
        os.makedirs(store_ckp, exist_ok=True)
        os.makedirs(os.path.join(store_ckp, "best"), exist_ok=True)
        with open(os.path.join(store_ckp, "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)
        with open(os.path.join(store_ckp, "best", "config.yaml"), "w") as f:
            OmegaConf.save(config=cfg, f=f.name)

        # dictionary of configs which will be stored on W&B
        wandb_config = OmegaConf.to_container(cfg)

        trainer = Trainer(
            model,
            case,
            data_train,
            data_valid,
            cfg.train,
            cfg.eval,
            cfg.logging,
            input_seq_length=cfg.model.input_seq_length,
            seed=cfg.seed,
        )

        _, _, _ = trainer.train(
            step_max=cfg.train.step_max,
            load_ckp=load_ckp,
            store_ckp=store_ckp,
            wandb_config=wandb_config,
        )

    if mode == "infer" or mode == "all":
        print("Start inference...")

        if mode == "infer":
            model_dir = load_ckp
        if mode == "all":
            model_dir = os.path.join(store_ckp, "best")
            assert osp.isfile(os.path.join(model_dir, "params_tree.pkl"))

            cfg.eval.rollout_dir = model_dir.replace("ckp", "rollout")
            os.makedirs(cfg.eval.rollout_dir, exist_ok=True)

            if cfg.eval.infer.n_trajs is None:
                cfg.eval.infer.n_trajs = cfg.eval.train.n_trajs

        assert model_dir, "model_dir must be specified for inference."
        metrics = infer(
            model,
            case,
            data_test if is_test else data_valid,
            load_ckp=model_dir,
            cfg_eval_infer=cfg.eval.infer,
            rollout_dir=cfg.eval.rollout_dir,
            n_rollout_steps=cfg.eval.n_rollout_steps,
            seed=cfg.seed,
        )

        split = "test" if is_test else "valid"
        print(f"Metrics of {model_dir} on {split} split:")
        print(averaged_metrics(metrics))


def load_configs(cli_args: DictConfig, config_path: str = None) -> DictConfig:
    """Loads the SitL configs and merge them with the cli overwrites."""

    cfgs = [OmegaConf.load(config_path)] if config_path is not None else []

    # SitL-specific default overwrites on top of the lagrangebench defaults
    defaults.dataset_path = "datasets/2D_RPF_3200_10kevery20"
    defaults.dtype = "float32"  # to speed up training
    defaults.model.name = "sitl"
    defaults.model.input_seq_length = 2  # one past velocity for SitL
    defaults.model.latent_dim = 64
    defaults.model.num_sitl_steps = 3
    defaults.train.step_max = 500_000
    defaults.train.noise_std = 1e-5
    defaults.train.pushforward.steps = [-1]
    defaults.train.pushforward.unrolls = [0]
    defaults.train.pushforward.probs = [1]
    defaults.eval.train.n_trajs = 10
    defaults.neighbors.multiplier = 2.5
    defaults.model.sph_prefactor = 1.0
    defaults.model.gnn_prefactor = 1.0

    cfgs = [defaults] + cfgs

    # merge all embedded configs and give highest priority to cli_args
    cfg = OmegaConf.merge(*cfgs, cli_args)
    return cfg


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()

    if "load_ckp" in cli_args:  # start from a checkpoint
        config_path = os.path.join(cli_args.load_ckp, "config.yaml")
    else:
        config_path = None

    cfg = load_configs(cli_args, config_path)

    print("#" * 79, "\nStarting a LagrangeBench run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    train_or_infer(cfg)
