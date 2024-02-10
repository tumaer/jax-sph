from typing import Dict, Tuple

import haiku as hk
import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space
from lagrangebench.models import GNS

from jax_sph import eos, kernels
from jax_sph.solver.sph_tvf import SPHTVF


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

        # TODO
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

        u_ref = 1.0
        c_ref = 10.0 * u_ref
        gamma_eos = 1.0
        p_ref = rho_ref * c_ref**2 / gamma_eos

        self.eos = eos.TaitEoS(
            p_ref=p_ref,
            rho_ref=rho_ref,
            p_background=p_bg_factor * p_ref,
            gamma=gamma_eos,
        )
        self.solver = SPHTVF(
            displacement_fn,
            self.eos,
            ext_force_fn,
            dx,
            dim,
            dt,
        )

        self.shift_fn = shift_fn
        self.disp_fn_vmap = vmap(displacement_fn)
        self.kernel_fn = kernels.QuinticKernel(dx, dim=dim)

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
        w_dist = vmap(self.kernel_fn.w)(dist)

        rho = mass * ops.segment_sum(w_dist, i_s, num_particles)

        p = self.eos.p_fn(rho)

        dudt = vmap(self.acceleration_fn)(
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
        dudt = ops.segment_sum(dudt, i_s, num_particles)

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
        N = (state["tag"] != -1).sum()

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
