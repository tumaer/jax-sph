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
        base_viscosity: float,
        p_bg_factor: float,
        dx: float,
        ext_force_fn,
        shift_fn,
        displacement_fn,
        neighbors_update_fn,
        normalization_stats,
    ):
        super().__init__()

        self.model = GNS(dim, latent_size, blocks_per_step, num_mp_steps, 16, 9)

        # TODO
        dt = dt / num_sitl_steps
        self.num_sitl_steps = num_sitl_steps
        self.dt = dt
        self.dx = dx
        self.dim = dim
        self.base_viscosity = base_viscosity
        self.normalization_stats = normalization_stats

        if ext_force_fn is None:
            ext_force_fn = lambda r: jnp.zeros_like(r)

        rho_ref = 1.0

        u_ref = 1.0 if not hasattr(self, "u_ref") else self.u_ref
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
        self.displacement_fn_vmap = vmap(displacement_fn)
        self.kernel_fn = kernels.QuinticKernel(dx, dim=dim)

        self.neighbors_update_fn = neighbors_update_fn

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

    def si_euler(self, state, acc_correction=0.0):
        # 1. Twice 1/2dt integration of u and v
        state["u"] += 1.0 * self.dt * (state["dudt"] + acc_correction)

        # 2. Integrate position with velocity v
        state["r"] = self.shift_fn(state["r"], 1.0 * self.dt * state["u"])

        # 3. Update neighbors list
        num_particles = (state["tag"] != -1).sum()
        neighbors = self.neighbors_update_fn(state["r"], num_particles=num_particles)

        # 4. Compute accelerations
        state = self.solver(state, neighbors)

        # 5. Impose boundary conditions on dummy particles (if applicable)
        # state = bc_fn(state)

        return state

    def _transform(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, tag = sample

        # TODO why
        # num_particles = (tag != -1).sum()
        num_particles = 3200

        u_stats = self.normalization_stats["velocity"]
        u = features["vel_hist"] * u_stats["std"] + u_stats["mean"]
        u = u / self.dt

        r = features["abs_pos"][:, -1, :]

        neighbors = self.neighbors_update_fn(r, num_particles=num_particles)
        i_s, j_s = neighbors.idx

        eta = jnp.ones(num_particles) * self.base_viscosity
        mass = jnp.ones(num_particles) * self.dx**self.dim

        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = self.displacement_fn_vmap(r_i_s, r_j_s)
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
        acc_stats = self.normalization_stats["acceleration"]

        # solver step
        acc_sph = self.si_euler(state)["dudt"]
        acc_sph = acc_sph * (self.dt * self.num_sitl_steps) ** 2 / acc_stats["std"]
        # correction
        acc_gns = self.model(sample)["acc"]

        return {"acc": acc_sph + acc_gns}
