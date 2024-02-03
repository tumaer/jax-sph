from typing import Tuple, Dict
from enum import Enum

import jax.numpy as jnp
from jax import vmap, ops
from jax import lax
from jax_md import space
import haiku as hk

from jax_sph import eos, kernels, integrator
from jax_sph.solver.sph_tvf import SPHTVF
from lagrangebench.models import GNS


class SITLMode(Enum):
    SITL = "sitl"
    SOLVER_ONLY = "solver"
    MP_ONLY = "mp"


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
        tvf: float,
        p_bg_factor: float,
        kernel_radius: float,
        # ext_force_fn,
        shift_fn,
        displacement_fn,
        neighbors,
        mode: str = SITLMode.SITL,
    ):
        super().__init__()

        self.model = GNS(dim, latent_size, blocks_per_step, num_mp_steps, 16, 9)

        # TODO
        dt = dt / num_sitl_steps

        ext_force_fn = lambda x: jnp.zeros(x.shape)

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
        solver = SPHTVF(
            displacement_fn,
            self.eos,
            ext_force_fn,
            kernel_radius,
            dim,
            dt,
        )
        self.solver = integrator.si_euler(tvf, solver, shift_fn, lambda x: x)

        self.num_sitl_steps = num_sitl_steps
        self.dt = dt
        self.base_viscosity = base_viscosity
        self.kernel_radius = kernel_radius

        self.mode = mode
        self.shift_fn = shift_fn
        self.displacement_fn = displacement_fn
        self.kernel_fn = kernels.QuinticKernel(kernel_radius, dim=dim)

        self.neighbors = lax.stop_gradient(neighbors)

    def acceleration_fn(
        self,
        r_ij,
        d_ij,
        rho_i,
        rho_j,
        u_i,
        u_j,
        v_i,
        v_j,
        m_i,
        m_j,
        eta_i,
        eta_j,
        p_i,
        p_j,
        p_bg_i,
    ):

        def stress(rho: float, u, v):
            """Transport stress tensor. See 'A' just under (Eq. 4)"""
            return jnp.outer(rho * u, v - u)

        # (Eq. 6) - inter-particle-averaged shear viscosity (harmonic mean)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + 1e-8)
        # (Eq. 7) - density-weighted pressure (weighted arithmetic mean)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = self.kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + 1e-8)

        # (Eq. 8): \boldsymbol{e}_{ij} is computed as r_ij/d_ij here.
        _A = (stress(rho_i, u_i, v_i) + stress(rho_j, u_j, v_j)) / 2
        _u_ij = u_i - u_j
        a_eq_8 = _c * (-p_ij * r_ij + jnp.dot(_A, r_ij) + eta_ij * _u_ij)

        # (Eq. 13) - or at least the acceleration term
        a_eq_13 = _c * 1.0 * p_bg_i * r_ij

        return a_eq_8, a_eq_13

    def _transform(
        self, sample: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        features, tag = sample

        # TODO why
        # num_particles = (tag != -1).sum()
        num_particles = 2500

        v = features["vel_hist"]

        r = features["abs_pos"][..., -1]

        self.neighbors = self.neighbors.update(r, num_particles=num_particles)
        i_s, j_s = self.neighbors.idx

        eta = jnp.ones(num_particles) * self.base_viscosity
        mass = jnp.ones(num_particles)

        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(self.displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(self.kernel_fn.w)(dist)

        rho = mass * ops.segment_sum(w_dist, i_s, num_particles)

        # # TODO: related to density evolution. Optimize implementation
        # # norm because we don't have the directions e_s
        # e_s = dr_i_j / (dist[:, None] + 1e-8)
        # grad_w_dist_norm = vmap(self.kernel_fn.grad_w)(dist)
        # grad_w_dist = grad_w_dist_norm[:, None] * e_s
        # v_j_s = (mass / rho)[j_s]
        # temp = v_j_s * ((v[i_s] - v[j_s]) * grad_w_dist).sum(axis=1)
        # drhodt = rho * ops.segment_sum(temp, i_s, num_particles)
        # rho = rho + self.dt * drhodt

        # if False:
        #     # # correction term, see "A generalized transport-velocity
        #     # # formulation for SPH" by Zhang et al. 2017
        #     # TODO: check renormalization with different kernel cutoff
        #     nominator = ops.segment_sum(mass[j_s] * w_dist, i_s, N)
        #     rho_denominator = ops.segment_sum((mass / rho)[j_s] * w_dist, i_s, N)
        #     rho_denominator = jnp.where(rho_denominator > 1, 1, rho_denominator)
        #     rho = nominator / rho_denominator

        p = self.eos.p_fn(rho)

        out = vmap(self.acceleration_fn)(
            dr_i_j,
            dist,
            rho[i_s],
            rho[j_s],
            v[i_s],
            v[j_s],
            v[i_s],
            v[j_s],
            mass[i_s],
            mass[j_s],
            eta[i_s],
            eta[j_s],
            p[i_s],
            p[j_s],
            jnp.zeros_like(p[i_s]),
        )
        dudt = ops.segment_sum(out[0], i_s, num_particles)
        dvdt = ops.segment_sum(out[1], i_s, num_particles)

        state = {
            "r": r,
            "tag": tag,
            "u": v,
            "v": v,
            "dudt": dudt,
            "dvdt": dvdt,
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

        if self.mode == SITLMode.MP_ONLY:
            pos = self.model(sample)["acc"]
        else:
            state = self._transform(sample)
            # num_particles = (state["tag"] != -1).sum()

            for s in range(self.num_sitl_steps):
                if self.mode in [SITLMode.SOLVER_ONLY, SITLMode.SITL]:
                    state, self.neighbors = self.solver(self.dt, state, self.neighbors)
                    # TODO list overflow

                if self.mode == SITLMode.SITL:
                    # correction
                    features, tag = sample
                    # features["senders"] = neighbors.idx[0]
                    # features["receivers"] = neighbors.idx[1]
                    # TODO recompute edge attributes
                    features["vel_hist"] = state["v"]
                    state["r"] = self.model((features, tag))["acc"]

            pos = state["r"]

        return {"pos": pos}
