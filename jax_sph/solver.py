"""Weakly compressible SPH solver."""

from typing import Callable, Union

import jax.numpy as jnp
from jax import ops, vmap

from jax_sph.eos import RIEMANNEoS, TaitEoS
from jax_sph.jax_md import space
from jax_sph.kernel import (
    CubicKernel,
    GaussianKernel,
    QuinticKernel,
    SuperGaussianKernel,
    WendlandC2Kernel,
    WendlandC4Kernel,
    WendlandC6Kernel,
)
from jax_sph.utils import Tag, wall_tags

EPS = jnp.finfo(float).eps


def rho_evol_fn(rho, mass, u, grad_w_dist, i_s, j_s, dt, N, **kwargs):
    """Density evolution according to Adami et al. 2013."""
    v_j_s = (mass / rho)[j_s]
    temp = v_j_s * ((u[i_s] - u[j_s]) * grad_w_dist).sum(axis=1)
    drhodt = rho * ops.segment_sum(temp, i_s, N)
    rho = rho + dt * drhodt
    return rho, drhodt


def rho_evol_riemann_fn_wrapper(kernel_fn, eos, c_ref):
    """Density evolution according to Zhang et al. 2017."""

    def rho_evol_riemann_fn(
        e_s,
        rho_i,
        rho_j,
        m_j,
        u_i,
        u_j,
        p_i,
        p_j,
        r_ij,
        d_ij,
        wall_mask_j,
        n_w_j,
        g_ext_i,
        u_tilde_j,
        **kwargs,
    ):
        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij = e_s

        # Compute kernel gradient
        kernel_grad = kernel_fn.grad_w(d_ij) * (e_ij)

        # Compute average states eq. (6)/(12)/(13), Zhang (2017)
        u_L = jnp.where(
            jnp.isin(wall_mask_j, wall_tags), jnp.dot(u_i, -n_w_j), jnp.dot(u_i, -e_ij)
        )
        p_L = p_i
        rho_L = rho_i

        #  u_w from eq. (15), Yang (2020)
        u_R = jnp.where(
            jnp.isin(wall_mask_j, wall_tags),
            -u_L + 2 * jnp.dot(u_j, n_w_j),
            jnp.dot(u_j, -e_ij),
        )
        p_R = jnp.where(
            jnp.isin(wall_mask_j, wall_tags), p_L + rho_L * jnp.dot(g_ext_i, -r_ij), p_j
        )
        rho_R = jnp.where(jnp.isin(wall_mask_j, wall_tags), eos.rho_fn(p_R), rho_j)

        U_avg = (u_L + u_R) / 2
        v_avg = (u_i + u_j) / 2
        rho_avg = (rho_L + rho_R) / 2

        # Compute Riemann states eq. (7) and below eq. (9), Zhang (2017)
        U_star = U_avg + 0.5 * (p_L - p_R) / (rho_avg * c_ref)
        v_star = U_star * (-e_ij) + (v_avg - U_avg * (-e_ij))

        # Mass conservation with linear Riemann solver eq. (8), Zhang (2017)
        eq_8 = 2 * rho_i * m_j / rho_j * jnp.dot((u_i - v_star), kernel_grad)
        return eq_8

    return rho_evol_riemann_fn


def rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N):
    """Renormalization of density according to Zhang et al. 2017."""
    nominator = ops.segment_sum(mass[j_s] * w_dist, i_s, N)
    rho_denominator = ops.segment_sum((mass / rho)[j_s] * w_dist, i_s, N)
    rho_denominator = jnp.where(rho_denominator > 1, 1, rho_denominator)
    rho = nominator / rho_denominator
    return rho


def rho_summation_fn(mass, i_s, w_dist, N):
    """Density summation."""
    return mass * ops.segment_sum(w_dist, i_s, N)


def wall_phi_vec_wrapper(kernel_fn):
    """Compute the wall phi vector according to Zhang et al. 2017."""

    def wall_phi_vec(rho_j, m_j, dr_ij, dist, tag_j, tag_i):
        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij_w = dr_ij / (dist + EPS)

        # Compute kernel gradient
        kernel_grad = kernel_fn.grad_w(dist) * (e_ij_w)

        # compute phi eq. (15), Zhang (2017)
        phi = -1.0 * m_j / rho_j * kernel_grad * tag_j * tag_i

        return phi

    return wall_phi_vec


def acceleration_tvf_fn_wrapper(kernel_fn):
    """Transport velocity formulation acceleration according to Adami et al. 2013."""

    def acceleration_tvf_fn(r_ij, d_ij, rho_i, rho_j, m_i, m_j, p_bg_i):
        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        # (Eq. 13) - or at least the acceleration term
        a_eq_13 = _c * 1.0 * p_bg_i * r_ij

        return a_eq_13

    return acceleration_tvf_fn


def tvf_stress_fn(rho: float, u, v):
    """Transport velocity stress tensor. See 'A' under (Eq. 4) in Adami et al. 2013."""
    return jnp.outer(rho * u, v - u)


def acceleration_standard_fn_wrapper(kernel_fn):
    """Standard SPH acceleration according to Adami et al. 2012."""

    def acceleration_standard_fn(
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
    ):
        # (Eq. 6) - inter-particle-averaged shear viscosity (harmonic mean)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
        # (Eq. 7) - density-weighted pressure (weighted arithmetic mean)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        # (Eq. 8): \boldsymbol{e}_{ij} is computed as r_ij/d_ij here.
        _A = (tvf_stress_fn(rho_i, u_i, v_i) + tvf_stress_fn(rho_j, u_j, v_j)) / 2
        _u_ij = u_i - u_j
        a_eq_8 = _c * (-p_ij * r_ij + jnp.dot(_A, r_ij) + eta_ij * _u_ij)
        return a_eq_8

    return acceleration_standard_fn


def acceleration_riemann_fn_wrapper(kernel_fn, eos, beta_fn, eta_limiter):
    """Riemann solver acceleration according to Zhang et al. 2017."""

    def acceleration_fn_riemann(
        e_s,
        r_ij,
        d_ij,
        rho_i,
        rho_j,
        m_j,
        m_i,
        u_i,
        u_j,
        p_i,
        p_j,
        eta_i,
        eta_j,
        wall_mask_j,
        mask,
        n_w_j,
        g_ext_i,
        u_tilde_j,
    ):
        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij = e_s

        # Compute kernel gradient
        kernel_part_diff = kernel_fn.grad_w(d_ij)
        kernel_grad = kernel_part_diff * (e_ij)

        # Compute average states eq. (6)/(12)/(13), Zhang (2017)
        u_L = jnp.where(
            jnp.isin(wall_mask_j, wall_tags), jnp.dot(u_i, -n_w_j), jnp.dot(u_i, -e_ij)
        )
        p_L = p_i
        rho_L = rho_i

        # u_w from eq. (15), Yang (2020)
        u_R = jnp.where(
            jnp.isin(wall_mask_j, wall_tags),
            -u_L + 2 * jnp.dot(u_j, n_w_j),
            jnp.dot(u_j, -e_ij),
        )
        p_R = jnp.where(
            jnp.isin(wall_mask_j, wall_tags), p_L + rho_L * jnp.dot(g_ext_i, -r_ij), p_j
        )
        rho_R = jnp.where(jnp.isin(wall_mask_j, wall_tags), eos.rho_fn(p_R), rho_j)

        P_avg = (p_L + p_R) / 2
        rho_avg = (rho_L + rho_R) / 2

        # Compute inter-particle-averaged shear viscosity (harmonic mean)
        # eq. (6), Adami (2013)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)

        # Compute Riemann states eq. (7) and (10), Zhang (2017)
        P_star = P_avg + 0.5 * rho_avg * (u_L - u_R) * beta_fn(u_L, u_R, eta_limiter)

        # pressure term with linear Riemann solver eq. (9), Zhang (2017)
        eq_9 = -2 * m_j * (P_star / (rho_i * rho_j)) * kernel_grad

        # viscosity term eq. (6), Zhang (2019)
        u_d = 2 * u_j - u_tilde_j
        v_ij = jnp.where(
            jnp.isin(wall_mask_j, wall_tags),
            u_i - u_d,
            u_i - u_j,
        )
        eq_6 = 2 * m_j * eta_ij / (rho_i * rho_j) * v_ij / (d_ij + EPS)
        eq_6 *= kernel_part_diff * mask

        # compute the prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        _A = jnp.where(
            jnp.isin(wall_mask_j, wall_tags),
            (tvf_stress_fn(rho_i, u_i, u_i) + tvf_stress_fn(rho_j, u_d, u_d)) / 2,
            (tvf_stress_fn(rho_i, u_i, u_i) + tvf_stress_fn(rho_j, u_j, u_j)) / 2,
        )
        a_eq_8 = _c * jnp.dot(_A, r_ij)

        return eq_9 + eq_6 + a_eq_8

    return acceleration_fn_riemann


def artificial_viscosity_fn_wrapper(dx, artificial_alpha, u_ref=1.0):
    """Artificial viscosity according to Adami et al. 2012."""
    h_ab = dx
    # if only artificial viscosity is used, then the following applies
    # nu = alpha * h * c_ab / 2 / (dim+2)
    #    = 0.1 * 0.02 * 10*1 /2/4= 0.0025
    # TODO: parse reference parameters from case setup
    c_ab = 10.0 * u_ref  # c_ref

    def artificial_viscosity_fn(
        rho, mass, u, tag, i_s, j_s, dr_i_j, dist, grad_w_dist, N
    ):
        rho_ab = (rho[i_s] + rho[j_s]) / 2
        numerator = mass[j_s] * artificial_alpha * h_ab * c_ab
        numerator = numerator * ((u[i_s] - u[j_s]) * dr_i_j).sum(axis=1)
        numerator = numerator[:, None] * grad_w_dist
        denominator = (rho_ab * (dist**2 + 0.01 * h_ab**2))[:, None]

        mask_fluid = tag == Tag.FLUID
        mask_fluid_edges = mask_fluid[j_s] * mask_fluid[i_s]
        res = mask_fluid_edges[:, None] * numerator / denominator
        dudt_artif = ops.segment_sum(res, i_s, N)
        return dudt_artif

    return artificial_viscosity_fn


def gwbc_fn_wrapper(is_free_slip, is_heat_conduction, eos):
    """Enforce wall boundary conditions by treating boundary particles in a special way.

    If solid walls -> apply BC tricks

    Update dummy particles before acceleration computation (if appl.).

    Steps for boundary particles:

    - sum pressure over fluid with sheparding
    - inverse EoS for density
    - sum velocity over fluid with sheparding and * (-1)
    - if free-slip: project velocity onto normal vector
    - subtract that from 2 * u_wall - keeps lid intact

    Based on: "A generalized wall boundary condition for smoothed
    particle hydrodynamics", Adami, Hu, Adams, 2012
    """

    def gwbc_fn(temperature, rho, tag, u, v, p, g_ext, i_s, j_s, w_dist, dr_i_j, nw, N):
        mask_bc = jnp.isin(tag, wall_tags)

        def no_slip_bc_fn(x):
            # for boundary particles, sum over fluid velocities
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

            # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            # eq. 23 from same paper

            x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x

        def free_slip_bc_fn(x, wall_inner_normals):
            # # normal vectors pointing from fluid to wall
            # (1) implement via summing over fluid particles
            # wall_inner = ops.segment_sum(dr_i_j * mask_j_s_fluid[:, None], i_s, N)
            # # (2) implement using color gradient. Requires 2*rc thick wall
            # # wall_inner = - ops.segment_sum(dr_i_j*mask_j_s_wall[:, None], i_s, N)

            # normalization = jnp.sqrt((wall_inner**2).sum(axis=1, keepdims=True))
            # wall_inner_normals = wall_inner / (normalization + EPS)
            # wall_inner_normals = jnp.where(mask_bc[:, None], wall_inner_normals, 0.0)

            # for boundary particles, sum over fluid velocities
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

            # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            x_wall = wall_inner_normals * (x_wall * wall_inner_normals).sum(
                axis=1, keepdims=True
            )

            # eq. 23 from same paper
            x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x

        # require operations with sender fluid and receiver wall/lid
        mask_j_s_fluid = jnp.where(tag[j_s] == Tag.FLUID, 1.0, 0.0)
        w_j_s_fluid = w_dist * mask_j_s_fluid
        # sheparding denominator
        w_i_sum_wf = ops.segment_sum(w_j_s_fluid, i_s, N)

        if is_free_slip:
            # free-slip boundary - ignore viscous interactions with wall
            u = free_slip_bc_fn(u, -nw)
            v = free_slip_bc_fn(v, -nw)
        else:
            # no-slip boundary condition
            u = no_slip_bc_fn(u)
            v = no_slip_bc_fn(v)

        # eq. 27 from "A Generalized Wall boundary condition for SPH", 2012
        # fluid pressure term
        p_wall_unnorm = ops.segment_sum(w_j_s_fluid * p[j_s], i_s, N)

        # external fluid acceleration term
        rho_wf_sum = (rho[j_s] * w_j_s_fluid)[:, None] * dr_i_j
        rho_wf_sum = ops.segment_sum(rho_wf_sum, i_s, N)
        p_wall_ext = (g_ext * rho_wf_sum).sum(axis=1)

        # normalize with sheparding
        p_wall = (p_wall_unnorm + p_wall_ext) / (w_i_sum_wf + EPS)
        p = jnp.where(mask_bc, p_wall, p)

        rho = vmap(eos.rho_fn)(p)

        if is_heat_conduction:
            # wall particles without temperature boundary condition obtain the adjacent
            # fluid temperature
            t_wall_unnorm = ops.segment_sum(w_j_s_fluid * temperature[j_s], i_s, N)
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            mask = jnp.isin(tag, jnp.array([Tag.SOLID_WALL, Tag.MOVING_WALL]))
            t_wall = jnp.where(mask, t_wall, temperature)

            temperature = t_wall

        return p, rho, u, v, temperature

    return gwbc_fn


def gwbc_fn_riemann_wrapper(is_free_slip, is_heat_conduction):
    """Riemann solver boundary condition for wall particles."""
    if is_free_slip:

        def free_weight(fluid_mask_i, tag_i):
            return fluid_mask_i

        def riemann_velocities(u, w_dist, fluid_mask, i_s, j_s, N):
            return u
    else:

        def free_weight(fluid_mask_i, tag_i):
            return jnp.ones_like(tag_i)

        def riemann_velocities(u, w_dist, fluid_mask, i_s, j_s, N):
            w_dist_fluid = w_dist * fluid_mask[j_s]
            u_wall_nom = ops.segment_sum(w_dist_fluid[:, None] * u[j_s], i_s, N)
            u_wall_denom = ops.segment_sum(w_dist_fluid, i_s, N)
            u_tilde = u_wall_nom / (u_wall_denom[:, None] + EPS)
            return u_tilde

    if is_heat_conduction:

        def heat_bc(mask_j_s_fluid, w_dist, temperature, i_s, j_s, tag, N):
            w_j_s_fluid = w_dist * mask_j_s_fluid
            # sheparding denominator
            w_i_sum_wf = ops.segment_sum(w_j_s_fluid, i_s, N)
            t_wall_unnorm = ops.segment_sum(w_j_s_fluid * temperature[j_s], i_s, N)
            t_wall = t_wall_unnorm / (w_i_sum_wf + EPS)
            mask = jnp.isin(tag, jnp.array([Tag.SOLID_WALL, Tag.MOVING_WALL]))
            t_wall = jnp.where(mask, t_wall, temperature)
            temperature = t_wall
            return temperature
    else:

        def heat_bc(mask_j_s_fluid, w_dist, temperature, i_s, j_s, tag, N):
            return temperature

    return free_weight, riemann_velocities, heat_bc


def limiter_fn_wrapper(eta_limiter, c_ref):
    """if eta_limiter != -1, introduce dissipation limiter eq. (11), Zhang (2017)."""

    if eta_limiter == -1:

        def beta_fn(u_L, u_R, eta_limiter):
            return c_ref
    else:

        def beta_fn(u_L, u_R, eta_limiter):
            temp = eta_limiter * jnp.maximum(u_L - u_R, jnp.zeros_like(u_L))
            beta = jnp.minimum(temp, jnp.full_like(temp, c_ref))
            return beta

    return beta_fn


def temperature_derivative_wrapper(kernel_fn):
    """Temperature derivative according to Cleary 1998."""

    def temperature_derivative(
        e_s, r_ij, d_ij, rho_i, rho_j, m_j, kappa_i, kappa_j, Cp_i, T_i, T_j
    ):
        e_ij = e_s
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _kernel_grad_vector = _kernel_grad * e_ij

        _effective_kappa = (kappa_i * kappa_j) / (kappa_i + kappa_j)
        F_ab = jnp.dot(r_ij, _kernel_grad_vector) / ((d_ij * d_ij) + EPS)  # scalar

        dTdt = (4 * m_j * _effective_kappa * (T_i - T_j) * F_ab) / (
            Cp_i * rho_i * rho_j
        )

        return dTdt

    return temperature_derivative


class WCSPH:
    """Weakly compressible SPH solver with transport velocity formulation."""

    def __init__(
        self,
        displacement_fn: Callable,
        eos: Union[TaitEoS, RIEMANNEoS],
        g_ext_fn: Callable,
        dx: float,
        dim: int,
        dt: float,
        c_ref: float,
        eta_limiter: float = 3,
        solver: str = "SPH",
        kernel: str = "QSK",
        is_bc_trick: bool = False,
        is_rho_evol: bool = False,
        artificial_alpha: float = 0.0,
        is_free_slip: bool = False,
        is_rho_renorm: bool = False,
        is_heat_conduction: bool = False,
    ):
        self.displacement_fn = displacement_fn
        self.solver = solver
        self.g_ext_fn = g_ext_fn
        self.is_bc_trick = is_bc_trick
        self.is_rho_evol = is_rho_evol
        self.is_rho_renorm = is_rho_renorm
        self.dt = dt
        self.eos = eos
        self.artificial_alpha = artificial_alpha
        self.is_heat_conduction = is_heat_conduction

        _beta_fn = limiter_fn_wrapper(eta_limiter, c_ref)
        match kernel:
            case "CSK":
                self._kernel_fn = CubicKernel(h=dx, dim=dim)
            case "QSK":
                self._kernel_fn = QuinticKernel(h=dx, dim=dim)
            case "WC2K":
                self._kernel_fn = WendlandC2Kernel(h=1.3 * dx, dim=dim)
            case "WC4K":
                self._kernel_fn = WendlandC4Kernel(h=1.3 * dx, dim=dim)
            case "WC6K":
                self._kernel_fn = WendlandC6Kernel(h=1.3 * dx, dim=dim)
            case "GK":
                self._kernel_fn = GaussianKernel(h=dx, dim=dim)
            case "SGK":
                self._kernel_fn = SuperGaussianKernel(h=dx, dim=dim)

        self._gwbc_fn = gwbc_fn_wrapper(is_free_slip, is_heat_conduction, eos)
        (
            self._free_weight,
            self._riemann_velocities,
            self._heat_bc,
        ) = gwbc_fn_riemann_wrapper(is_free_slip, is_heat_conduction)
        self._acceleration_tvf_fn = acceleration_tvf_fn_wrapper(self._kernel_fn)
        self._acceleration_riemann_fn = acceleration_riemann_fn_wrapper(
            self._kernel_fn, eos, _beta_fn, eta_limiter
        )
        self._acceleration_fn = acceleration_standard_fn_wrapper(self._kernel_fn)
        self._artificial_viscosity_fn = artificial_viscosity_fn_wrapper(
            dx, artificial_alpha
        )
        self._wall_phi_vec = wall_phi_vec_wrapper(self._kernel_fn)
        self._rho_evol_riemann_fn = rho_evol_riemann_fn_wrapper(
            self._kernel_fn, eos, c_ref
        )
        self._temperature_derivative = temperature_derivative_wrapper(self._kernel_fn)

    def forward_wrapper(self):
        """Wrapper of update step of SPH."""

        def forward(state, neighbors):
            """Update step of SPH solver.

            Args:
                state (dict): Flow fields and particle properties.
                neighbors (_type_): Neighbors object.
            """

            r, tag, mass, eta = state["r"], state["tag"], state["mass"], state["eta"]
            u, v, dudt, dvdt = state["u"], state["v"], state["dudt"], state["dvdt"]
            rho, drhodt, p = state["rho"], state["drhodt"], state["p"]
            nw, kappa, Cp = state["nw"], state["kappa"], state["Cp"]
            temperature, dTdt = state["T"], state["dTdt"]
            N = len(r)

            # precompute displacements `dr` and distances `dist`
            # the second vector is sorted
            i_s, j_s = neighbors.idx
            r_i_s, r_j_s = r[i_s], r[j_s]
            dr_i_j = vmap(self.displacement_fn)(r_i_s, r_j_s)
            dist = space.distance(dr_i_j)
            w_dist = vmap(self._kernel_fn.w)(dist)

            e_s = dr_i_j / (dist[:, None] + EPS)
            # currently only for density evolution and with artificial viscosity
            grad_w_dist_norm = vmap(self._kernel_fn.grad_w)(dist)
            grad_w_dist = grad_w_dist_norm[:, None] * e_s

            # external acceleration field
            g_ext = self.g_ext_fn(r)  # e.g. np.array([[0, -1], [0, -1], ...])

            # masks
            wall_mask = jnp.where(jnp.isin(tag, wall_tags), 1.0, 0.0)
            fluid_mask = jnp.where(tag == Tag.FLUID, 1.0, 0.0)

            ##### Riemann velocity BCs
            if self.is_bc_trick and (self.solver == "RIE"):
                u_tilde = self._riemann_velocities(u, w_dist, fluid_mask, i_s, j_s, N)
            else:
                u_tilde = u

            ##### Density summation or evolution

            # update evolution

            if self.is_rho_evol and (self.solver == "SPH"):
                rho, drhodt = rho_evol_fn(
                    rho, mass, u, grad_w_dist, i_s, j_s, self.dt, N
                )

                if self.is_rho_renorm:
                    rho = rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N)
            elif self.is_rho_evol and (self.solver == "RIE"):
                temp = vmap(self._rho_evol_riemann_fn)(
                    e_s,
                    rho[i_s],
                    rho[j_s],
                    mass[j_s],
                    u[i_s],
                    u[j_s],
                    p[i_s],
                    p[j_s],
                    dr_i_j,
                    dist,
                    wall_mask[j_s],
                    nw[j_s],
                    g_ext[i_s],
                    u_tilde[j_s],
                )
                drhodt = ops.segment_sum(temp, i_s, N) * fluid_mask
                rho = rho + self.dt * drhodt

                if self.is_rho_renorm:
                    rho = rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N)
            else:
                rho_ = rho_summation_fn(mass, i_s, w_dist, N)
                rho = jnp.where(fluid_mask, rho_, rho)

            ##### Compute primitives

            # pressure, and background pressure
            p = vmap(self.eos.p_fn)(rho)
            background_pressure_tvf = vmap(self.eos.p_fn)(jnp.zeros_like(p))

            #####  Apply BC trick

            if self.is_bc_trick and (self.solver == "SPH"):
                p, rho, u, v, temperature = self._gwbc_fn(
                    temperature,
                    rho,
                    tag,
                    u,
                    v,
                    p,
                    g_ext,
                    i_s,
                    j_s,
                    w_dist,
                    dr_i_j,
                    nw,
                    N,
                )
            elif self.is_bc_trick and (self.solver == "RIE"):
                mask = self._free_weight(fluid_mask[i_s], tag[i_s])
                temperature = self._heat_bc(
                    fluid_mask[j_s], w_dist, temperature, i_s, j_s, tag, N
                )
            elif (not self.is_bc_trick) and (self.solver == "RIE"):
                mask = jnp.ones_like(tag[i_s])

            ##### compute heat conduction

            if self.is_heat_conduction:
                # integrate the incomming temperature derivative
                temperature += self.dt * dTdt

                # compute temperature derivative for next step
                out = vmap(self._temperature_derivative)(
                    e_s,
                    dr_i_j,
                    dist,
                    rho[i_s],
                    rho[j_s],
                    mass[j_s],
                    kappa[i_s],
                    kappa[j_s],
                    Cp[i_s],
                    temperature[i_s],
                    temperature[j_s],
                )
                dTdt = ops.segment_sum(out, i_s, N)

            ##### Compute RHS

            if self.solver == "SPH":
                out = vmap(self._acceleration_fn)(
                    dr_i_j,
                    dist,
                    rho[i_s],
                    rho[j_s],
                    u[i_s],
                    u[j_s],
                    v[i_s],
                    v[j_s],
                    mass[i_s],
                    mass[j_s],
                    eta[i_s],
                    eta[j_s],
                    p[i_s],
                    p[j_s],
                )
            elif self.solver == "RIE":
                out = vmap(self._acceleration_riemann_fn)(
                    e_s,
                    dr_i_j,
                    dist,
                    rho[i_s],
                    rho[j_s],
                    mass[j_s],
                    mass[i_s],
                    u[i_s],
                    u[j_s],
                    p[i_s],
                    p[j_s],
                    eta[i_s],
                    eta[j_s],
                    wall_mask[j_s],
                    mask,
                    nw[j_s],
                    g_ext[i_s],
                    u_tilde[j_s],
                )
            dudt = ops.segment_sum(out, i_s, N)

            out_tv = vmap(self._acceleration_tvf_fn)(
                dr_i_j,
                dist,
                rho[i_s],
                rho[j_s],
                mass[i_s],
                mass[j_s],
                background_pressure_tvf[i_s],
            )
            dvdt = ops.segment_sum(out_tv, i_s, N)

            ##### Additional things

            if self.artificial_alpha != 0.0:
                dudt += self._artificial_viscosity_fn(
                    rho, mass, u, tag, i_s, j_s, dr_i_j, dist, grad_w_dist, N
                )

            state = {
                "r": r,
                "tag": tag,
                "u": u,
                "v": v,
                "drhodt": drhodt,
                "dudt": dudt + g_ext,
                "dvdt": dvdt,
                "rho": rho,
                "p": p,
                "mass": mass,
                "eta": eta,
                "dTdt": dTdt,
                "T": temperature,
                "kappa": kappa,
                "Cp": Cp,
                "nw": nw,
            }

            return state

        return forward
