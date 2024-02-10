"""Transport velocity SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel

EPS = jnp.finfo(float).eps


def rho_evol_fn(rho, mass, u, grad_w_dist, i_s, j_s, dt, N, **kwargs):
    """Density evolution according to Adami et al. 2013."""
    v_j_s = (mass / rho)[j_s]
    temp = v_j_s * ((u[i_s] - u[j_s]) * grad_w_dist).sum(axis=1)
    drhodt = rho * ops.segment_sum(temp, i_s, N)
    rho = rho + dt * drhodt
    return rho, drhodt


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


def acceleration_tvf_fn_wrapper(kernel_fn):
    def acceleration_tvf_fn(
        r_ij,
        d_ij,
        rho_i,
        rho_j,
        m_i,
        m_j,
        p_bg_i,
    ):
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


def artificial_viscosity_fn_wrapper(dx, artificial_alpha, u_ref=1.0):
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

        water_mask = jnp.where((tag[j_s] == 0) * (tag[i_s] == 0), 1.0, 0.0)
        res = water_mask[:, None] * numerator / denominator
        dudt_artif = ops.segment_sum(res, i_s, N)
        return dudt_artif

    return artificial_viscosity_fn


def gwbc_fn_wrapper(is_free_slip, eos):
    def gwbc_fn(rho, tag, u, v, p, g_ext, i_s, j_s, w_dist, dr_i_j, N):
        """Enforce wall BC by treating boundary particles in special way

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

        def no_slip_bc_fn(x):
            # for boundary particles, sum over fluid velocities
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

            # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            # eq. 23 from same paper
            x = jnp.where(tag[:, None] > 0, 2 * x - x_wall, x)
            return x

        def free_slip_bc_fn(x):
            # # normal vectors pointing from fluid to wall
            # (1) implement via summing over fluid particles
            wall_inner = ops.segment_sum(dr_i_j * mask_j_s_fluid[:, None], i_s, N)
            # (2) implement using color gradient. Requires 2*rc thick wall
            # wall_inner = - ops.segment_sum(dr_i_j*mask_j_s_wall[:, None], i_s, N)

            normalization = jnp.sqrt((wall_inner**2).sum(axis=1, keepdims=True))
            wall_inner_normals = wall_inner / (normalization + EPS)
            wall_inner_normals = jnp.where(tag[:, None] > 0, wall_inner_normals, 0.0)

            # for boundary particles, sum over fluid velocities
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

            # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            x_wall = wall_inner_normals * (x_wall * wall_inner_normals).sum(
                axis=1, keepdims=True
            )

            # eq. 23 from same paper
            x = jnp.where(tag[:, None] > 0, 2 * x - x_wall, x)
            return x

        # require operations with sender fluid and receiver wall/lid
        mask_j_s_fluid = jnp.where(tag[j_s] == 0, 1.0, 0.0)
        # mask_j_s_wall = jnp.where(tag[j_s] > 0, 1.0, 0.0)
        w_j_s_fluid = w_dist * mask_j_s_fluid
        # sheparding denominator
        w_i_sum_wf = ops.segment_sum(w_j_s_fluid, i_s, N)

        if is_free_slip:  # TODO: implement reversal of normal velocity!
            # free-slip boundary - ignore viscous interactions with wall
            u = free_slip_bc_fn(u)
            v = free_slip_bc_fn(v)
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
        p = jnp.where(tag > 0, p_wall, p)

        rho = vmap(eos.rho_fn)(p)
        return p, rho, u, v

    return gwbc_fn


def WCSPH(
    displacement_fn,
    eos,
    g_ext_fn,
    dx,
    dim,
    dt,
    is_bc_trick=False,
    is_rho_evol=False,
    artificial_alpha=0.0,
    is_free_slip=False,
    is_rho_renorm=False,
):
    """Weakly compressible SPH solver with transport velocity formulation."""

    kernel_fn = QuinticKernel(h=dx, dim=dim)
    _gwbc_fn = gwbc_fn_wrapper(is_free_slip, eos)
    _acceleration_tvf_fn = acceleration_tvf_fn_wrapper(kernel_fn)
    _acceleration_fn = acceleration_standard_fn_wrapper(kernel_fn)
    _artificial_viscosity_fn = artificial_viscosity_fn_wrapper(dx, artificial_alpha)

    def forward(state, neighbors):
        """Update step of SPH solver

        Args:
            state (dict): Flow fields and particle properties.
            neighbors (_type_): Neighbors object.
        """

        r, tag, mass = state["r"], state["tag"], state["mass"]
        u, v, dudt, dvdt = state["u"], state["v"], state["dudt"], state["dvdt"]
        rho, eta, p = state["rho"], state["eta"], state["p"]
        drhodt = state["drhodt"]
        N = len(r)

        # precompute displacements `dr` and distances `dist`
        # the second vector is sorted
        i_s, j_s = neighbors.idx
        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(kernel_fn.w)(dist)

        # TODO: related to density evolution. Optimize implementation
        # norm because we don't have the directions e_s
        e_s = dr_i_j / (dist[:, None] + EPS)
        grad_w_dist_norm = vmap(kernel_fn.grad_w)(dist)
        grad_w_dist = grad_w_dist_norm[:, None] * e_s

        # external acceleration field
        g_ext = g_ext_fn(r)  # e.g. np.array([[0, -1], [0, -1], ...])

        ##### Density summation or evolution

        # update evolution

        if is_rho_evol:
            rho, drhodt = rho_evol_fn(rho, mass, u, grad_w_dist, i_s, j_s, dt, N)

            if is_rho_renorm:
                rho = rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N)
        else:
            rho = rho_summation_fn(mass, i_s, w_dist, N)

        ##### Compute primitives

        # pressure, and background pressure
        p = vmap(eos.p_fn)(rho)
        background_pressure_tvf = vmap(eos.p_fn)(jnp.zeros_like(p))

        #####  Apply BC trick

        if is_bc_trick:  # TODO: put everything in a dedicated function for this
            p, rho, u, v = _gwbc_fn(
                rho, tag, u, v, p, g_ext, i_s, j_s, w_dist, dr_i_j, N
            )
        ##### Compute RHS

        out = vmap(_acceleration_fn)(
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
        dudt = ops.segment_sum(out, i_s, N)

        out_tv = vmap(_acceleration_tvf_fn)(
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

        if artificial_alpha != 0.0:
            dudt += _artificial_viscosity_fn(
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
        }

        return state

    return forward
