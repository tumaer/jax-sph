"""Transport velocity SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel, WendlandC2Kernel
from jax_sph.utils import Tag

EPS = jnp.finfo(float).eps


def rho_evol_fn(rho, mass, u, grad_w_dist, i_s, j_s, dt, N, **kwargs):
    """Density evolution according to Adami et al. 2013."""
    v_j_s = (mass / rho)[j_s]
    temp = v_j_s * ((u[i_s] - u[j_s]) * grad_w_dist).sum(axis=1)
    drhodt = rho * ops.segment_sum(temp, i_s, N)
    rho = rho + dt * drhodt
    return rho, drhodt


def rho_evol_riemann_fn_wrapper(kernel_fn, eos, c0):
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
        **kwargs,
    ):
        """Density evolution according to Zhang et al. 2017."""

        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij = e_s

        # Compute kernel gradient
        kernel_grad = kernel_fn.grad_w(d_ij) * (e_ij)

        # Compute average states eq. (6)/(12)/(13), Zhang (2017)
        u_L = jnp.where(wall_mask_j == 1, jnp.dot(u_i, -n_w_j), jnp.dot(u_i, -e_ij))
        p_L = p_i
        rho_L = rho_i

        #  u_w from eq. (15), Yang (2020)
        u_R = jnp.where(
            wall_mask_j == 1,
            -u_L + 2 * jnp.dot(u_j, n_w_j),
            jnp.dot(u_j, -e_ij),
        )
        p_R = jnp.where(wall_mask_j == 1, p_L + rho_L * jnp.dot(g_ext_i, -r_ij), p_j)
        rho_R = jnp.where(wall_mask_j == 1, eos.rho_fn(p_R), rho_j)

        U_avg = (u_L + u_R) / 2
        v_avg = (u_i + u_j) / 2
        rho_avg = (rho_L + rho_R) / 2

        # Compute Riemann states eq. (7) and below eq. (9), Zhang (2017)
        U_star = U_avg + 0.5 * (p_L - p_R) / (rho_avg * c0)
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
    ):
        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij = e_s

        # Compute kernel gradient
        kernel_part_diff = kernel_fn.grad_w(d_ij)
        kernel_grad = kernel_part_diff * (e_ij)

        # Compute average states eq. (6)/(12)/(13), Zhang (2017)
        u_L = jnp.where(wall_mask_j == 1, jnp.dot(u_i, -n_w_j), jnp.dot(u_i, -e_ij))
        p_L = p_i
        rho_L = rho_i

        #  u_w from eq. (15), Yang (2020)
        u_R = jnp.where(
            wall_mask_j == 1,
            -u_L + 2 * jnp.dot(u_j, n_w_j),
            jnp.dot(u_j, -e_ij),
        )
        p_R = jnp.where(wall_mask_j == 1, p_L + rho_L * jnp.dot(g_ext_i, -r_ij), p_j)
        rho_R = jnp.where(wall_mask_j == 1, eos.rho_fn(p_R), rho_j)

        P_avg = (p_L + p_R) / 2
        rho_avg = (rho_L + rho_R) / 2

        # Compute inter-particle-averaged shear viscosity (harmonic mean)
        # eq. (6), Adami (2013)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)

        # Compute Riemann states eq. (7) and (10), Zhang (2017)
        # u_R = jnp.where(
        # wall_mask_j == 1, -u_L - 2 * jnp.dot(v_j, -n_w_j), jnp.dot(v_j, -e_ij)
        # )
        P_star = P_avg + 0.5 * rho_avg * (u_L - u_R) * beta_fn(u_L, u_R, eta_limiter)

        # pressure term with linear Riemann solver eq. (9), Zhang (2017)
        eq_9 = -2 * m_j * (P_star / (rho_i * rho_j)) * kernel_grad

        # viscosity term eq. (6), Zhang (2019)
        v_ij = u_i - u_j
        eq_6 = 2 * m_j * eta_ij / (rho_i * rho_j) * v_ij / (d_ij + EPS)
        eq_6 *= kernel_part_diff * mask

        # compute the prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        _A = (tvf_stress_fn(rho_i, u_i, u_i) + tvf_stress_fn(rho_j, u_j, u_j)) / 2
        a_eq_8 = _c * jnp.dot(_A, r_ij)

        return eq_9 + eq_6 + a_eq_8

    return acceleration_fn_riemann


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

        mask_fluid = tag == Tag.FLUID
        mask_fluid_edges = mask_fluid[j_s] * mask_fluid[i_s]
        res = mask_fluid_edges[:, None] * numerator / denominator
        dudt_artif = ops.segment_sum(res, i_s, N)
        return dudt_artif

    return artificial_viscosity_fn


def gwbc_fn_wrapper(is_free_slip, is_heat_conduction, eos):
    def gwbc_fn(temperature, rho, tag, u, v, p, g_ext, i_s, j_s, w_dist, dr_i_j, N):
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

        mask_bc = jnp.isin(tag, jnp.array(Tag.WALL))

        def no_slip_bc_fn(x):
            # for boundary particles, sum over fluid velocities
            x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

            # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
            x_wall = x_wall_unnorm / (w_i_sum_wf[:, None] + EPS)
            # eq. 23 from same paper

            x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
            return x

        def free_slip_bc_fn(x):
            # # normal vectors pointing from fluid to wall
            # (1) implement via summing over fluid particles
            wall_inner = ops.segment_sum(dr_i_j * mask_j_s_fluid[:, None], i_s, N)
            # (2) implement using color gradient. Requires 2*rc thick wall
            # wall_inner = - ops.segment_sum(dr_i_j*mask_j_s_wall[:, None], i_s, N)

            normalization = jnp.sqrt((wall_inner**2).sum(axis=1, keepdims=True))
            wall_inner_normals = wall_inner / (normalization + EPS)
            wall_inner_normals = jnp.where(mask_bc[:, None], wall_inner_normals, 0.0)

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
    if is_free_slip:

        def free_weight(fluid_mask_i, tag_i):
            return fluid_mask_i
    else:

        def free_weight(fluid_mask_i, tag_i):
            return jnp.ones_like(tag_i)

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

    return free_weight, heat_bc


def limiter_fn_wrapper(eta_limiter, c0):
    """if != -1, introduce dissipation limiter eq. (11), Zhang (2017)"""

    if eta_limiter == -1:

        def beta_fn(u_L, u_R, eta_limiter):
            return c0
    else:

        def beta_fn(u_L, u_R, eta_limiter):
            temp = eta_limiter * jnp.maximum(u_L - u_R, jnp.zeros_like(u_L))
            beta = jnp.minimum(temp, jnp.full_like(temp, c0))
            return beta

    return beta_fn


def temperature_derivative_wrapper(kernel_fn):
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


def WCSPH(
    displacement_fn,
    eos,
    g_ext_fn,
    dx,
    dim,
    dt,
    c0,
    eta_limiter=3,
    solver="SPH",
    kernel="QSK",
    is_bc_trick=False,
    is_rho_evol=False,
    artificial_alpha=0.0,
    is_free_slip=False,
    is_rho_renorm=False,
    is_heat_conduction=False,
):
    """Weakly compressible SPH solver with transport velocity formulation."""

    _beta_fn = limiter_fn_wrapper(eta_limiter, c0)
    if kernel == "QSK":
        _kernel_fn = QuinticKernel(h=dx, dim=dim)
    elif kernel == "W2CK":
        _kernel_fn = WendlandC2Kernel(h=1.3 * dx, dim=dim)

    _gwbc_fn = gwbc_fn_wrapper(is_free_slip, is_heat_conduction, eos)
    _free_weight, _heat_bc = gwbc_fn_riemann_wrapper(is_free_slip, is_heat_conduction)
    _acceleration_tvf_fn = acceleration_tvf_fn_wrapper(_kernel_fn)
    _acceleration_riemann_fn = acceleration_riemann_fn_wrapper(
        _kernel_fn, eos, _beta_fn, eta_limiter
    )
    _acceleration_fn = acceleration_standard_fn_wrapper(_kernel_fn)
    _artificial_viscosity_fn = artificial_viscosity_fn_wrapper(dx, artificial_alpha)
    _wall_phi_vec = wall_phi_vec_wrapper(_kernel_fn)
    _rho_evol_riemann_fn = rho_evol_riemann_fn_wrapper(_kernel_fn, eos, c0)
    _temperature_derivative = temperature_derivative_wrapper(_kernel_fn)

    def forward(state, neighbors):
        """Update step of SPH solver

        Args:
            state (dict): Flow fields and particle properties.
            neighbors (_type_): Neighbors object.
        """

        r, tag, mass, eta = state["r"], state["tag"], state["mass"], state["eta"]
        u, v, dudt, dvdt = state["u"], state["v"], state["dudt"], state["dvdt"]
        rho, drhodt, p = state["rho"], state["drhodt"], state["p"]
        kappa, Cp = state["kappa"], state["Cp"]
        temperature, dTdt = state["T"], state["dTdt"]
        N = len(r)

        # precompute displacements `dr` and distances `dist`
        # the second vector is sorted
        i_s, j_s = neighbors.idx
        r_i_s, r_j_s = r[i_s], r[j_s]
        dr_i_j = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr_i_j)
        w_dist = vmap(_kernel_fn.w)(dist)

        # TODO: related to density evolution. Optimize implementation
        # norm because we don't have the directions e_s
        e_s = dr_i_j / (dist[:, None] + EPS)
        grad_w_dist_norm = vmap(_kernel_fn.grad_w)(dist)
        grad_w_dist = grad_w_dist_norm[:, None] * e_s

        # external acceleration field
        g_ext = g_ext_fn(r)  # e.g. np.array([[0, -1], [0, -1], ...])

        # masks
        wall_mask = jnp.where(jnp.isin(tag, jnp.array(Tag.WALL)), 1.0, 0.0)
        fluid_mask = jnp.where(tag == Tag.FLUID, 1.0, 0.0)

        # calculate normal vector of wall boundaries
        temp = vmap(_wall_phi_vec)(
            rho[j_s], mass[j_s], dr_i_j, dist, wall_mask[j_s], wall_mask[i_s]
        )
        phi = ops.segment_sum(temp, i_s, N)

        # compute normal vector for boundary particles eq. (15), Zhang (2017)
        n_w = (
            phi
            / (jnp.linalg.norm(phi, ord=2, axis=1) + EPS)[:, None]
            * wall_mask[:, None]
        )
        n_w = jnp.where(jnp.absolute(n_w) < EPS, 0.0, n_w)

        ##### Density summation or evolution

        # update evolution

        if is_rho_evol and (solver == "SPH"):
            rho, drhodt = rho_evol_fn(rho, mass, u, grad_w_dist, i_s, j_s, dt, N)

            if is_rho_renorm:
                rho = rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N)
        elif is_rho_evol and (solver == "RIE"):
            temp = vmap(_rho_evol_riemann_fn)(
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
                n_w[j_s],
                g_ext[i_s],
            )
            drhodt = ops.segment_sum(temp, i_s, N) * fluid_mask
            rho = rho + dt * drhodt

            if is_rho_renorm:
                rho = rho_renorm_fn(rho, mass, i_s, j_s, w_dist, N)
        else:
            rho = rho_summation_fn(mass, i_s, w_dist, N)

        ##### Compute primitives

        # pressure, and background pressure
        p = vmap(eos.p_fn)(rho)
        background_pressure_tvf = vmap(eos.p_fn)(jnp.zeros_like(p))

        #####  Apply BC trick

        if is_bc_trick and (solver == "SPH"):
            p, rho, u, v, temperature = _gwbc_fn(
                temperature, rho, tag, u, v, p, g_ext, i_s, j_s, w_dist, dr_i_j, N
            )
        elif is_bc_trick and (solver == "RIE"):
            mask = _free_weight(fluid_mask[i_s], tag[i_s])
            temperature = _heat_bc(
                fluid_mask[j_s], w_dist, temperature, i_s, j_s, tag, N
            )
        elif (not is_bc_trick) and (solver == "RIE"):
            mask = jnp.ones_like(tag[i_s])

        ##### compute heat conduction

        if is_heat_conduction:
            # integrate the incomming temperature derivative
            temperature += dt * dTdt

            # compute temperature derivative for next step
            out = vmap(_temperature_derivative)(
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

        if solver == "SPH":
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
        elif solver == "RIE":
            out = vmap(_acceleration_riemann_fn)(
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
                n_w[j_s],
                g_ext[i_s],
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
            "dTdt": dTdt,
            "T": temperature,
            "kappa": kappa,
            "Cp": Cp,
        }

        return state

    return forward
