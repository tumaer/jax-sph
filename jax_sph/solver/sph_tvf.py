"""Transport velocity SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel

EPS = jnp.finfo(float).eps


def SPHTVF(
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
    """Acceleration according to transport velocity SPH

    Based on: "A transport-velocity formulation for smoothed particle
    hydrodynamics", Adami, Hu, Adams, 2013
    """

    # SPH kernel function
    kernel_fn = QuinticKernel(h=dx, dim=dim)

    def forward(state, neighbors):
        """Update step of SPH solver

        Args:
            state (dict): Flow fields and particle properties.
            neighbors (_type_): Neighbors object.
        """

        def stress(rho: float, u, v):
            """Transport stress tensor. See 'A' just under (Eq. 4)"""
            return jnp.outer(rho * u, v - u)

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
            # TODO: should this be in the RHS computation?
            v_j_s = (mass / rho)[j_s]
            temp = v_j_s * ((u[i_s] - u[j_s]) * grad_w_dist).sum(axis=1)
            drhodt = rho * ops.segment_sum(temp, i_s, N)
            rho = rho + dt * drhodt

            if is_rho_renorm:
                # # correction term, see "A generalized transport-velocity
                # # formulation for SPH" by Zhang et al. 2017
                # TODO: check renormalization with different kernel cutoff
                nominator = ops.segment_sum(mass[j_s] * w_dist, i_s, N)
                rho_denominator = ops.segment_sum((mass / rho)[j_s] * w_dist, i_s, N)
                rho_denominator = jnp.where(rho_denominator > 1, 1, rho_denominator)
                rho = nominator / rho_denominator
        else:
            rho = mass * ops.segment_sum(w_dist, i_s, N)

        ##### Compute primitives

        # pressure, and background pressure
        p = vmap(eos.p_fn)(rho)
        background_pressure_tvf = vmap(eos.p_fn)(jnp.zeros_like(p))

        #####  Apply BC trick

        if is_bc_trick:  # TODO: put everything in a dedicated function for this
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
                wall_inner_normals = jnp.where(
                    tag[:, None] > 0, wall_inner_normals, 0.0
                )

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

            eta_j_s_ = eta[j_s]
            if is_free_slip:  # TODO: implement reversal of normal velocity!
                # free-slip boundary condition - ignore viscous interactions with wall
                # eta_j_s_ = eta_j_s_ * jnp.where(tag[j_s] == 1, 0., 1.)
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
            p_ = jnp.where(tag > 0, p_wall, p)

            rho_ = vmap(eos.rho_fn)(p_)
            u_, v_ = u, v

            p, rho = p_, rho_
        else:
            u_, v_, p_, eta_j_s_ = u, v, p, eta[j_s]
            rho_ = vmap(eos.rho_fn)(p)

        ##### Compute RHS

        def acceleration_fn(
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
            # (Eq. 6) - inter-particle-averaged shear viscosity (harmonic mean)
            eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
            # (Eq. 7) - density-weighted pressure (weighted arithmetic mean)
            p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

            # compute the common prefactor `_c`
            _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
            _kernel_grad = kernel_fn.grad_w(d_ij)
            _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

            # (Eq. 8): \boldsymbol{e}_{ij} is computed as r_ij/d_ij here.
            _A = (stress(rho_i, u_i, v_i) + stress(rho_j, u_j, v_j)) / 2
            _u_ij = u_i - u_j
            a_eq_8 = _c * (-p_ij * r_ij + jnp.dot(_A, r_ij) + eta_ij * _u_ij)

            # (Eq. 13) - or at least the acceleration term
            a_eq_13 = _c * 1.0 * p_bg_i * r_ij

            return a_eq_8, a_eq_13

        out = vmap(acceleration_fn)(
            dr_i_j,
            dist,
            rho_[i_s],
            rho_[j_s],
            u_[i_s],
            u_[j_s],
            v_[i_s],
            v_[j_s],
            mass[i_s],
            mass[j_s],
            eta[i_s],
            eta_j_s_,
            p_[i_s],
            p_[j_s],
            background_pressure_tvf[i_s],
        )
        dudt = ops.segment_sum(out[0], i_s, N)
        dvdt = ops.segment_sum(out[1], i_s, N)

        ##### Additional things

        # set pressure at wall to 0.0; better for ParaView
        # p = jnp.where(tag > 0, 0.0, p)

        if artificial_alpha != 0.0:
            # if only artificial viscosity is used, then the following applies
            # nu = alpha * h * c_ab / 2 / (dim+2)
            #    = 0.1 * 0.02 * 10*1 /2/4= 0.0025
            # TODO: parse reference parameters from case setup
            h_ab = dx
            u_ref = 1.0  # this works fine for 2D dam break, but should have been 2.0
            c_ref = 10.0 * u_ref
            c_ab = c_ref
            rho_ab = (rho_[i_s] + rho_[j_s]) / 2
            numerator = mass[j_s] * artificial_alpha * h_ab * c_ab
            numerator = numerator * ((u[i_s] - u[j_s]) * dr_i_j).sum(axis=1)
            numerator = numerator[:, None] * grad_w_dist
            denominator = (rho_ab * (dist**2 + 0.01 * h_ab**2))[:, None]

            water_mask = jnp.where((tag[j_s] == 0) * (tag[i_s] == 0), 1.0, 0.0)
            res = water_mask[:, None] * numerator / denominator
            dudt_artif = ops.segment_sum(res, i_s, N)
        else:
            dudt_artif = jnp.zeros_like(dudt)

        state = {
            "r": r,
            "tag": tag,
            "u": u,
            "v": v,
            "drhodt": drhodt,
            "dudt": dudt + g_ext + dudt_artif,
            "dvdt": dvdt,
            "rho": rho,
            "p": p,
            "mass": mass,
            "eta": eta,
        }

        return state

    return forward
