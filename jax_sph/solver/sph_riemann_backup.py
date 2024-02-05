"""Riemann-SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel
from jax_sph.kernels import WendlandC2Kernel

EPS = jnp.finfo(float).eps


def SPHRIEMANN(
    displacement_fn,
    eos,
    g_ext_fn,
    dx,
    dim,
    dt,
    Vmax,
    eta_limiter=3,
    is_limiter=False,
    is_rho_evol=False,
):
    """Conservation laws according to Riemann-SPH 

    Based on: "A weakly compressible SPH method based on a
    low-dissipation Riemann solver", Zhang, Hu, Adams, 2017

    and: "Dual-criteria time stepping for weakly compressible smoothed
    particle hydrodynamics", Zhang, Rezavand, Hu, 2019

    and: "A transport-velocity formulation for smoothed particle
    hydrodynamics", Adami, Hu, Adams, 2013
    """

    # SPH kernel function
    kernel_fn = WendlandC2Kernel(h=1.3*dx, dim=dim)

    def forward(state, neighbors):
        """Update step of SPH solver

        Args:
            state (dict): Flow fields and particle properties.
            neighbors (_type_): Neighbors object.
        """

        r, tag, mass = state["r"], state["tag"], state["mass"]
        v, dvdt = state["v"], state["dvdt"]
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

        # external acceleration field
        g_ext = g_ext_fn(r)

        # artificial speed of sound, below eq. (2), Zhang (2017)
        c0 = 10 * Vmax

        
        if is_rho_evol:
            # Compute density gradient
            def rho_evol_fn(
                r_ij, 
                d_ij, 
                rho_i, 
                rho_j, 
                m_j, 
                v_i, 
                v_j, 
                p_i, 
                p_j
                ):


                # Compute unit vector, above eq. (6), Zhang (2017)                 
                e_ij = r_ij / (d_ij + EPS)

                # Compute kernel gradient
                kernel_grad = kernel_fn.grad_w(d_ij) * (e_ij) 

                # Compute average states eq. (6), Zhang (2017)
                u_L = jnp.dot(v_i, -e_ij)
                u_R = jnp.dot(v_j, -e_ij)
                U_avg= (u_L + u_R) / 2
                v_avg = (v_i +  v_j) / 2
                rho_avg = (rho_i + rho_j) / 2

                # Compute Riemann states eq. (7) and below eq. (9), Zhang (2017)
                U_star = U_avg + 0.5 * (p_i - p_j) / (rho_avg * c0)
                v_star = U_star * (-e_ij) + (v_avg - U_avg * (-e_ij))

                # Mass conservation with linear Riemann solver eq. (8), Zhang (2017)
                eq_8 = 2 * rho_i * m_j / rho_j * jnp.dot((v_i - v_star), kernel_grad)
                return eq_8


            temp = vmap(rho_evol_fn)(
                dr_i_j, 
                dist, 
                rho[i_s], 
                rho[j_s], 
                mass[j_s], 
                v[i_s], 
                v[j_s], 
                p[i_s], 
                p[j_s],
                )
            
            drhodt = ops.segment_sum(temp, i_s, N)
            rho = rho + dt * drhodt

        else:
            rho = mass * ops.segment_sum(w_dist, i_s, N)


        # pressure
        p = vmap(eos.p_fn)(rho)

        # if enabled, introduce dissipation limiter eq. (11), Zhang (2017) 
        if is_limiter:
            def beta_fn(u_L, u_R, eta_limiter):
                temp = eta_limiter * jnp.maximum(u_L - u_R, jnp.zeros_like(u_L))
                beta = jnp.minimum(temp, jnp.full_like(temp, c0))
                return beta
        else:
            def beta_fn(u_L, u_R, eta_limiter):
                return c0
            

        # Compute velocity gradient

        def acceleration_fn_riemann(
            r_ij,
            d_ij,
            rho_i,
            rho_j,
            m_j,
            v_i,
            v_j,
            p_i,
            p_j,
            eta_i,
            eta_j,
            ):
            
            # Compute unit vector, above eq. (6), Zhang (2017) 
            e_ij = r_ij / (d_ij + EPS)

            # Compute kernel gradient
            kernel_part_diff = kernel_fn.grad_w(d_ij)
            kernel_grad = kernel_part_diff * (e_ij)

            # Compute average states Riemann eq. (6), Zhang (2017)
            u_L = jnp.dot(v_i, -e_ij)
            u_R = jnp.dot(v_j, -e_ij)
            P_avg = (p_i + p_j) / 2
            rho_avg = (rho_i + rho_j) / 2

            # Compute inter-particle-averaged shear viscosity (harmonic mean) eq. (6), Adami (2013)
            eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
            

            # Compute Riemann states eq. (7) and (10), Zhang (2017)
            P_star = P_avg + 0.5 * rho_avg * (u_L - u_R) * beta_fn(u_L, u_R, eta_limiter)

            # pressure term with linear Riemann solver eq. (9), Zhang (2017)
            eq_9 = -2 * m_j * (P_star / (rho_i * rho_j)) * kernel_grad 
         
            # viscosity term eq. (6), Zhang (2019)
            v_ij = v_i - v_j
            eq_6 = 2 * m_j * eta_ij / (rho_i * rho_j) * v_ij / (d_ij + EPS) * kernel_part_diff
            


            return eq_9 + eq_6

        out = vmap(acceleration_fn_riemann)(
            dr_i_j, 
            dist, 
            rho[i_s], 
            rho[j_s], 
            mass[j_s], 
            v[i_s], 
            v[j_s], 
            p[i_s], 
            p[j_s],
            eta[i_s],
            eta[j_s],
            )

        dvdt = ops.segment_sum(out, i_s, N)

        state = {
            "r": r,
            "tag": tag,
            "u": v,
            "v": v,
            "drhodt": drhodt,
            "dudt": dvdt + g_ext,
            "dvdt": dvdt + g_ext,
            "rho": rho,
            "p": p,
            "mass": mass,
            "eta": eta,
        }

        return state

    return forward