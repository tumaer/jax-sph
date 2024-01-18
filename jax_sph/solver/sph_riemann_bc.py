"""Riemann-SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel
from jax_sph.kernels import WendlandC2Kernel

EPS = jnp.finfo(float).eps

# BC und density revolution von Artur
def SPHRIEMANN_Artur(
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
    is_bc_trick=False,
    is_free_slip=False,
    is_rho_renorm=False,
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
    kernel_fn = QuinticKernel(h=dx, dim=dim)

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

        tag_inv = jnp.where(tag == 0, 1.0, 0.0)

        # external acceleration field
        g_ext = g_ext_fn(r)

        # artificial speed of sound, below eq. (2), Zhang (2017)
        c0 = 10 * Vmax


        e_s = dr_i_j / (dist[:, None] + EPS)
        grad_w_dist_norm = vmap(kernel_fn.grad_w)(dist)
        grad_w_dist = grad_w_dist_norm[:, None] * e_s
        
        if is_rho_evol:
            # TODO: should this be in the RHS computation?
            v_j_s = (mass / rho)[j_s]
            temp = v_j_s * ((v[i_s] - v[j_s]) * grad_w_dist).sum(axis=1)
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


        '''
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
        
        '''


        # pressure
        p = vmap(eos.p_fn)(rho)    


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

            #eta_j_s_ = eta[j_s]
            if is_free_slip:  # TODO: implement reversal of normal velocity!
                # free-slip boundary condition - ignore viscous interactions with wall
                # eta_j_s_ = eta_j_s_ * jnp.where(tag[j_s] == 1, 0., 1.)
                #u = free_slip_bc_fn(u)
                v = free_slip_bc_fn(v)
            else:
                # no-slip boundary condition
                #u = no_slip_bc_fn(u)
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
            #u_, v_ = u, v

            #p, rho = p_, rho_
        



        
        

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
            tag_inv_j,
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
            tag_inv[j_s],
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
