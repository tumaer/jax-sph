"""Riemann-SPH implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel

EPS = jnp.finfo(float).eps


def SPHRIEMANN(
    displacement_fn,
    eos,
    dx,
    dim,
    Vmax,
    eta_limiter=3,
    is_limiter=False,
):
    """Acceleration according to Riemann-SPH

    Based on: "A weakly compressible SPH method based on a
    low-dissipation Riemann solver", Zhang, Hu, Adams, 2017
    """

    # SPH kernel function
    # h = 1.3dx in paper
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
        #w_dist = vmap(kernel_fn.w)(dist)

        # norm because we don't have the directions e_s
        e_ij = -dr_i_j / (dist[:, None] + EPS)

        ##### Compute primitives

        # pressure
        p = vmap(eos.p_fn)(rho)
        #background_pressure_tvf = vmap(eos.p_fn)(jnp.zeros_like(p))

        # artificial speed of sound, below eq. (2)
        c0 = 10 * Vmax

        # if enabled, introduce dissipation limiter eq. (11)  jnp.full_like(temp, c0)  10 * jnp.absolute(U_av)
        if is_limiter:
            def beta_fn(u_L, u_R, eta_limiter):
                #u_L = jnp.absolute(u_L)
                #u_R = jnp.absolute(u_R)
                temp = eta_limiter * jnp.maximum(u_L - u_R, jnp.zeros_like(u_L))
                beta = jnp.minimum(temp, jnp.full_like(temp, c0))
                return beta
        else:
            def beta_fn(u_L, u_R, eta_limiter):
                return c0



   
        ##### Compute RHS

        def conservation_fn(
            d_ij,
            rho_L,
            rho_R,
            v_i,
            v_j,
            m_j,
            p_L,
            p_R,
            e_ij,
        ):
            
            # [:, None]

            # state velocities eq. (6)
            u_L = jnp.dot(v_i, e_ij)
            u_R = jnp.dot(v_j, e_ij)
            #print(jnp.shape(U_L))

            # average quantities
            U_av = (u_L + u_R) / 2
            rho_av = (rho_L + rho_R) / 2
            p_av = (p_L + p_R) / 2
            v_av = (v_i + v_j) / 2

            # linear Riemann solver eq. (7)
            U_star = U_av + 0.5 * (p_L - p_R) / (c0 * rho_av + EPS)
            p_star = p_av + 0.5 * beta_fn(u_L, u_R, eta_limiter) * rho_av * (u_L - u_R)
            v_star = U_star * e_ij + (v_av - U_av * e_ij)

            # compute the common prefactor | minus for eij
            _kernel_grad = kernel_fn.grad_w(d_ij) * e_ij
            _c = 2 * m_j / rho_R 
            # mass conservation eq. (8)
            eq_8 = rho_L * _c * jnp.dot((v_i - v_star), _kernel_grad)

            # momentum conservation eq. (9)
            eq_9 = (-1) * _c * p_star / rho_L * _kernel_grad

            return eq_8, eq_9

        out = vmap(conservation_fn)(
            dist,
            rho[i_s],
            rho[j_s],
            v[i_s],
            v[j_s],
            mass[j_s],
            p[i_s],
            p[j_s],
            e_ij,
        )
        
        drhodt = ops.segment_sum(out[0], i_s, N)
        dvdt = ops.segment_sum(out[1], i_s, N)

        state = {
            "r": r,
            "tag": tag,
            "u": v,
            "v": v,
            "drhodt": drhodt,
            "dudt": dvdt,
            "dvdt": dvdt,
            "rho": rho,
            "p": p,
            "mass": mass,
            "eta": eta,
        }

        return state

    return forward
