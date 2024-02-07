"""Moving Least Squares implementation"""

import jax.numpy as jnp
from jax import vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel
from jax_sph.kernels import WendlandC2Kernel
from jax_sph.kernels import M4PrimeKernel


def MLS_2nd_order_3D(kernel_arg, prop_j, dr_j_i, dim, dx):

    """3D Moving Least Squares implementation
     
    based on: "Analysis of interpolation schemes for the accurate 
    estimation of energy spectrum in Lagrangian methods", Shi, Zhu, Ellero, Adams, 2013
    """

    # define the tensors for the linear system of quations to solve
    c = jnp.empty((10, 10))
    r = jnp.empty((jnp.shape(dr_j_i)[0], 10))

    # define used Kernel
    if kernel_arg == 'QSK':
        kernel_fn = QuinticKernel(h=dx, dim=dim)
        dist = space.distance(dr_j_i)
        w_dist = vmap(kernel_fn.w)(dist)[:, jnp.newaxis]
    elif kernel_arg == 'WC2K':
        kernel_fn = WendlandC2Kernel(h=1.3*dx, dim=dim)
        dist = space.distance(dr_j_i)
        w_dist = vmap(kernel_fn.w)(dist)[:, jnp.newaxis]
    elif kernel_arg == 'M4K':
        kernel_fn = M4PrimeKernel(h=dx, dim=dim)
        w_dist = vmap(kernel_fn.w)(dr_j_i)[:, jnp.newaxis]
    
    # Rij from eq. (21), Shi 2013
    r = r.at[:, 0].set(1)
    r = r.at[:, 1:4].set(dr_j_i)
    r = r.at[:, 4:7].set(jnp.square(dr_j_i) * 0.5)
    r = r.at[:, 7].set(jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 1]))
    r = r.at[:, 8].set(jnp.multiply(dr_j_i[:, 1], dr_j_i[:, 2]))
    r = r.at[:, 9].set(jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 2]))
    r = r.at[:, :].mul(w_dist)

    # set of equations from eq. (19), Shi 2013
    c = c.at[0,:].set(jnp.sum(r, axis=0)) 
    c = c.at[1,:].set(jnp.sum(jnp.multiply(r, dr_j_i[:, 0][:, jnp.newaxis]), axis=0))
    c = c.at[2,:].set(jnp.sum(jnp.multiply(r, dr_j_i[:, 1][:, jnp.newaxis]), axis=0)) 
    c = c.at[3,:].set(jnp.sum(jnp.multiply(r, dr_j_i[:, 2][:, jnp.newaxis]), axis=0))  
    c = c.at[4,:].set(jnp.sum(jnp.multiply(r, jnp.square(dr_j_i[:, 0])[:, jnp.newaxis]), axis=0))
    c = c.at[5,:].set(jnp.sum(jnp.multiply(r, jnp.square(dr_j_i[:, 1])[:, jnp.newaxis]), axis=0)) 
    c = c.at[6,:].set(jnp.sum(jnp.multiply(r, jnp.square(dr_j_i[:, 2])[:, jnp.newaxis]), axis=0))
    c = c.at[7,:].set(jnp.sum(jnp.multiply(r, jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 1])[:, jnp.newaxis]), axis=0))
    c = c.at[8,:].set(jnp.sum(jnp.multiply(r, jnp.multiply(dr_j_i[:, 1], dr_j_i[:, 2])[:, jnp.newaxis]), axis=0))
    c = c.at[9,:].set(jnp.sum(jnp.multiply(r, jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 2])[:, jnp.newaxis]), axis=0))
    c = c.at[7:10,:].mul(+2)

    # Fj from eq. (21), Shi 2013
    f = r.at[:, :].mul(prop_j[:, jnp.newaxis])
    f = jnp.sum(f, axis=0).T

    # solve linear system of equations, below eq. (20), Shi 2013
    return jnp.linalg.solve(c, f)


def MLS_2nd_order_2D(kernel_arg, prop_j, dr_j_i, dim, dx):

    """2D Moving Least Squares implementation
     
    based on: "Analysis of interpolation schemes for the accurate 
    estimation of energy spectrum in Lagrangian methods", Shi, Zhu, Ellero, Adams, 2013
    """

    # define the tensors for the linear system of quations to solve
    c = jnp.empty((6, 6))
    r = jnp.empty((jnp.shape(dr_j_i)[0], 6))

    # define used Kernel
    if kernel_arg == 'QSK':
        kernel_fn = QuinticKernel(h=dx, dim=dim)
        dist = space.distance(dr_j_i)
        w_dist = vmap(kernel_fn.w)(dist)[:, jnp.newaxis]
    elif kernel_arg == 'WC2K':
        kernel_fn = WendlandC2Kernel(h=1.3*dx, dim=dim)
        dist = space.distance(dr_j_i)
        w_dist = vmap(kernel_fn.w)(dist)[:, jnp.newaxis]
    elif kernel_arg == 'M4K':
        kernel_fn = M4PrimeKernel(h=dx, dim=dim)
        w_dist = vmap(kernel_fn.w)(dr_j_i)[:, jnp.newaxis]
    
    # Rij from eq. (21), Shi 2013
    r = r.at[:, 0].set(1)
    r = r.at[:, 1:3].set(dr_j_i)
    r = r.at[:, 3:5].set(jnp.square(dr_j_i) * 0.5)
    r = r.at[:, 5].set(jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 1]))
    r = r.at[:, :].mul(w_dist)

    # set of equations from eq. (19), Shi 2013
    c = c.at[0,:].set(jnp.sum(r, axis=0)) 
    c = c.at[1,:].set(jnp.sum(jnp.multiply(r, dr_j_i[:, 0][:, jnp.newaxis]), axis=0))
    c = c.at[2,:].set(jnp.sum(jnp.multiply(r, dr_j_i[:, 1][:, jnp.newaxis]), axis=0)) 
    c = c.at[3,:].set(jnp.sum(jnp.multiply(r, jnp.square(dr_j_i[:, 0])[:, jnp.newaxis]), axis=0))
    c = c.at[4,:].set(jnp.sum(jnp.multiply(r, jnp.square(dr_j_i[:, 1])[:, jnp.newaxis]), axis=0)) 
    c = c.at[5,:].set(jnp.sum(jnp.multiply(r, jnp.multiply(dr_j_i[:, 0], dr_j_i[:, 1])[:, jnp.newaxis]), axis=0))
    c = c.at[5,:].mul(+2)

    # Fj from eq. (21), Shi 2013
    f = r.at[:, :].mul(prop_j[:, jnp.newaxis])
    f = jnp.sum(f, axis=0).T

    # solve linear system of equations, below eq. (20), Shi 2013
    return jnp.linalg.solve(c, f)


