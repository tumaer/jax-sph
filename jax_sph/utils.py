"""General jax-sph utils"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from jax_md import partition, space

from jax_sph.io_state import read_h5
from jax_sph.kernels import QuinticKernel

EPS = jnp.finfo(float).eps


def pos_init_cartesian_2d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r


def pos_init_cartesian_3d(box_size, dx):
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r


def noise_masked(r, mask, key, std):
    noise = std * jax.random.normal(key, r.shape)
    masked_noise = jnp.where(mask[:, None], noise, 0.0)
    r += masked_noise
    return r


def get_ekin(state, dx):
    v = state["v"]
    v_water = jnp.where(state["tag"][:, None] == 0, v, 0)
    ekin = jnp.square(v_water).sum().item()
    return 0.5 * ekin * dx ** v.shape[1]


def get_val_max(state, var="u"):
    "Extract the largest velocity magnitude, needed for TGV"
    if jnp.size(state[var].shape) > 1:
        max = jnp.sqrt(jnp.square(state[var]).sum(axis=1)).max()
    else:
        max = jnp.absolute(state[var]).max()
    return max


def sph_interpolator(args, src_path, prop_type="vector"):
    """Interpolate SPH data to a regular grid / line.

    Args:
        args (_type_): Simulation arguments.
        src_path (str): used only for instantiating the neighbors object
        prop (str, optional): Which velocity to use. Defaults to 'u'.
        dim_ind (int, optional): Which component of velocity. Defaults to 0.
    """
    state = read_h5(src_path)
    N = len(state["r"])
    dim = args.dim

    # invert velocity for boundary particles
    def comp_bc_interm(x, i_s, j_s, w_j_s_fluid, w_i_sum):
        # for boundary particles, sum over fluid velocities
        x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

        # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
        x_wall = x_wall_unnorm / (w_i_sum[:, None] + EPS)
        # eq. 23 from same paper
        x = jnp.where(state["tag"][:, None] > 0, 2 * x - x_wall, x)
        return x
    
    # Set the wall particle tempertature or pressure the same as the neighbouring
    # fluid particles, so that the neighboring fluid particles get the full suport.
    def comp_bc_interm_scalar(x, i_s, j_s, w_j_s_fluid, w_i_sum):
        # for boundary particles, sum over fluid velocities
        x_wall_unnorm = ops.segment_sum(w_j_s_fluid * x[j_s], i_s, N)

        # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
        x_wall = x_wall_unnorm / (w_i_sum + EPS)
        # eq. 23 from same paper
        x = jnp.where(state["tag"] > 0, x_wall, x)
        return x 

    kernel_fn = QuinticKernel(h=args.dx, dim=args.dim)

    if prop_type == "vector" or prop_type == "scalar":
        box_size = np.array(args.bounds)[:, 1]
        if np.array(args.periodic_boundary_conditions).sum() > 0:
            displacement_fn, shift_fn = space.periodic(side=box_size)
        else:
            displacement_fn, shift_fn = space.free()

        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box_size,
            r_cutoff=3 * args.dx,
            dr_threshold=3 * args.dx * 0.25,
            capacity_multiplier=1.25,
            mask_self=False,
            format=partition.Sparse,
        )
        neighbors = neighbor_fn.allocate(
            state["r"],
        )

    def interp_vel(src_path, r_target, prop="u", dim_ind=0):
        #### SPH interpolate from "set_src" onto "set_dst"
        state = read_h5(src_path)
        # compute kernel avarages at y_axis positions in the center, x=0.2
        vel = state[prop]

        i_s, j_s = neighbors.idx
        r_i_s, r_j_s = state["r"][i_s], state["r"][j_s]
        dr = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr)
        w_dist = vmap(kernel_fn.w)(dist)

        # require operations with sender fluid and receiver wall/lid
        w_j_s_fluid = w_dist * jnp.where(state["tag"][j_s] == 0, 1.0, 0.0)
        # sheparding denominator
        w_i_sum = ops.segment_sum(w_j_s_fluid, i_s, N)

        # invert directions of velocities of wall particles
        vel = comp_bc_interm(vel, i_s, j_s, w_j_s_fluid, w_i_sum)

        # discrete points
        dist = (((r_target[:, None] - state["r"][None, :]) ** 2).sum(axis=-1)) ** 0.5
        w_dist = kernel_fn.w(dist)
        # weight normalization for non-full support
        w_norm = w_dist.sum(axis=-1) * args.dx**dim

        u_val = (w_dist * vel[:, dim_ind][None, :]).sum(axis=1)
        u_val *= args.dx**dim
        u_val /= w_norm

        return u_val

    def interp_scalar(src_path, r_target, prop="p"):
        #### SPH interpolate from "set_src" onto "set_dst"
        state = read_h5(src_path)
        # compute kernel avarages at y_axis positions in the center, x=0.2
        p = state[prop]

        # Note: Currently no inversion of pressure for boundary particles
        i_s, j_s = neighbors.idx
        r_i_s, r_j_s = state["r"][i_s], state["r"][j_s]
        dr = vmap(displacement_fn)(r_i_s, r_j_s)
        dist = space.distance(dr)
        w_dist = vmap(kernel_fn.w)(dist)

        # require operations with sender fluid and receiver wall/lid
        w_j_s_fluid = w_dist * jnp.where(state["tag"][j_s] == 0, 1.0, 0.0)
        # sheparding denominator
        w_i_sum = ops.segment_sum(w_j_s_fluid, i_s, N)

        p = comp_bc_interm_scalar(p, i_s, j_s, w_j_s_fluid, w_i_sum)

        # discrete points
        dist = (((r_target[:, None] - state["r"][None, :]) ** 2).sum(axis=-1)) ** 0.5
        w_dist = kernel_fn.w(dist)
        # weight normalization for non-full support
        w_norm = w_dist.sum(axis=-1) * args.dx**dim

        p_val = (w_dist * p).sum(axis=1)
        p_val *= args.dx**dim
        p_val /= w_norm

        return p_val

    if prop_type == "vector":
        return interp_vel
    elif prop_type == "scalar":
        return interp_scalar
