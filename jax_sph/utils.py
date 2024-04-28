"""General jax-sph utils."""

import enum
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from jax_md import partition, space
from numpy import array
from omegaconf import DictConfig

from jax_sph.io_state import read_h5
from jax_sph.kernel import QuinticKernel

EPS = jnp.finfo(float).eps


class Tag(enum.IntEnum):
    """Particle types."""

    PAD_VALUE = -1  # when number of particles varies
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    DIRICHLET_WALL = 3  # for temperature boundary condition


wall_tags = jnp.array([tag.value for tag in Tag if "WALL" in tag.name])


class Phase(enum.IntEnum):
    """Differentiate between fluid phases."""

    FLUID_PHASE0 = 0
    FLUID_PHASE1 = 1
    FLUID_PHASE2 = 2
    FLUID_PHASE3 = 3


def pos_init_cartesian_2d(box_size: array, dx: float):
    """Create a grid of particles in 2D.

    Particles are at the center of the corresponding Cartesian grid cells.
    Example: if box_size=np.array([1, 1]) and dx=0.1, then the first particle will be at
    position [0.05, 0.05].
    """
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r


def pos_init_cartesian_3d(box_size: array, dx: float):
    """Create a grid of particles in 3D."""
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (jnp.vstack(list(map(jnp.ravel, grid))).T + 0.5) * dx
    return r


def pos_box_2d(L: float, H: float, dx: float, num_wall_layers: int = 3):
    """Create an empty box of particles in 2D.

    The box is of size (L + num_wall_layers * dx) x (H + num_wall_layers * dx).
    The inner part of the box starts at (num_wall_layers * dx, num_wall_layers * dx).
    """
    dx3 = num_wall_layers * dx
    # horizontal and vertical blocks
    vertical = pos_init_cartesian_2d(np.array([dx3, H + 2 * dx3]), dx)
    horiz = pos_init_cartesian_2d(np.array([L, dx3]), dx)

    # wall: left, bottom, right, top
    wall_l = vertical.copy()
    wall_b = horiz.copy() + np.array([dx3, 0.0])
    wall_r = vertical.copy() + np.array([L + dx3, 0.0])
    wall_t = horiz.copy() + np.array([dx3, H + dx3])

    res = jnp.concatenate([wall_l, wall_b, wall_r, wall_t])
    return res


def get_noise_masked(shape: tuple, mask: array, key: jax.random.PRNGKey, std: float):
    """Generate Gaussian noise with `std` where `mask` is True."""
    noise = std * jax.random.normal(key, shape)
    masked_noise = jnp.where(mask[:, None], noise, 0.0)
    return masked_noise


def get_ekin(state: Dict, dx: float):
    """Compute the kinetic energy of the fluid from `state["v"]`."""
    v = state["v"]
    v_water = jnp.where(state["tag"][:, None] == Tag.FLUID, v, 0.0)
    ekin = jnp.square(v_water).sum().item()
    return 0.5 * ekin * dx ** v.shape[1]


def get_val_max(state: Dict, var: str = "u"):
    """Extract the largest magnitude of `state["var"]`.

    For vectorial quantities, the magnitude is the Euclidean norm.
    """
    if jnp.size(state[var].shape) > 1:
        max = jnp.sqrt(jnp.square(state[var]).sum(axis=1)).max()
    else:
        max = jnp.absolute(state[var]).max()
    return max


def sph_interpolator(cfg: DictConfig, src_path: str, prop_type: str = "vector"):
    """Interpolate properties from a `state` to arbitrary coordinates, e.g. a line.

    Args:
        cfg: Simulation arguments.
        src_path: used only for instantiating the neighbors object.
        prop_type: Whether the target will be of vectorial or scalar type.

    Returns:
        Callable: Interpolation function.
    """
    state = read_h5(src_path)
    N = len(state["r"])
    dim = cfg.case.dim

    mask_bc = jnp.isin(state["tag"], wall_tags)

    # invert velocity for boundary particles
    def comp_bc_interm(x, i_s, j_s, w_j_s_fluid, w_i_sum):
        # for boundary particles, sum over fluid velocities
        x_wall_unnorm = ops.segment_sum(w_j_s_fluid[:, None] * x[j_s], i_s, N)

        # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
        x_wall = x_wall_unnorm / (w_i_sum[:, None] + EPS)
        # eq. 23 from same paper
        x = jnp.where(mask_bc[:, None], 2 * x - x_wall, x)
        return x

    # Set the wall particle tempertature or pressure the same as the neighbouring
    # fluid particles, so that the neighboring fluid particles get the full suport.
    def comp_bc_interm_scalar(x, i_s, j_s, w_j_s_fluid, w_i_sum):
        # for boundary particles, sum over fluid velocities
        x_wall_unnorm = ops.segment_sum(w_j_s_fluid * x[j_s], i_s, N)

        # eq. 22 from "A Generalized Wall boundary condition for SPH", 2012
        x_wall = x_wall_unnorm / (w_i_sum + EPS)
        # eq. 23 from same paper
        x = jnp.where(mask_bc, x_wall, x)
        return x

    kernel_fn = QuinticKernel(h=cfg.case.dx, dim=cfg.case.dim)

    if prop_type == "vector" or prop_type == "scalar":
        box_size = np.array(cfg.case.bounds)[:, 1]
        if np.array(cfg.case.pbc).sum() > 0:
            displacement_fn, shift_fn = space.periodic(side=box_size)
        else:
            displacement_fn, shift_fn = space.free()

        neighbor_fn = partition.neighbor_list(
            displacement_fn,
            box_size,
            r_cutoff=3 * cfg.case.dx,
            dr_threshold=3 * cfg.case.dx * 0.25,
            capacity_multiplier=1.25,
            mask_self=False,
            format=partition.Sparse,
        )
        neighbors = neighbor_fn.allocate(
            state["r"],
        )

    def interp_vel(src_path: str, r_target: array, prop: str = "u", dim_ind: int = 0):
        """Interpolator for vectorial quantities.

        Args:
            src_path: Path to the source state.
            r_target: Target positions.
            prop: Which quantity to use. Defaults to 'u'.
            dim_ind: Which component of velocity. Defaults to 0.
        """
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
        w_j_s_fluid = w_dist * jnp.where(state["tag"][j_s] == Tag.FLUID, 1.0, 0.0)
        # sheparding denominator
        w_i_sum = ops.segment_sum(w_j_s_fluid, i_s, N)

        # invert directions of velocities of wall particles
        vel = comp_bc_interm(vel, i_s, j_s, w_j_s_fluid, w_i_sum)

        # discrete points
        dist = (((r_target[:, None] - state["r"][None, :]) ** 2).sum(axis=-1)) ** 0.5
        w_dist = kernel_fn.w(dist)
        # weight normalization for non-full support
        w_norm = w_dist.sum(axis=-1) * cfg.case.dx**dim

        u_val = (w_dist * vel[:, dim_ind][None, :]).sum(axis=1)
        u_val *= cfg.case.dx**dim
        u_val /= w_norm

        return u_val

    def interp_scalar(src_path: str, r_target: array, prop: str = "p"):
        """Interpolator for scalar quantities.

        Args:
            src_path: Path to the source state.
            r_target: Target positions.
            prop: Which quantity to use. Defaults to 'u'.
        """
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
        w_j_s_fluid = w_dist * jnp.where(state["tag"][j_s] == Tag.FLUID, 1.0, 0.0)
        # sheparding denominator
        w_i_sum = ops.segment_sum(w_j_s_fluid, i_s, N)

        p = comp_bc_interm_scalar(p, i_s, j_s, w_j_s_fluid, w_i_sum)

        # discrete points
        dist = (((r_target[:, None] - state["r"][None, :]) ** 2).sum(axis=-1)) ** 0.5
        w_dist = kernel_fn.w(dist)
        # weight normalization for non-full support
        w_norm = w_dist.sum(axis=-1) * cfg.case.dx**dim

        p_val = (w_dist * p).sum(axis=1)
        p_val *= cfg.case.dx**dim
        p_val /= w_norm

        return p_val

    if prop_type == "vector":
        return interp_vel
    elif prop_type == "scalar":
        return interp_scalar
