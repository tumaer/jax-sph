"""General jax-sph utils."""

import enum
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from numpy import array
from omegaconf import DictConfig
from scipy.spatial import KDTree

from jax_sph.io_state import read_h5
from jax_sph.jax_md import partition, space
from jax_sph.jax_md.partition import Dense
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


def pos_box_2d(fluid_box: array, dx: float, n_walls: int = 3):
    """Create an empty box of particles in 2D.

    fluid_box is an array of the form: [L, H]
    The box is of size (L + n_walls * dx) x (H + n_walls * dx).
    The inner part of the box starts at (n_walls * dx, n_walls * dx).
    """
    # thickness of wall particles
    dxn = n_walls * dx

    # horizontal and vertical blocks
    vertical = pos_init_cartesian_2d(np.array([dxn, fluid_box[1] + 2 * dxn]), dx)
    horiz = pos_init_cartesian_2d(np.array([fluid_box[0], dxn]), dx)

    # wall: left, bottom, right, top
    wall_l = vertical.copy()
    wall_b = horiz.copy() + np.array([dxn, 0.0])
    wall_r = vertical.copy() + np.array([fluid_box[0] + dxn, 0.0])
    wall_t = horiz.copy() + np.array([dxn, fluid_box[1] + dxn])

    res = jnp.concatenate([wall_l, wall_b, wall_r, wall_t])
    return res


def pos_box_3d(fluid_box: array, dx: float, n_walls: int = 3, z_periodic: bool = True):
    """Create an z-periodic empty box of particles in 3D.

    fluid_box is an array of the form: [L, H, D]
    The box is of size (L + n_walls * dx) x (H + n_walls * dx) x D.
    The inner part of the box starts at (n_walls * dx, n_walls * dx).
    z_periodic states whether the box is periodic in z-direction.
    """
    # thickness of wall particles
    dxn = n_walls * dx

    # horizontal and vertical blocks
    vertical = pos_init_cartesian_3d(
        np.array([dxn, fluid_box[1] + 2 * dxn, fluid_box[2]]), dx
    )
    horiz = pos_init_cartesian_3d(np.array([fluid_box[0], dxn, fluid_box[2]]), dx)

    # wall: left, bottom, right, top
    wall_l = vertical.copy()
    wall_b = horiz.copy() + np.array([dxn, 0.0, 0.0])
    wall_r = vertical.copy() + np.array([fluid_box[0] + dxn, 0.0, 0.0])
    wall_t = horiz.copy() + np.array([dxn, fluid_box[1] + dxn, 0.0])

    res = jnp.concatenate([wall_l, wall_b, wall_r, wall_t])

    # add walls in z-direction
    if not z_periodic:
        res += np.array([0.0, 0.0, dxn])
        # front block
        front = pos_init_cartesian_3d(
            np.array([fluid_box[0] + 2 * dxn, fluid_box[1] + 2 * dxn, dxn]), dx
        )

        # wall: front, end
        wall_f = front.copy()
        wall_e = front.copy() + np.array([0.0, 0.0, fluid_box[2] + dxn])
        res = jnp.concatenate([res, wall_f, wall_e])

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


def get_array_stats(state: Dict, var: str = "u", operation="max"):
    """Extract the min, max, or mean of `state["var"]`.

    For vectorial quantities, use the Euclidean norm.

    Args:
        state: Simulation state dictionary.
        var: Variable to extract, i.e. dict key.
        operation: One of "min", "max", "mean".
    """
    operations = {"min": jnp.min, "max": jnp.max, "mean": jnp.mean}
    func = operations[operation]

    if jnp.size(state[var].shape) > 1:
        val_array = jnp.sqrt(jnp.square(state[var]).sum(axis=1))
    else:
        val_array = state[var]
    return func(val_array)


def get_stats(state: Dict, props: list, dx: float):
    """Extract values from `state` for printing."""

    res = {}
    for prop in props:
        if prop == "Ekin":
            res[prop] = get_ekin(state, dx)
        else:
            var, operation = prop.split("_")  # e.g. "u_max"
            res[prop] = get_array_stats(state, var, operation)
    return res


def compute_nws_scipy(r, tag, dx, n_walls, offset_vec, wall_part_fn):
    """Computes the normal vectors of all wall boundaries. Jit-able pure_callback."""

    dx_fac = 5

    # operate only on wall particles, i.e. remove fluid
    r_walls = r[np.isin(tag, wall_tags)]

    # align fluid to [0, 0]
    r_aligned = r_walls - offset_vec

    # define fine layer of wall BC partilces and position them accordingly
    layer = wall_part_fn(dx / dx_fac, 1) - offset_vec / n_walls / dx_fac

    # match thin layer to particles
    tree = KDTree(layer)
    dist, match_idx = tree.query(r_aligned, k=1)
    dr = layer[match_idx] - r_aligned
    nw_walls = dr / (dist[:, None] + EPS)
    nw_walls = jnp.asarray(nw_walls, dtype=r.dtype)

    # compute normal vectors
    nw = jnp.zeros_like(r)
    nw = nw.at[np.isin(tag, wall_tags)].set(nw_walls)

    return nw


def compute_nws_jax_wrapper(
    state0: Dict,
    dx: float,
    n_walls: int,
    offset_vec: jax.Array,
    box_size: jax.Array,
    pbc: jax.Array,
    cfg_nl: DictConfig,
    displacement_fn: Callable,
    wall_part_fn: Callable,
):
    """Compute wall normal vectors from wall to fluid. Jit-able JAX implementation.

    For the particles from `r_walls`, find the closest particle from `layer`
    and compute the normal vector from each `r_walls` particle.
    """
    r = state0["r"]
    tag = state0["tag"]

    # operate only on wall particles, i.e. remove fluid
    r_walls = r[np.isin(tag, wall_tags)] - offset_vec

    # discretize wall with one layer of 5x smaller particles
    dx_fac = 5
    offset = offset_vec / n_walls / dx_fac
    layer = wall_part_fn(dx / dx_fac, 1) - offset

    # construct a neighbor list over both point clouds
    r_full = jnp.concatenate([r_walls, layer], axis=0)

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=dx * n_walls * 2.0**0.5 * 1.01,
        backend=cfg_nl.backend,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Dense,
        num_particles_max=r_full.shape[0],
        num_partitions=cfg_nl.num_partitions,
        pbc=np.array(pbc),
    )
    num_particles = len(r_full)
    neighbors = neighbor_fn.allocate(r_full, num_particles=num_particles)

    # jit-able function
    def body(r: jax.Array):
        r_walls = r[np.isin(tag, wall_tags)] - offset_vec
        r_full = jnp.concatenate([r_walls, layer], axis=0)

        nbrs = neighbors.update(r_full, num_particles=num_particles)

        # get the relevant entries from the dense neighbor list
        idx = nbrs.idx  # dense list: [[0, 1, 5], [0, 1, 3], [2, 3, 6], ...]
        idx = idx[: len(r_walls)]  # only the wall particle neighbors
        mask_to_layer = idx > len(r_walls)  # mask toward `layer` particles
        idx = jnp.where(mask_to_layer, idx, len(r_full))  # get rid of unwanted edges

        # compute distances `r_wall` and `layer` particles and set others to infinity
        r_i_s = r_full[idx]
        dr_i_j = vmap(vmap(displacement_fn, in_axes=(0, None)))(r_i_s, r_walls)
        dist = space.distance(dr_i_j)
        mask_real = idx != len(r_full)  # identify padding entries
        dist = jnp.where(mask_real, dist, jnp.inf)

        # find closest `layer` particle for each `r_wall` particle and normalize
        # displacement vector between the two to use it as the normal vector
        idx_closest = jnp.argmin(dist, axis=1)
        nw_walls = dr_i_j[jnp.arange(len(r_walls)), idx_closest]
        nw_walls /= (dist[jnp.arange(len(r_walls)), idx_closest] + EPS)[:, None]
        nw_walls = jnp.asarray(nw_walls, dtype=r.dtype)

        # update normals only of wall particles
        nw = jnp.zeros_like(r)
        nw = nw.at[np.isin(tag, wall_tags)].set(nw_walls)

        return nw

    return body


class Logger:
    """Logger for printing stats to stdout."""

    def __init__(self, dt, dx, print_props, sequence_length) -> None:
        self.dt = dt
        self.dx = dx
        self.print_props = print_props
        self.sequence_length = sequence_length
        self.digits = len(str(sequence_length))

    def print_stats(self, state, step):
        t_ = (step + 1) * self.dt

        stats_dict = get_stats(state, self.print_props, self.dx)
        stats_str = ", ".join([f"{k}={v:.5f}" for k, v in stats_dict.items()])

        msg = f"{str(step).zfill(self.digits)}/{self.sequence_length}"
        msg += f", t={t_:.4f}, {stats_str}"
        print(msg)


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
