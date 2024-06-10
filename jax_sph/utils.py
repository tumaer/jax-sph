"""General jax-sph utils."""

import enum
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from numpy import array
from omegaconf import DictConfig
from scipy.spatial import KDTree

from jax_sph.io_state import read_h5
from jax_sph.jax_md import partition, space
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


def pos_box_2d(fluid_box: array, dx: float, num_wall_layers: int = 3):
    """Create an empty box of particles in 2D.

    fluid_box is an array of the form: [L, H]
    The box is of size (L + num_wall_layers * dx) x (H + num_wall_layers * dx).
    The inner part of the box starts at (num_wall_layers * dx, num_wall_layers * dx).
    """
    dxn = num_wall_layers * dx
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


def pos_box_3d(fluid_box: array, dx: float, num_wall_layers: int = 3):
    """Create an z-periodic empty box of particles in 3D.

    fluid_box is an array of the form: [L, H, D]
    The box is of size (L + num_wall_layers * dx) x (H + num_wall_layers * dx) x D.
    The inner part of the box starts at (num_wall_layers * dx, num_wall_layers * dx).
    """
    dxn = num_wall_layers * dx
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
        val_array = state[var]  # TODO: check difference to jnp.absolute(state[var])
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


def get_box_nws(box_size, dx, n_walls, dim, rho, m):
    """Computes the normal vectors at box wall boundaries"""

    # TODO: having a pos_box_3d would be useful
    box = box_size - 2 * n_walls * dx

    # define 5 layers of wall BC partilces and position them accordingly
    layers = {}
    idx_len = {}
    for i in range(5):
        layer = pos_box_2d(box + 2 * i * dx, dx, 1)
        layers[f"layer_{i}"] = layer + np.ones(2) * ((n_walls - 1) - i) * dx
        idx_len[f"len_{i}"] = len(layer)

    # define kernel function
    kernel_fn = QuinticKernel(h=dx, dim=dim)

    # define function to calculate phi, Zhang (2017)
    def wall_phi_vec(rho_j, m_j, dr_ij, dist):
        # Compute unit vector, above eq. (6), Zhang (2017)
        e_ij_w = dr_ij / (dist + EPS)

        # Compute kernel gradient
        kernel_grad = kernel_fn.grad_w(dist) * (e_ij_w)

        # compute phi eq. (15), Zhang (2017)
        phi = -1.0 * m_j / rho_j * kernel_grad

        return phi

    nw = []
    for i in range(3):
        # setup of the temporary box, consisting out of 3 particle layers
        temp_box = np.concatenate(
            (
                layers[f"layer_{i}"],
                layers[f"layer_{i + 1}"],
                layers[f"layer_{i + 2}"],
            ),
            axis=0,
        )
        # define KD tree and get neighbors
        tree = KDTree(temp_box)
        neighbors = tree.query_ball_point(
            temp_box[0 : idx_len[f"len_{i}"]], 3 * dx, p=2.0
        )
        # get neighbor and nw indices
        neighbors_idx = np.concatenate(neighbors, axis=0)
        nw_idx = np.repeat(range(idx_len[f"len_{i}"]), [len(x) for x in neighbors])

        # calculate distances
        dr_ij = vmap(space.pairwise_displacement)(
            temp_box[nw_idx], temp_box[neighbors_idx]
        )
        dist = space.distance(dr_ij)

        # calculate normal vectors
        temp = vmap(wall_phi_vec)(rho[neighbors_idx], m[neighbors_idx], dr_ij, dist)
        phi = ops.segment_sum(temp, nw_idx, idx_len[f"len_{i}"])
        nw_temp = phi / (np.linalg.norm(phi, ord=2, axis=1) + EPS)[:, None]
        nw.append(nw_temp)

    nw = np.concatenate(nw, axis=0)
    nw = np.where(np.absolute(nw) < EPS, 0.0, nw)
    r_nw = np.concatenate(
        (
            layers["layer_0"],
            layers["layer_1"],
            layers["layer_2"],
        ),
        axis=0,
    )

    return nw, r_nw


def get_nws(r, tag, fluid_size, dx, offset_vec, wall_part_fn):
    """Computes the normal vectors of all wall boundaries"""

    # align fluid to [0, 0]
    r_aligned = r - offset_vec

    # define fine layer of wall BC partilces and position them accordingly
    layer = wall_part_fn(fluid_size, dx / 5, 1) - np.ones(2) * dx / 5

    # match thin layer to particles
    tree = KDTree(layer)
    dist, match_idx = tree.query(r_aligned, k=1)
    dr = layer[match_idx] - r_aligned

    # compute normal vectors
    nw = dr / (dist[:, None] + EPS)
    nw = np.where(np.isin(tag, wall_tags)[:, None], nw, np.zeros(2))

    return nw


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
