"""Turbulence jax-sph utils."""


import jax.numpy as jnp
import numpy as np
from jax import ops, vmap
from jax.scipy.special import factorial
from numpy import array
from scipy.spatial import KDTree

from jax_sph.jax_md import space

EPS = jnp.finfo(float).eps


class M4PrimeKernel:
    """The M'4 kernel"""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h

    def w(self, r):
        """Evaluates the kernel at the radial displacement vector r."""
        q = r * self._one_over_h
        q1 = 1 - 2.5 * q**2 + 1.5 * q**3
        q2 = 0.5 * (1 - q) * (2 - q) ** 2

        res = jnp.where(q < 1, q1, 0)
        res = jnp.where((q >= 1) * (q < 2), q2, res)
        return jnp.prod(res)  # , axis=1


class FourierQuinticKernel:
    """The quintic kernel function of Morris in Fourier space."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 1.0 / 120.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 7.0 / 478.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / jnp.pi * self._one_over_h**3

    def w(self, k):
        q1 = (jnp.exp(-2 * jnp.pi * 1j * k) * (3 * jnp.exp(2 * jnp.pi * 1j * k) * \
                (44 * jnp.pi**5 * 1j**5 * k**5 - 20 * jnp.pi**3 * 1j**3 * k**3 + 30 * \
                jnp.pi * 1j * k - 25) - 2 * jnp.pi * 1j * k * ( \
                jnp.pi * 1j * k * (jnp.pi * 1j * k * (jnp.pi * 1j * k * (26 * jnp.pi * \
                1j * k - 25) + 10) + 15) - 30) + 75)) / (4 * jnp.pi**6 * 1j**6 * k**6)
        q2 = (jnp.exp(-4 * jnp.pi * 1j * k) * (-2 * jnp.pi * 1j * k * (jnp.pi * 1j * k * \
                (jnp.pi * 1j * k * (jnp.pi * 1j * k * (2 * jnp.pi * 1j * k - 5) + 10) - 15) \
                + 15) + jnp.exp(2 * jnp.pi * 1j * k) * (4 * jnp.pi * 1j * k * (jnp.pi * \
                1j * k * (jnp.pi * 1j * k * (jnp.pi * 1j * k * (26 * jnp.pi * 1j * k \
                - 25) + 10) + 15) - 30) + 75) - 75)) / (8 * jnp.pi**6 * 1j**6 * k**6)
        q3 = (jnp.exp(-6 * jnp.pi * 1j * k) * (jnp.exp(2 * jnp.pi * 1j * k) * (2 * \
                jnp.pi * 1j * k * (jnp.pi * 1j * k * (jnp.pi * 1j * k * \
                (jnp.pi * 1j * k * (2 * jnp.pi * 1j * k - 5) + 10) - 15) + 15) - 15) + 15)) \
                / (8 * jnp.pi**6 * 1j**6 * k**6)
        return self._sigma * (q1 + q2 + q3)


def get_real_wavenumber_grid(n, dim):
    Nf = n // 2 + 1
    k = np.fft.fftfreq(n, 1.0 / n)  # for other dimensions
    kx = k[:Nf].copy()
    kx[-1] *= -1
    if dim == 2:
        k_field = np.array(np.meshgrid(kx, k, indexing="ij"), dtype=int)
    elif dim == 3:
        k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    return k_field, k


def get_energy_spectrum(vel):
    """JAX implemented energy spectrum computation on a grid."""

    dim = vel.shape[0]
    ns = vel.shape[1:]

    # check for square box with equal side length
    assert jnp.array_equal(ns, jnp.ones(dim) * ns[0])

    n = ns[0]

    k_field, k = get_real_wavenumber_grid(n, dim)
    k_mag = jnp.sqrt(jnp.sum(k_field**2, axis=0))

    shell = (k_mag + 0.5).astype(int)
    fact = (
        2 * (k_field[0] > 0) * (k_field[0] < n // 2)
        + 1 * (k_field[0] == 0)
        + 1 * (k_field[0] == n // 2)
    )

    # Fourier transform
    if dim == 1:
        # TODO: check whether 1D is working
        vel_hat = jnp.fft.rfftn(vel, axes=0)
    elif dim == 2:
        vel_hat = jnp.stack([jnp.fft.rfftn(vel[ii], axes=(1, 0)) for ii in range(dim)])
    elif dim == 3:
        vel_hat = jnp.stack(
            [jnp.fft.rfftn(vel[ii], axes=(2, 1, 0)) for ii in range(dim)]
        )

    ek = jnp.zeros(n)
    n_samples = jnp.zeros(n)

    uu = fact * 0.5 * (jnp.sum(jnp.abs(vel_hat**2), axis=0))

    ek = ek.at[shell.flatten()].add(uu.flatten())
    n_samples = n_samples.at[shell.flatten()].add(1)
    ek *= 4 * jnp.pi * k**2 / (n_samples + EPS)
    ek *= 1 / (n**dim)

    return ek


def sph_fourier_transformation(quant, r, m, rho, hl, k):
    """SPH summation instead of fourier integral.

    from "Direct numerical simulation of decaying two-dimensional turbulence in a
    no-slip square box using smoothed particle hydrodynamics", by Martin Robinson
    and Joseph J. Monaghan, 2011
    """
    return quant * jnp.exp(-1j * jnp.pi / hl * jnp.dot(k, r)) * m / rho / hl**1


def pbc_copy_vector(
    r: array, f: array, box_size: array, halo: float, dim: int, unsorted: bool = True
):
    """Copy particles of a vector field for PBCs on a rectangular domain

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        f (np.ndarray): vector field, e.g. velocity of shape (N, dim)
        box_size (np.ndarray): Domain box of form np.array([x_size, y_size, z_size])
        halo (float): Width of halo region, e.g. 3h for Quintic spline
        dim (int): dimension of the data
        unsorted (bool): whether indices remain the same

    Returns:
        (np.ndarray, np.ndarray): new positions and properties after copying
    """
    if dim == 1:
        # get right side indices
        right = np.where(r <= halo, True, False)

        # get left side indices
        left = np.where(r >= box_size - halo, True, False)

        # concatenate pbc values
        r = np.concatenate((r, r[right] + box_size, r[left] - box_size))
        f = np.concatenate((f, f[right], f[left]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind]
        f_pbc = f[ind]

    elif dim == 2:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right, :], f[left, :]), axis=0)

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1]])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top, :], f[bottom, :]), axis=0)

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind, :]

    elif dim == 3:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0, 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right, :], f[left, :]), axis=0)

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1], 0.0])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top, :], f[bottom, :]), axis=0)

        # now front and back
        # get indices
        back = np.where(r[:, 2] <= halo, True, False)
        front = np.where(r[:, 2] >= box_size[2] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, 0.0, box_size[2]])[None, :]
        r = np.concatenate((r, r[back, :] + dr, r[front, :] - dr), axis=0)
        f = np.concatenate((f, f[back, :], f[front, :]), axis=0)

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind, :]

    return r_pbc, f_pbc


def pbc_copy_scalar(
    r: array, f: array, box_size: array, halo: float, dim: int, unsorted: bool = True
):
    """Copy particles of a scalar field for PBCs on a rectangular domain

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        f (np.ndarray): vector field, e.g. velocity of shape (N, dim)
        box_size (np.ndarray): Domain box of form np.array([x_size, y_size, z_size])
        halo (float): Width of halo region, e.g. 3h for Quintic spline
        dim (int): dimension of the data
        unsorted (bool): whether indices remain the same

    Returns:
        (np.ndarray, np.ndarray): new positions and properties after copying
    """
    if dim == 1:
        # get right side indices
        right = np.where(r <= halo, True, False)

        # get left side indices
        left = np.where(r >= box_size - halo, True, False)

        # concatenate pbc values
        r = np.concatenate((r, r[right] + box_size, r[left] - box_size))
        f = np.concatenate((f, f[right], f[left]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind]
        f_pbc = f[ind]

    elif dim == 2:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1]])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    elif dim == 3:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0, 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1], 0.0])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # now front and back
        # get indices
        back = np.where(r[:, 2] <= halo, True, False)
        front = np.where(r[:, 2] >= box_size[2] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, 0.0, box_size[2]])[None, :]
        r = np.concatenate((r, r[back, :] + dr, r[front, :] - dr), axis=0)
        f = np.concatenate((f, f[back], f[front]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    return r_pbc, f_pbc


def mls_2nd_order(r, r_target, f, box_size, dx, dim):
    """2nd-order moving least squares interpolation for periodic flows in a
    rectangular box.

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        r_target (np.ndarray): coordinates of target particles of shape (N, dim)
        f (np.ndarray): scalar field, e.g. velocity of shape (N, dim)
        box_size (np.ndarray): Domain box, e.g. np.array([1., 2., 3.])
        dx (float): average particle spacing
        dim (int): dimension of the vector field

    Returns:
        (np.ndarray): interpolated values of f on r_target
    """

    # define kernel function
    kernel_fn = M4PrimeKernel(h=dx, dim=dim)

    # displacement function for neighbors list
    displacement_fn, shift_fn = space.periodic(side=box_size)

    # number of target particles
    n_target = jnp.shape(r_target)[0]

    # enforce periodic boundary conditions
    r_pbc, f_pbc = pbc_copy_scalar(r, f, box_size, kernel_fn.cutoff, dim)

    # compute edge list
    tree = KDTree(r_pbc)
    senders = tree.query_ball_point(r_target, kernel_fn.cutoff * 1.415)
    i_s = np.repeat(range(n_target), [len(x) for x in senders])
    j_s = np.concatenate(senders, axis=0)

    # precompute quantities
    r_ji = vmap(displacement_fn)(r_pbc[j_s], r_target[i_s])
    w_dist = vmap(kernel_fn.w)(r_ji)

    # define size of the linear system of equations
    mat_size = 4 + round(factorial(dim))

    # calculate indices
    ind_d = jnp.diag_indices(dim)
    ind_u = jnp.triu_indices(dim, 1)

    # mls matrix entries
    def matrix(w_dist, r_ji):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)

        row = jnp.ones(mat_size)
        row = row.at[1 : dim + 1].mul(r_ji)
        row = row.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d] * 0.5)
        row = row.at[2 * dim + 1 :].mul(tensor[ind_u])

        column = jnp.ones(mat_size)
        column = column.at[1 : dim + 1].mul(r_ji)
        column = column.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        column = column.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)

        return jnp.tensordot(column, row, axes=0) * w_dist

    # calculate matrix
    temp = vmap(matrix)(w_dist, r_ji)
    mat = ops.segment_sum(temp, i_s, n_target)

    # define solution vector entries
    def vector(w_dist, r_ji, f_j):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)
        vector = jnp.ones(mat_size) * w_dist * f_j
        vector = vector.at[1 : dim + 1].mul(r_ji)
        vector = vector.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        vector = vector.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)
        return vector

    # calculate vector
    temp = vmap(vector)(w_dist, r_ji, f_pbc[j_s])
    vec = ops.segment_sum(temp, i_s, n_target)

    # function for solving the system
    def solve_lin(matrix, vector):
        temp = jnp.linalg.solve(matrix, vector)
        return temp[0]

    # calculate interpolated value
    f_target = vmap(solve_lin)(mat, vec)

    return f_target
