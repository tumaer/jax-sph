"""Turbulence jax-sph utils."""

import enum
from typing import Callable, Dict

import os
import jax
import jax.numpy as jnp
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from jax import ops, vmap
from numpy import array
from omegaconf import DictConfig
from scipy.spatial import KDTree

from jax_sph.io_state import read_h5
from jax_sph.jax_md import partition, space
from jax_sph.jax_md.partition import Dense
from jax_sph.kernel import QuinticKernel

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
        q1 = 1 - 2.5 * q ** 2 + 1.5 * q ** 3
        q2 = 0.5 * (1 - q) * (2 - q) ** 2

        res = jnp.where(q < 1, q1, 0)
        res = jnp.where((q >= 1)*(q < 2), q2, res)
        return jnp.prod(res) #, axis=1

def get_real_wavenumber_grid(n, dim):
    Nf = n//2 + 1
    k = np.fft.fftfreq(n, 1./n)  # for other dimensions
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
    assert (jnp.array_equal(ns, jnp.ones(dim) * ns[0]))

    n = ns[0]

    k_field, k = get_real_wavenumber_grid(n, dim)
    k_mag = jnp.sqrt(jnp.sum(k_field**2, axis=0))

    shell = (k_mag + 0.5).astype(int)
    fact = 2 * (k_field[0] > 0) * (k_field[0] < n//2) + \
        1 * (k_field[0] == 0) + 1 * (k_field[0] == n//2)

    # Fourier transform
    if dim == 1:
        raise NotImplementedError
    elif dim == 2:
        vel_hat = jnp.stack([jnp.fft.rfftn(vel[ii], axes=(1, 0)) for ii in range(dim)])
    elif dim == 3:
        vel_hat = jnp.stack([jnp.fft.rfftn(vel[ii], axes=(2, 1, 0)) for ii in range(dim)])

    ek = jnp.zeros(n)
    n_samples = jnp.zeros(n)

    uu = fact * 0.5 * (jnp.sum(jnp.abs(vel_hat**2), axis=0))

    ek = ek.at[shell.flatten()].add(uu.flatten())
    n_samples = n_samples.at[shell.flatten()].add(1)
    ek *= 4 * jnp.pi * k**2 / (n_samples + EPS)
    ek *= 1/(n**dim)

    return ek

def sph_fourier_transformation(quant, r, m, rho, l, k):
    """SPH summation instead of fourier integral.
    
    from "Direct numerical simulation of decaying two-dimensional turbulence in a
    no-slip square box using smoothed particle hydrodynamics", by Martin Robinson
    and Joseph J. Monaghan, 2011
    """
    return quant * jnp.exp(-1j * jnp.pi / l * jnp.dot(k, r)) * m / rho / l**1

# def get_sph_energy_spectrum(quant, r, m, rho, box_size):
#     nx, ny, nz = quant.shape[1:]
#     assert (nx == ny and ny == nz)
#     n = nx
#     l = box_size[0] / 2
#     dim = quant.shape[0]

#     k_field, k = get_real_wavenumber_grid(N=nx)

    


#     pass
