"""Energy Spectrum implementation"""

import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space

from jax_sph.kernels import QuinticKernel
from jax_sph.kernels import WendlandC2Kernel
from jax_sph.kernels import M4PrimeKernel