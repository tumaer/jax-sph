"""SPH interpolation kernels"""

import jax.numpy as jnp
from jax import grad


class QuinticKernel:
    """The quintic kernel function of Morris."""

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

    def w(self, r):
        """Evaluates the kernel at the radial distance r."""

        q = r * self._one_over_h
        q1 = jnp.maximum(0.0, 1.0 - q)
        q2 = jnp.maximum(0.0, 2.0 - q)
        q3 = jnp.maximum(0.0, 3.0 - q)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""

        return grad(self.w)(r)
