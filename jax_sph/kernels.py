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
    
    def grad_w_analytical(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r analytically."""

        q = r * self._one_over_h
        q1 = jnp.maximum(0.0, 1.0 - q)
        q2 = jnp.maximum(0.0, 2.0 - q)
        q3 = jnp.maximum(0.0, 3.0 - q)

        return self._sigma * (-5.0 * q3**4 + 6.0 * 5 * q2**4 - 15.0 * 5 * q1**4)


class WendlandC2Kernel:
    """The 5th-order C2 kernel function of Wendland."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self.dim = dim

        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 5.0 / 8.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 7.0 / 4.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 21.0 / 16.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        """Evaluates the kernel at the radial distance r."""

        if self.dim == 1:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 1.5 * q + 1.0

            return self._sigma * (q1**3 * q2)
        else:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 2.0 * q + 1.0

            return self._sigma * (q1**4 * q2)

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""

        return grad(self.w)(r)
    
    def grad_w_analytical(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r analytically."""

        if self.dim == 1:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = -3.0 * q

            return self._sigma * (q1**2 * q2)
        else:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = -5.0 * q

            return self._sigma * (q1**3 * q2)
