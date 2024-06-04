"""SPH interpolation kernels."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import grad


class BaseKernel(ABC):
    """Base class for SPH interpolation kernels."""

    def __init__(self, h: float):
        self._one_over_h = 1.0 / h

    @abstractmethod
    def w(self, r):
        """Evaluates the kernel at the radial distance r."""
        pass

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""

        return grad(self.w)(r)


class CubicKernel(BaseKernel):
    """The cubic kernel function of Monaghan."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 2.0 / 3.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 10.0 / 7.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 1.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        q = r * self._one_over_h
        c1 = jnp.where(1 - q >= 0, 1, 0)
        c2 = jnp.where(jnp.logical_and(2 - q < 1, 2 - q >= 0), 1, 0)
        q1 = 1 - 1.5 * q**2 * (1 - q / 2)
        q2 = 0.25 * (2 - q) ** 3

        return self._sigma * (q1 * c1 + q2 * c2)


class QuinticKernel(BaseKernel):
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
        q = r * self._one_over_h
        q1 = jnp.maximum(0.0, 1.0 - q)
        q2 = jnp.maximum(0.0, 2.0 - q)
        q3 = jnp.maximum(0.0, 3.0 - q)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)


class WendlandC2Kernel(BaseKernel):
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


class WendlandC4Kernel(BaseKernel):
    """The 5th-order C4 kernel function of Wendland."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self.dim = dim

        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 3.0 / 4.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 9.0 / 4.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 495.0 / 256.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        if self.dim == 1:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 2.0 * q**2 + 2.5 * q + 1.0

            return self._sigma * (q1**5 * q2)
        else:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 35.0 / 12.0 * q**2 + 3 * q + 1.0

            return self._sigma * (q1**6 * q2)


class WendlandC6Kernel(BaseKernel):
    """The 5th-order C6 kernel function of Wendland."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self.dim = dim

        self._normalized_cutoff = 2.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 55.0 / 64.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 78.0 / 28.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 1365.0 / 512.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        if self.dim == 1:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 21.0 / 8.0 * q**3 + 19.0 / 4.0 * q**2 + 3.5 * q + 1.0

            return self._sigma * (q1**7 * q2)
        else:
            q = r * self._one_over_h
            q1 = jnp.maximum(0.0, 1.0 - 0.5 * q)
            q2 = 4.0 * q**3 + 6.25 * q**2 + 4 * q + 1.0

            return self._sigma * (q1**8 * q2)


class GaussianKernel(BaseKernel):
    """The gaussian kernel function of Monaghan."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        self._sigma = 1.0 / jnp.pi ** (dim / 2) * self._one_over_h ** (dim)

    def w(self, r):
        q = r * self._one_over_h
        q1 = jnp.where(3 - q >= 0, 1, 0)

        return self._sigma * q1 * jnp.exp(-(q**2))


class SuperGaussianKernel(BaseKernel):
    # TODO: We want this? Intendent but negativ in some regions
    """The supergaussian kernel function of Monaghan."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self.dim = dim

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        self._sigma = 1.0 / jnp.pi ** (dim / 2) * self._one_over_h ** (dim)

    def w(self, r):
        q = r * self._one_over_h
        q1 = jnp.where(3 - q >= 0, 1, 0)

        return self._sigma * q1 * jnp.exp(-(q**2)) * (self.dim / 2 + 1 - q**2)
