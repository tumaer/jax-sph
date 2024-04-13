"""Test whether the interpolation kernels are implemented correctly."""

import numpy as np
import pytest
from jax import vmap

from jax_sph.kernel import QuinticKernel, WendlandC2Kernel


@pytest.mark.parametrize(
    "Kernel, dx_factor", [(QuinticKernel, 1), (WendlandC2Kernel, 1.3)]
)
def test_kernel_1d(Kernel, dx_factor):
    """Test the interpolation kernels in 1 dimension."""

    N = 500

    for h, dim in zip([1, 0.1, 0.01], [1, 1, 1]):
        kernel = Kernel(h=dx_factor * h, dim=dim)
        dx = kernel.cutoff / N
        x = np.linspace(dx / 2, kernel.cutoff + dx / 2, N)
        w = vmap(kernel.w)(x)
        w_grad = vmap(kernel.grad_w)(x)
        int = dx * np.sum(w)

        m = f" for h = {h} and dim = {dim}."
        assert np.isclose(int, 0.5, atol=1e-2), f"Half integral of kernel is {int}{m}"
        assert (w >= 0).all(), f"Kernel is negative{m}"
        assert (w_grad <= 0).all(), f"Kernel grad positive{m}"
