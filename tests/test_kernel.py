"""Test whether the interpolation kernels are implemented correctly."""

import numpy as np
from jax import vmap

from jax_sph.kernels import QuinticKernel, WendlandC2Kernel


# TODO: convert to pytest test
def test_kernel_1d():
    """Test the interpolation kernels in 1 dimension."""

    args_kernels = {"class": [QuinticKernel, WendlandC2Kernel], "dx_factor": [1, 1.3]}
    args = {"h": [1, 0.1, 0.01], "dim": [1, 1, 1]}
    N = 500

    for i, (h, dim) in enumerate(zip(args["h"], args["dim"])):
        for j, (Kernel, dx_factor) in enumerate(
            zip(args_kernels["class"], args_kernels["dx_factor"])
        ):
            kernel = Kernel(h=dx_factor * h, dim=dim)
            dx = kernel.cutoff / N
            x = np.linspace(dx / 2, kernel.cutoff + dx / 2, N)
            w = vmap(kernel.w)(x)
            w_grad = vmap(kernel.grad_w)(x)
            int = dx * np.sum(w)

            assert np.isclose(int, 0.5, atol=1e-2), f"Half integral of kernel is {int}."
            assert (w >= 0).all(), "Kernel is negative."
            assert (w_grad <= 0).all(), "Kernel grad positive."


if __name__ == "__main__":
    test_kernel_1d()
