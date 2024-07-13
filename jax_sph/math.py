"""Math jax-sph utils."""

import jax
import jax.numpy as jnp
from jax import vmap
from numpy import array

EPS = jnp.finfo(float).eps


@jax.jit
def factorial_jax(x: int):
    """JAX jit-able vmap compatible implementation for computing the factorial.

    The inputs are defined as:
    x:      variable (int)
    """

    # define body function
    def factorial_body(k, i):
        temp = k * i
        return temp

    # calculate factorial via fori loop
    res = jax.lax.fori_loop(1, x + 1, factorial_body, 1)

    return res


_erfc_nominator = jnp.array(
    [
        [0, 0, 0.56418958354775629],
        [1, 2.71078540045147805, 5.80755613130301624],
        [1, 3.47469513777439592, 12.07402036406381411],
        [1, 4.00561509202259545, 9.30596659485887898],
        [1, 5.16722705817812584, 9.12661617673673262],
        [1, 5.95908795446633271, 9.19435612886969243],
    ]
)

_erfc_denominator = jnp.array(
    [
        [0, 1, 2.06955023132914151],
        [1, 3.47954057099518960, 12.06166887286239555],
        [1, 3.72068443960225092, 8.44319781003968454],
        [1, 3.90225704029924078, 6.36161630953880464],
        [1, 4.03296893109262491, 5.13578530585681539],
        [1, 4.11240942957450885, 4.48640329523408675],
    ]
)


@jax.jit
def erfc_jax(x: float):
    """JAX jit-able vmap compatible implementation for computing the complementary
    error function.

    from "Approximate Incomplete Integrals, Application to Complementary Error
    Function", by Yaya D. Dia, 2023

    The inputs are defined as:
    x:      random variable (float)
    """

    def _erfc_fn(x):
        nom = jnp.tensordot(_erfc_nominator, jnp.array([x**2, x, 1]), axes=1)
        denom = jnp.tensordot(_erfc_denominator, jnp.array([x**2, x, 1]), axes=1)
        return jnp.exp(-1 * (x**2)) * jnp.prod(nom / denom)

    def _neg_erfc_fn(x):
        return 2 - _erfc_fn(-x)

    return jax.lax.cond(x < 0.0, _neg_erfc_fn, _erfc_fn, x)


@jax.jit
def erf_jax_power(x: float, n: int = 10):
    """JAX jit-able vmap compatible implementation for computing the error function
    as power series.

    The inputs are defined as:
    x:      random variable (float)
    n:      number of summation terms (int)
    """

    # define body function
    def erf_body(n, a):
        fac1 = 2 * (-1) ** n / jnp.sqrt(jnp.pi)
        fac2 = a[0] ** (2 * n + 1) / (factorial_jax(n) * (2 * n + 1))
        temp = fac1 * fac2
        jnp.where(temp <= EPS, 0, temp)
        a = a.at[1].add(temp)
        return a

    # calculate erf via fori loop
    res = jax.lax.fori_loop(0, n + 1, erf_body, jnp.array([x, 0]))

    return res[1]


@jax.jit
def erf_jax(x: float):
    """JAX jit-able vmap compatible implementation for computing the error function.

    The inputs are defined as:
    x:      random variable (float)
    """

    def _erf_fn(x):
        index = jnp.zeros_like(x, int)
        index = jnp.where(jnp.logical_and(x >= 1.0, x < 6.0), 1, index)
        index = jnp.where(x >= 6.0, 2, index)

        def asymp_erf(x):
            return 1.0

        def inter_erf(x):
            return 1.0 - erfc_jax(x)

        branches = [erf_jax_power, inter_erf, asymp_erf]

        return jax.lax.switch(index, branches, x)

    def _neg_erf_fn(x):
        return -_erf_fn(-x)

    return jax.lax.cond(x < 0.0, _neg_erf_fn, _erf_fn, x)


@jax.jit
def gaussian_pdf_jax(x: array, mu: float, sigma: float):
    """JAX jit-able implementation for computing the gaussian PDF.

    The inputs are defined as:
    x:      random variable (array)
    mu:     mean of x (float)
    sigma:  standard deviation of x (float)
    """

    # actual gaussian pdf
    def pdf(x):
        fac = 1 / jnp.sqrt(2 * jnp.pi * sigma**2)
        n = (x - mu) ** 2 / (2 * sigma**2)
        return fac * jnp.exp(-n)

    # calculate pdf
    res = vmap(pdf)(x)

    return res


@jax.jit
def gaussian_cdf_jax(x: array, mu: float, sigma: float):
    """JAX jit-able implementation for computing the gaussian CDF.

    The inputs are defined as:
    x:      random variable (array)
    mu:     mean of x (float)
    sigma:  standard deviation of x (float)
    """

    # actual gaussian pdf
    def cdf(x):
        fac = 1 / 2
        n = (x - mu) / (sigma * jnp.sqrt(2))
        return fac * (1 + erf_jax(n))

    # calculate pdf
    res = vmap(cdf)(x)

    return res
