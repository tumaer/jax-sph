"""Equation of state."""

from abc import ABC, abstractmethod

import jax.numpy as jnp


class BaseEoS(ABC):
    """Base class for SPH equation of state."""

    @abstractmethod
    def p_fn(self, rho: float):
        """Compute pressure from density."""
        pass

    @abstractmethod
    def rho_fn(self, p: float):
        """Compute density from pressure."""
        pass


class TaitEoS(BaseEoS):
    """Tait equation of state.

    From: "A generalized wall boundary condition for smoothed particle
    hydrodynamics", Adami et al 2012
    """

    def __init__(self, p_ref, rho_ref, p_background, gamma):
        self.p_ref = p_ref
        self.rho_ref = rho_ref
        self.p_bg = p_background
        self.gamma = gamma

    def p_fn(self, rho, *phase):
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p, *phase):
        p_temp = p + self.p_ref - self.p_bg
        return self.rho_ref * (p_temp / self.p_ref) ** (1 / self.gamma)


class MultiphaseTaitEoS(BaseEoS):
    """Tait equation of state for multiphase simulations.

    From: "A generalized wall boundary condition for smoothed particle
    hydrodynamics", Adami et al 2012
    """

    def __init__(self, p_ref, rho_ref, rho_ref_factor, p_background, gamma):
        const = jnp.ones(1)
        fac = jnp.array(rho_ref_factor).ravel()
        self.p_ref = p_ref * jnp.concatenate((const, fac))
        self.rho_ref = rho_ref * jnp.concatenate((const, fac))
        self.p_bg = p_background  # TODO: unified pb correct?
        self.gamma = gamma

    def p_fn(self, rho, phase):
        return (
            self.p_ref[phase] * ((rho / self.rho_ref[phase]) ** self.gamma - 1)
            + self.p_bg
        )

    def rho_fn(self, p, phase):
        p_temp = p + self.p_ref[phase] - self.p_bg
        return self.rho_ref[phase] * (p_temp / self.p_ref[phase]) ** (1 / self.gamma)


class MultiphaseHighRhoTaitEoS(BaseEoS):
    """Tait equation of state for multiphase simulations with high densitz ratios.

    From: "A simple SPH algorithm for multi-fluid flow with high density ratios",
    Monaghan and Rafiee 2012
    """

    def __init__(
        self, u_ref, u_ref_factor, rho_ref, rho_ref_factor, p_background, gamma
    ):
        const = jnp.ones(1)
        rho_fac = jnp.array(rho_ref_factor).ravel()
        u_fac = jnp.array(u_ref_factor).ravel()
        self.rho_ref = rho_ref * jnp.concatenate((const, rho_fac))
        self.p_bg = p_background  # TODO: unified pb correct?
        self.u_ref = u_ref * jnp.concatenate((const, u_fac))
        self.gamma = gamma

    def p_fn(self, rho, phase):
        coeff = self.rho_ref[phase] * self.u_ref[phase] ** 2 / self.gamma
        return coeff * ((rho / self.rho_ref[phase]) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p, phase):
        coeff = self.rho_ref[phase] * self.u_ref[phase] ** 2 / self.gamma
        p_temp = p + coeff - self.p_bg
        return self.rho_ref[phase] * (p_temp / coeff) ** (1 / self.gamma)


class RIEMANNEoS(BaseEoS):
    """Riemann SPH equation of state.

    From: "A weakly compressible SPH method based on a
    low-dissipation Riemann solver", Zhang, Hu, Adams, 2017
    """

    def __init__(self, rho_ref, p_background, u_ref):
        self.rho_ref = rho_ref
        self.u_ref = u_ref
        self.p_bg = p_background

    def p_fn(self, rho, *phase):
        return 100 * self.u_ref**2 * (rho - self.rho_ref) + self.p_bg

    def rho_fn(self, p, *phase):
        return (p - self.p_bg) / (100 * self.u_ref**2) + self.rho_ref
