"""Equation of state"""

class TaitEoS:
    """Equation of state

    From: "A generalized wall boundary condition for smoothed particle
    hydrodynamics", Adami et al 2012
    """

    def __init__(self, p_ref, rho_ref, p_background, gamma):
        self.p_ref = p_ref
        self.rho_ref = rho_ref
        self.p_bg = p_background
        self.gamma = gamma

    def p_fn(self, rho):
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p):
        p_temp = p + self.p_ref - self.p_bg
        return self.rho_ref * (p_temp / self.p_ref) ** (1 / self.gamma)
    


class RIEMANNEoS:
    """Equation of state

    From: "A weakly compressible SPH method based on a
    low-dissipation Riemann solver", Zhang, Hu, Adams, 2017
    """

    def __init__(self, rho_ref, p_background, vmax):
        self.rho_ref = rho_ref
        self.vmax = vmax
        self.p_bg = p_background

    def p_fn(self, rho):
        return 100 * self.vmax ** 2 * (rho - self.rho_ref) + self.p_bg
    
    def rho_fn(self, p):
        return (p - self.p_bg) / (100 * self.vmax ** 2) + self.rho_ref
