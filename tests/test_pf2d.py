import os

import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from jax_sph.utils import sph_interpolator


def u_series_exp(y, t, n_max=10):
    """Analytical solution to unsteady Poiseuille flow (low Re)

    Based on Series expansion as shown in:
    "Modeling Low Reynolds Number Incompressible Flows Using SPH"
    ba Morris et al. 1997
    """

    eta = 100.0  # dynamic viscosity
    rho = 1.0  # denstiy
    nu = eta / rho  # kinematic viscosity
    u_max = 1.25  # max velocity in middle of channel
    d = 1.0  # channel width
    fx = -8 * nu * u_max / d**2
    offset = fx / (2 * nu) * y * (y - d)

    def term(n):
        base = np.pi * (2 * n + 1) / d
        prefactor = 4 * fx / (nu * base**3 * d)
        sin_term = np.sin(base * y)
        exp_term = np.exp(-(base**2) * nu * t)
        return prefactor * sin_term * exp_term

    res = offset
    for i in range(n_max):
        res += term(i)

    return res


# get analytical solution
y_axis = np.linspace(0, 1, 21)
t_dimless = [0.0005, 0.001, 0.005, 0.01]
for t_val in t_dimless:
    ref = u_series_exp(y_axis, t_val)

# get 2D poiseuille flow SPH solution by running a simulation
os.system(
    "python main.py config=cases/pf.yaml solver.tvf=1.0"
    + " io.write_every=1000 io.data_path=/tmp/pf_tvf_test"
)
os.system(
    "python main.py config=cases/pf.yaml solver.tvf=0.0"
    + " io.write_every=1000 io.data_path=/tmp/pf_test"
)

dirs = os.listdir("/tmp/pf_test/")
dirs = [d for d in dirs if ("2D_PF_SPH" in d)]
dirs = sorted(dirs, reverse=True)

dirs_tvf = os.listdir("/tmp/pf_tvf_test/")
dirs_tvf = [d for d in dirs_tvf if ("2D_PF_SPH" in d)]
dirs_tvf = sorted(dirs_tvf, reverse=True)

cfg = OmegaConf.load(os.path.join("/tmp/pf_test/", dirs[0], "config.yaml"))
tvf_cfg = OmegaConf.load(os.path.join("/tmp/pf_tvf_test/", dirs_tvf[0], "config.yaml"))
step_max = np.array(np.rint(cfg.solver.t_end / cfg.solver.dt), dtype=int)
digits = len(str(step_max))

y_axis += 3 * cfg.case.dx
rs = 0.2 * jnp.ones([y_axis.shape[0], 2])
rs = rs.at[:, 1].set(y_axis)
for i in range(len(t_dimless)):
    file_name = (
        "traj_" + str(int(t_dimless[i] / tvf_cfg.solver.dt)).zfill(digits) + ".h5"
    )
    tvf_src_path = os.path.join("/tmp/pf_tvf_test/", dirs_tvf[0], file_name)
    interp_vel_fn_tvf = sph_interpolator(tvf_cfg, tvf_src_path)
    sol_tvf = interp_vel_fn_tvf(tvf_src_path, rs, prop="u", dim_ind=0)

    src_path = os.path.join("/tmp/pf_test/", dirs[0], file_name)
    interp_vel_fn = sph_interpolator(cfg, src_path)
    sol = interp_vel_fn(src_path, rs, prop="u", dim_ind=0)


@pytest.mark.parametrize("solution, ref_solution", [(sol_tvf, ref), (sol, ref)])
def test_pf2d(solution, ref_solution):
    """Test whether the poiseuille flow simulation matches the analytical solution"""
    assert np.allclose(
        solution, ref_solution, atol=1e-2
    ), "Velocity profile does not match."
