"""Test a full run of the solver on the Coette flow case from the validations."""

import os

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from omegaconf import OmegaConf

from main import load_embedded_configs


def u_series_cf_exp(y, t, n_max=10):
    """Analytical solution to unsteady Couette flow (low Re)

    Based on Series expansion as shown in:
    "Modeling Low Reynolds Number Incompressible Flows Using SPH"
    ba Morris et al. 1997
    """

    eta = 100.0  # dynamic viscosity
    rho = 1.0  # denstiy
    nu = eta / rho  # kinematic viscosity
    u_max = 1.25  # max velocity in middle of channel
    d = 1.0  # channel width

    Re = u_max * d / nu
    print(f"Couette flow at Re={Re}")

    offset = u_max * y / d

    def term(n):
        base = np.pi * n / d

        prefactor = 2 * u_max / (n * np.pi) * (-1) ** n
        sin_term = np.sin(base * y)
        exp_term = np.exp(-(base**2) * nu * t)
        return prefactor * sin_term * exp_term

    res = offset
    for i in range(1, n_max):
        res += term(i)

    return res


@pytest.fixture
def setup_simulation():
    y_axis = np.linspace(0, 1, 21)
    t_dimless = [0.0005, 0.001, 0.005]
    # get analytical solution
    ref_solutions = []
    for t_val in t_dimless:
        ref_solutions.append(u_series_cf_exp(y_axis, t_val))
    return y_axis, t_dimless, ref_solutions


def run_simulation(tmp_path, tvf, solver):
    """Emulate `main.py`."""
    data_path = tmp_path / f"cf_test_{tvf}"

    cli_args = OmegaConf.create(
        {
            "config": "cases/cf.yaml",
            "case": {"dx": 0.0333333},
            "solver": {"name": solver, "tvf": tvf, "dt": 0.000002, "t_end": 0.005},
            "io": {"write_every": 250, "data_path": str(data_path)},
        }
    )
    cfg = load_embedded_configs(cli_args)

    # Specify cuda device. These setting must be done before importing jax-md.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152 from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cfg.xla_mem_fraction)

    if cfg.dtype == "float64":
        config.update("jax_enable_x64", True)

    from jax_sph.simulate import simulate

    simulate(cfg)

    return data_path


def get_solution(data_path, t_dimless, y_axis):
    from jax_sph.utils import sph_interpolator

    dir = os.listdir(data_path)[0]
    cfg = OmegaConf.load(data_path / dir / "config.yaml")
    step_max = np.array(np.rint(cfg.solver.t_end / cfg.solver.dt), dtype=int)
    digits = len(str(step_max))

    y_axis += 3 * cfg.case.dx
    rs = 0.2 * jnp.ones([y_axis.shape[0], 2])
    rs = rs.at[:, 1].set(y_axis)
    solutions = []
    for i in range(len(t_dimless)):
        file_name = (
            "traj_" + str(int(t_dimless[i] / cfg.solver.dt)).zfill(digits) + ".h5"
        )
        src_path = data_path / dir / file_name
        interp_vel_fn = sph_interpolator(cfg, src_path)
        solutions.append(interp_vel_fn(src_path, rs, prop="u", dim_ind=0))
    return solutions


@pytest.mark.parametrize(
    "tvf, solver", [(0.0, "SPH"), (1.0, "SPH"), (0.0, "RIE"), (0.0, "DELTA")]
)
def test_cf2d(tvf, solver, tmp_path, setup_simulation):
    """Test whether the couette flow simulation matches the analytical solution"""
    y_axis, t_dimless, ref_solutions = setup_simulation
    data_path = run_simulation(tmp_path, tvf, solver)
    solutions = get_solution(data_path, t_dimless, y_axis)
    for sol, ref_sol in zip(solutions, ref_solutions):
        assert np.allclose(sol, ref_sol, atol=1e-2), "Velocity profile does not match."
