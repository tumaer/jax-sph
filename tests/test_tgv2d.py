import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from jax_sph.io_state import read_h5
from jax_sph.utils import get_ekin, get_val_max

# get refernce solutions
Ekin = np.loadtxt("tests/ref/tgv_2d_Ekin_ref.txt")
u_max = np.loadtxt("tests/ref/tgv_2d_u_max_ref.txt")
ref = np.vstack((Ekin, u_max))

Ekin = np.loadtxt("tests/ref/tgv_2d_Ekin_ref_tvf.txt")
u_max = np.loadtxt("tests/ref/tgv_2d_u_max_ref_tvf.txt")
ref_tvf = np.vstack((Ekin, u_max))

Ekin = np.loadtxt("tests/ref/tgv_2d_Ekin_ref_Rie.txt")
u_max = np.loadtxt("tests/ref/tgv_2d_u_max_ref_Rie.txt")
ref_Rie = np.vstack((Ekin, u_max))

# get 2D Taylor Green flow SPH solutions by running various simulations
os.system(
    "python main.py config=cases/tgv.yaml case.mode=rlx solver.tvf=1.0 case.dim=2 case.dx=0.02 seed=123 case.r0_noise_factor=0.25 io.data_path=/tmp/tgv2d_relaxed/ eos.p_bg_factor=0.02"
)
os.system(
    "python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=1.0 case.dim=2 case.dx=0.02 solver.t_end=5 io.write_every=25 case.state0_path=/tmp/tgv2d_relaxed/tgv_2_0.02_123.h5 io.data_path=/tmp/tgv2d_tvf/"
)
os.system(
    "python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=0.0 case.dim=2 case.dx=0.02 solver.t_end=5 io.write_every=25 case.state0_path=/tmp/tgv2d_relaxed/tgv_2_0.02_123.h5 io.data_path=/tmp/tgv2d/"
)
os.system(
    "python main.py config=cases/tgv.yaml solver.name=RIE solver.tvf=0.0 case.dim=2 case.dx=0.02 solver.t_end=5 io.write_every=25 case.state0_path=/tmp/tgv2d_relaxed/tgv_2_0.02_123.h5 io.data_path=/tmp/tgv2d_Rie/ solver.density_evolution=True"
)

dirs = os.listdir("/tmp/tgv2d/")
dirs_tvf = os.listdir("/tmp/tgv2d_tvf/")
dirs_Rie = os.listdir("/tmp/tgv2d_Rie/")

dirs = [d for d in dirs if ("2D_TGV_SPH" in d)]
dirs_tvf = [d for d in dirs_tvf if ("2D_TGV_SPH" in d)]
dirs_Rie = [d for d in dirs_Rie if ("2D_TGV_RIE" in d)]

dirs = sorted(dirs, reverse=True)
dirs_tvf = sorted(dirs_tvf, reverse=True)
dirs_Rie = sorted(dirs_Rie, reverse=True)

cfg = OmegaConf.load(os.path.join("/tmp/tgv2d/", dirs[0], "config.yaml"))
cfg_tvf = OmegaConf.load(os.path.join("/tmp/tgv2d_tvf/", dirs_tvf[0], "config.yaml"))
cfg_Rie = OmegaConf.load(os.path.join("/tmp/tgv2d_Rie/", dirs_Rie[0], "config.yaml"))

files = os.listdir(os.path.join("/tmp/tgv2d/", dirs[0]))
files_tvf = os.listdir(os.path.join("/tmp/tgv2d_tvf/", dirs_tvf[0]))
files_Rie = os.listdir(os.path.join("/tmp/tgv2d_Rie/", dirs_Rie[0]))
files_h5 = [f for f in files if (".h5" in f)]
files_h5 = sorted(files_h5)
files_h5_tvf = [f for f in files_tvf if (".h5" in f)]
files_h5_tvf = sorted(files_h5_tvf)
files_h5_Rie = [f for f in files_Rie if (".h5" in f)]
files_h5_Rie = sorted(files_h5_Rie)

u_max = np.zeros((len(files_h5)))
Ekin = np.zeros((len(files_h5)))

for i, filename in enumerate(files_h5):
    state = read_h5(os.path.join("/tmp/tgv2d/", dirs[0], filename))
    u_max[i] = get_val_max(state, "v")
    Ekin[i] = get_ekin(state, 0.02)
sol = np.vstack((Ekin, u_max))

for i, filename in enumerate(files_h5_tvf):
    state = read_h5(os.path.join("/tmp/tgv2d_tvf/", dirs_tvf[0], filename))
    u_max[i] = get_val_max(state, "v")
    Ekin[i] = get_ekin(state, 0.02)
sol_tvf = np.vstack((Ekin, u_max))

for i, filename in enumerate(files_h5_Rie):
    state = read_h5(os.path.join("/tmp/tgv2d_Rie/", dirs_Rie[0], filename))
    u_max[i] = get_val_max(state, "v")
    Ekin[i] = get_ekin(state, 0.02)
sol_Rie = np.vstack((Ekin, u_max))


@pytest.mark.parametrize(
    "solution, ref_solution, cfg",
    [(sol_tvf, ref_tvf, cfg_tvf), (sol, ref, cfg), (sol_Rie, ref_Rie, cfg_Rie)],
)
def test_pf2d(solution, ref_solution, cfg):
    """Test whether the Taylor Green flow simulation matches the refernce solution"""
    name = str(cfg.solver.name)
    tvf = str(cfg.solver.tvf)
    assert np.allclose(
        solution, ref_solution, atol=1e-2
    ), f"{name} solution with {tvf} tvf does not match the reference solution."
