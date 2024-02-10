#!/bin/bash
# Validation of the 2D Lid-Driven Cavity
# Reference result from:
# "A transport-velocity formulation [...]", Adami 2013

# Generate data
./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.02 --nxnynz=50_50_0 --seed=123 --write-h5 --write-every=10 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
./venv/bin/python main.py --case=LDC --solver=SPH --viscosity=0.01 --tvf=1.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=20.0 --seed=123 --write-h5 --write-every=100 --write-vtk --data-path="data_valid/ldc2d_tvf"
./venv/bin/python main.py --case=LDC --solver=SPH --viscosity=0.01 --tvf=0.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=20.0 --seed=123 --write-h5 --write-every=100 --write-vtk --data-path="data_valid/ldc2d_notvf" --p-bg-factor=0.01
./venv/bin/python main.py --case=LDC --solver=RIE2 --viscosity=0.01 --tvf=0.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=20.0 --seed=123 --write-h5 --write-every=100 --write-vtk --data-path="data_valid/ldc2d_Riemann" --density-evolution --is-limiter --eta-limiter=3


# Run validation script
./venv/bin/python validation_paper/validate_paper.py --case=2D_LDC --src_dir_Rie="data_valid/ldc2d_Riemann" --src_dir_tvf="data_valid/ldc2d_tvf" --src_dir_notvf="data_valid/ldc2d_notvf"
