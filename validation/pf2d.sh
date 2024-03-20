#!/bin/bash
# Validation of the 2D Poiseuille Flow
# Reference result from:
# "Modeling Low Reynolds Number Incompressible Flows Using SPH", Morris 1997

# Generate data
python main.py --case=PF --solver=SPH --viscosity=100 --tvf=1.0 --dim=2 --dx=0.0166666 --dt=0.0000005 --t-end=0.01 --seed=123 --write-h5 --write-every=200 --data-path="data_valid/pf2d_tvf"
python main.py --case=PF --solver=SPH --viscosity=100 --tvf=0.0 --dim=2 --dx=0.0166666 --dt=0.0000005 --t-end=0.01 --seed=123 --write-h5 --write-every=200 --data-path="data_valid/pf2d_notvf"

# Run validation script
python validation/validate.py --case=2D_PF --src_dir="data_valid/pf2d_tvf"
python validation/validate.py --case=2D_PF --src_dir="data_valid/pf2d_notvf"
