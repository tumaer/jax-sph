#!/bin/bash
# Validation of the 2D Poiseuille Flow
# Reference result from:
# "Modeling Low Reynolds Number Incompressible Flows Using SPH", Morris 1997

# Generate data
python main.py config=cases/pf.yaml solver.tvf=1.0 io.data_path=data_valid/pf2d_tvf/
python main.py config=cases/pf.yaml solver.tvf=0.0 io.data_path=data_valid/pf2d_notvf/

# Run validation script
python validation/validate.py --case=2D_PF --src_dir=data_valid/pf2d_tvf/
python validation/validate.py --case=2D_PF --src_dir=data_valid/pf2d_notvf/
