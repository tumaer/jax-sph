#!/bin/bash
# Validation of the 2D Couette Flow
# Reference result from:
# "Modeling Low Reynolds Number Incompressible Flows Using SPH", Morris 1997

# Generate data
python main.py config=cases/cf.yaml solver.tvf=1.0 io.data_path=data_valid/cf2d_tvf/
python main.py config=cases/cf.yaml solver.tvf=0.0 io.data_path=data_valid/cf2d_notvf/
python main.py config=cases/cf.yaml solver.tvf=0.0 solver.name=RIE solver.density_evolution=True io.data_path=data_valid/cf2d_Rie/

# Run validation script
python validation/validate.py --case=2D_CF --src_dir=data_valid/cf2d_tvf/
python validation/validate.py --case=2D_CF --src_dir=data_valid/cf2d_notvf/
python validation/validate.py --case=2D_CF --src_dir=data_valid/cf2d_Rie/