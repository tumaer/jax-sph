#!/bin/bash

# # # ####### 2D Dam break
./venv/bin/python main.py --case=DB --solver=RIE2 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_Riemann --density-evolution --is-limiter --eta-limiter=3 

# Run validation script
./venv/bin/python validation_paper/validate_paper.py --case=2D_DB --src_dir="data_valid/db2d_Riemann"

