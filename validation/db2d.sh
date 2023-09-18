#!/bin/bash

# # # ####### 2D Dam break
# baseline
./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf --density-evolution --artificial-alpha=0.1
# with renormalization
./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_renorm --density-evolution --artificial-alpha=0.1 --density-renormalize
# with renormalization and additional viscosity for stability
./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.00005 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_renorm_lessvisc --density-evolution --artificial-alpha=0.1 --density-renormalize
# higher resolution
./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.01 --t-end=7.5 --dt=0.0001 --viscosity=0.0 --free-slip --write-every=100 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_fine --density-evolution --artificial-alpha=0.2

./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf"
./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm"
./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm_visc"
./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm_lessvisc"
