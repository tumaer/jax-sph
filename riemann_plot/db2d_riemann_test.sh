#!/bin/bash

# # # ####### 2D Dam break
#./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.01 --nxnynz=200_100_0 --seed=123 --write-h5 --r0-noise-factor=0.25 --data-path=data_relaxed
# baseline
#./venv/bin/python main.py --case=DB --solver=RIE2 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_Riemann/db2d_notvf --density-evolution --is-limiter --eta-limiter=3
#./venv/bin/python main.py --case=DB --solver=RIE2 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_Riemann/db2d_notvf --density-evolution --is-limiter --eta-limiter=3 

# with renormalization
./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.0 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_renorm --density-evolution --artificial-alpha=0.1 --density-renormalize
# with renormalization and additional viscosity for stability
#./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=7.5 --dt=0.0002 --viscosity=0.00005 --free-slip --write-every=50 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_renorm_lessvisc --density-evolution --artificial-alpha=0.1 --density-renormalize
# higher resolution
#./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.01 --t-end=7.5 --dt=0.0001 --viscosity=0.0 --free-slip --write-every=100 --write-vtk --write-h5 --seed=123 --data-path=data_valid/db2d_notvf_fine --density-evolution --artificial-alpha=0.2

#./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_Riemann/db2d_notvf"
#./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm"
#./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm_visc"
#./venv/bin/python validation/validate.py --case=2D_DB --src_dir="data_valid/db2d_notvf_renorm_lessvisc"
