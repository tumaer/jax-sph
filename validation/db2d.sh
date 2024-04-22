#!/bin/bash

####### 2D Dam break
python main.py config=cases/db.yaml case.u_ref=2 case.viscosity=0.0 solver.name=RIE solver.t_end=7.5 solver.dt=0.0002 solver.free_slip=True solver.artificial_alpha=0.0 solver.eta_limiter=3 io.write_every=50 io.data_path=data_valid/db2d_Riemann/

# Run validation script
python validation/validate.py --case=2D_DB --src_dir=data_valid/db2d_Riemann/
