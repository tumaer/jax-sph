#!/bin/bash
# Validation of the 2D Lid-Driven Cavity
# Reference result from:
# "A transport-velocity formulation [...]", Adami 2013

# Generate data
python main.py config=cases/ldc.yaml case.mode=rlx solver.tvf=1.0 seed=123 case.r0_noise_factor=0.25 io.data_path=data_relaxed/ eos.p_bg_factor=0.0 
python main.py config=cases/ldc.yaml solver.name=SPH solver.tvf=1.0 solver.t_end=20.0 case.state0_path=data_relaxed/ldc_2_0.02_123.h5 io.data_path=data_valid/ldc2d_tvf/ eos.p_bg_factor=0.0 
python main.py config=cases/ldc.yaml solver.name=SPH solver.tvf=0.0 solver.t_end=20.0 case.state0_path=data_relaxed/ldc_2_0.02_123.h5 io.data_path=data_valid/ldc2d_notvf/
python main.py config=cases/ldc.yaml solver.name=RIE solver.tvf=0.0 solver.t_end=20.0 case.state0_path=data_relaxed/ldc_2_0.02_123.h5 io.data_path=data_valid/ldc2d_Riemann/ solver.density_evolution=True eos.p_bg_factor=0.0 

# Run validation script
python validation/validate.py --case=2D_LDC --src_dir_Rie=data_valid/ldc2d_Riemann/ --src_dir_tvf=data_valid/ldc2d_tvf/ --src_dir_notvf=data_valid/ldc2d_notvf/
