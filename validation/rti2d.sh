#!/bin/bash

####### 2D Rayleigh-Taylor Instability
python main.py config=cases/rti.yaml solver.name=SPH solver.tvf=1.0 eos.p_bg_factor=0.02 solver.density_evolution=True io.data_path=data_valid/rti2d/

# Run validation script
CUDA_VISIBLE_DEVICES=0 python validation/validate.py --case=2D_RTI --src_dir=data_valid/rti2d/
