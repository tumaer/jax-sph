#!/bin/bash

####### 2D Rayleigh-Taylor Instability
python main.py config=cases/rti.yaml case.viscosity=0.0 solver.name=SPH solver.dt=0.0002 solver.artificial_alpha=0.0 solver.tvf=1.0

# Run validation script

