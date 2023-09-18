#!/bin/bash

# Very small 3D TGV dataset for debugging.
# 5 trajectories (3+1+1), each with 125 particles and 21 steps

for seed in {0..3}
do
    echo "Run with seed = $seed"
    ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.125 --nxnynz=8_8_8 --seed=$seed --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed
    ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=1.0 --dim=3 --dx=0.125 --dt=0.002 --t-end=0.04001 --seed=$seed --write-h5 --write-vtk --write-every=1 --data-path="dataset_tgv_debug/"
done
