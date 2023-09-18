#!/bin/bash

# Unit test dataset
# Consists of 8x8x8 particles between [0.25 ... 0.75] in a [0, 1]^3 box

for seed in {0..3}
do
    echo "Run with seed = $seed"
    ./venv/bin/python main.py --case=UT --solver=UT --dim=3 --dx=0.0625 --dt=0.001 --t-end=1.0 --seed=$seed --write-h5 --write-vtk --write-every=50 --data-path="./dataset_ut/"
done
