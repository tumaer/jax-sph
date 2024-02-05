#!/bin/bash
# Generate validation data and validate 2D TGV
# with number of particles per direction nx = [20, 50, 100]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012


for i in "0.05 20_20_0 10" "0.02 50_50_0 25" "0.01 100_100_0 50" 
do
    a=( $i )
    dx="${a[0]}"
    nxnynz="${a[1]}"
    write_every="${a[2]}"
    ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=$dx --nxnynz=$nxnynz --seed=123 --write-h5 --write-every=1000 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.02 --gpu=0
done