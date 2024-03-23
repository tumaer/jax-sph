#!/bin/bash
# Generate validation data and validate 2D TGV
# with number of particles per direction nx = [20, 50, 100]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
for i in "0.02 25" "0.01 50" 
do
    a=( $i )
    dx="${a[0]}"
    write_every="${a[1]}"
    python main.py --case=TGV --mode=rlx --solver=SPH --tvf=1.0 --dim=2 --dx=$dx --seed=123 --write-h5 --write-every=10 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.02
    python main.py --case=TGV --solver=SPH --tvf=1.0 --dim=2 --dx=$dx --t-end=5 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv2d_tvf/"  --r0-type=relaxed
    python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=2 --dx=$dx --t-end=5 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv2d_notvf/" --r0-type=relaxed
    python main.py --case=TGV --solver=RIE --dim=2 --dx=$dx --t-end=5 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv2d_Rie/" --density-evolution --eta-limiter=3 --r0-type=relaxed
done

# Run validation script
CUDA_VISIBLE_DEVICES=0 python validation/validate.py --case=2D_TGV --src_dir="data_valid/"