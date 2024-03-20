#!/bin/bash
# Generate validation data and validate 3D TGV
# with number of particles per direction nx = [20, 40, 64]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
# nx = 20: dx = 0.314159265
# nx = 32: dx = 0.19634954
# nx = 40: dx = 0.157079633
# nx = 50: dx = 0.125663706
# nx = 64: dx = 0.09817477

for i in "0.314159265 10" "0.19634954 16" "0.125663706 25"
do
    a=( $i )
    dx="${a[0]}"
    write_every="${a[1]}"
    python main.py --case=TGV --mode=rlx --solver=SPH --tvf=1.0 --dim=3 --dx=$dx --seed=123 --write-h5 --write-every=1 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.02 --nl-backend=jaxmd_vmap
    python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=1.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_tvf" --r0-type=relaxed
    python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_notvf" --r0-type=relaxed
    python main.py --case=TGV --solver=RIE --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_Rie" --density-evolution --eta-limiter=3 --r0-type=relaxed

done

# Run validation script
python validation/validate.py --case=3D_TGV --src_dir="data_valid/"

