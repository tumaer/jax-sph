#!/bin/bash
# Generate validation data and validate 3D TGV
# with number of particles per direction nx = [20, 40, 64]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
# "0.157079633 40_40_40 20" "0.09817477 64_64_64 32" # "0.125663706 50_50_50 25"
for i in "0.314159265 10" "0.19634954 16" "0.125663706 25"  #"0.125663706 50_50_50 25" #"0.314159265 20_20_20 14" "0.157079633 40_40_40 28"
do
    a=( $i )
    dx="${a[0]}"
    write_every="${a[1]}"
    python main.py --case=TGV --mode=rlx --solver=SPH --tvf=1.0 --dim=3 --dx=$dx --seed=123 --write-h5 --write-vtk --write-every=1 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.02 --nl-backend=jaxmd_vmap
    python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=1.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-vtk --write-every=$write_every --data-path="data_valid/tgv3d_tvf" --gpu=0 
    python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-vtk --write-every=$write_every --data-path="data_valid/tgv3d_notvf" --gpu=0 
    python main.py --case=TGV --solver=RIE --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --t-end=10 --seed=123 --write-h5 --write-vtk --write-every=$write_every --data-path="data_valid/tgv3d_Rie" --gpu=0 --density-evolution --eta-limiter=3

done

# Run validation script
python validation/validate.py --case=3D_TGV --src_dir="data_valid/"

