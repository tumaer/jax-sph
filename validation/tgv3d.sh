#!/bin/bash
# Generate validation data and validate 3D TGV
# with number of particles per direction nx = [20, 40, 64]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
# for i in "0.314159265 20_20_20 10" # "0.157079633 40_40_40 20" "0.09817477 64_64_64 32" # "0.125663706 50_50_50 25"
for i in "0.314159265 20_20_20 14" "0.157079633 40_40_40 28"
do
    a=( $i )
    dx="${a[0]}"
    nxnynz="${a[1]}"
    write_every="${a[2]}"
    # ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=$dx --nxnynz=$nxnynz --seed=123 --write-h5 --write-vtk --write-every=1 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.02 --nl-backend=jaxmd_scan
    ./venv/bin/python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=1.0 --dim=3 --dx=$dx --nxnynz=$nxnynz --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_tvf"
    # ./venv/bin/python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --nxnynz=$nxnynz --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_notvf"
    # ./venv/bin/python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --nxnynz=$nxnynz --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_notvf" --nl-backend=jaxmd_scan
    # ./venv/bin/python main.py --case=TGV --solver=SPH --viscosity=0.02 --tvf=0.0 --dim=3 --dx=$dx --nxnynz=$nxnynz --t-end=10 --seed=123 --write-h5 --write-every=$write_every --data-path="data_valid/tgv3d_notvf" --nl-backend=matscipy
done

# Run validation script
./venv/bin/python validation/validate.py --case=3D_TGV --src_dir="data_valid/tgv3d_tvf"
# ./venv/bin/python validation/validate.py --case=3D_TGV --src_dir="data_valid/tgv3d_notvf"
