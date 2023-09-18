#!/bin/bash

###### 2D TGV

# for seed in {0..199}
# do
#     echo "Run with seed = $seed"
#     ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.02 --nxnynz=50_50_0 --seed=$seed --write-h5 --write-every=100 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed  --p-bg-factor=0.0
#     ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=5 --seed=$seed --write-h5 --write-every=100 --data-path="datasets/2D_TGV_2500_10kevery100/"
# done
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_TGV_2500_10kevery100' --dst_dir='/home/atoshev/data/2D_TGV_2500_10kevery100' --split=2_1_1


###### 3D TGV

# Resolution investigation -> all the below are too poorly resolved for Re=100
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --seed=0 --write-h5 --write-every=10 --r0-noise-factor=0.01 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --t-end=10 --seed=0 --write-h5 --write-every=20 --data-path="datasets/3D_TGV_8000_100every20/" --write-vtk --dt=0.005
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.196349541 --nxnynz=32_32_32 --seed=0 --write-h5 --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.196349541 --nxnynz=32_32_32 --dt=0.002 --t-end=10 --seed=0 --write-h5 --write-every=10 --data-path="datasets/3D_TGV_32768_100every20/"  --write-vtk
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.157079633 --nxnynz=40_40_40 --seed=0 --write-h5 --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.01
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.157079633 --nxnynz=40_40_40 --dt=0.002 --t-end=10 --seed=999 --write-h5 --write-every=10 --data-path="datasets/3D_TGV_64000_100every10/"  --write-vtk

# explore Reynolds numbers comparing agains jax-fluids results
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --seed=123 --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.01
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --t-end=10 --dt=0.005 --seed=123 --write-h5 --write-vtk --write-every=2 --data-path="datasets/3D_TGV_8000_100every20/" --viscosity=0.1
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --t-end=10 --dt=0.005 --seed=123 --write-h5 --write-vtk --write-every=2 --data-path="datasets/3D_TGV_8000_100every20/" --viscosity=0.05
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --t-end=10 --dt=0.005 --seed=123 --write-h5 --write-vtk --write-every=2 --data-path="datasets/3D_TGV_8000_100every20/" --viscosity=0.02
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --t-end=10 --dt=0.005 --seed=123 --write-h5 --write-vtk --write-every=2 --data-path="datasets/3D_TGV_8000_100every20/" --viscosity=0.01

# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.157079633 --nxnynz=40_40_40 --seed=999 --write-h5 --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.01
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=1.0 --dim=3 --dx=0.157079633 --nxnynz=40_40_40 --dt=0.0025 --t-end=10 --seed=999 --write-h5 --write-every=4 --data-path="datasets/3D_TGV_64000_100every10/" --write-vtk --viscosity=0.02
# ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.157079633 --nxnynz=40_40_40 --dt=0.0025 --t-end=10 --seed=999 --write-h5 --write-every=4 --data-path="datasets/3D_TGV_64000_100every10/" --write-vtk --viscosity=0.02

# generate 3D dataset
for seed in {0..399}
do
    echo "Run with seed = $seed"
    ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.314159265 --nxnynz=20_20_20 --seed=$seed --write-h5 --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.01
    ./venv/bin/python main.py --case=TGV --solver=SPH --tvf=0.0 --dim=3 --dx=0.314159265 --dt=0.005 --t-end=30 --seed=$seed --write-h5 --write-every=100 --data-path="datasets/3D_TGV_8000_10kevery100" --viscosity=0.02
done
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/3D_TGV_8000_10kevery100' --dst_dir='/home/atoshev/data/3D_TGV_8000_10kevery100' --split=2_1_1
