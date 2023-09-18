#!/bin/bash

##### 2D dataset

# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.02 --nxnynz=50_50_0 --seed=0 --write-h5 --write-every=10 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=850 --seed=0 --write-h5 --write-every=100 --data-path="datasets/2D_LDC_2500_10kevery100/" --p-bg-factor=0.01
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery100' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery100' --split=2_1_1 --skip_first_n_frames=1248

# s = v * dt
# s = 2*dx = 2 * 1/50 = 0.04
# v = 1
# dt = 0.04
# num_samples = 20k (10 train, 5 val, 5 test)
# t_end = num_samples * dt = 20k * 0.04 = 800 (+ 50 to develop)
# simulator_dt = 0.0004
# write_every_n = dt / simulator_dt = 100

# writing every 10th step to explore training on 10th/20th/50th/100th step
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --dt=0.0004 --t-end=850 --seed=0 --write-h5 --write-every=10 --data-path="datasets/2D_LDC_2500_10kevery10/" --p-bg-factor=0.01
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery10' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=1
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery20' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=2
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery30' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=3
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery40' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=4
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery50' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=5
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_LDC_2500_10kevery10' --dst_dir='/home/atoshev/data/2D_LDC_2500_10kevery100' --split=2_1_1 --skip_first_n_frames=12480 --slice_every_nth_frame=10

# CUDA_VISIBLE_DEVICES=7 nohup ./scripts/dataset_ldc.sh 2>&1 &

##### 3D dataset

# resolution exploration
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.05 --nxnynz=20_20_10 --seed=0 --write-h5 --write-vtk --write-every=100 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.041666667 --nxnynz=24_24_12 --seed=0 --write-h5 --write-vtk --write-every=100 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.025 --nxnynz=40_40_20 --seed=0 --write-h5 --write-vtk --write-every=100 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --dt=0.001 --t-end=20 --seed=0 --write-h5 --write-vtk --write-every=10 --data-path="datasets/3D_LDC_4000_10kevery100" --p-bg-factor=0.01 # ekin = 0.0140 , without pb:0.0142
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=3 --dx=0.041666667 --dt=0.0009 --t-end=50 --seed=0 --write-h5 --write-vtk --write-every=100 --data-path="datasets/3D_LDC_8112_10kevery100" --p-bg-factor=0.01 # ekin = 0.0146
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=3 --dx=0.025 --dt=0.0005 --t-end=50 --seed=0 --write-h5 --write-vtk --write-every=10 --data-path="datasets/3D_LDC_32000_10kevery100" --p-bg-factor=0.01 # ekin = 0.0156

# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.041666667 --nxnynz=24_24_12 --seed=0 --write-h5 --write-every=100 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
# ./venv/bin/python main.py --case=LDC --solver=SPH --tvf=0.0 --dim=3 --dx=0.041666667 --dt=0.0009 --t-end=1850 --seed=0 --write-h5 --write-every=100 --data-path="datasets/3D_LDC_8160_10kevery100" --p-bg-factor=0.01
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/3D_LDC_8160_10kevery100' --dst_dir='/home/atoshev/data/3D_LDC_8160_10kevery100' --split=2_1_1 --skip_first_n_frames=555

# s = v * dt
# s = 2*dx = 2 * 1/24 = 1/12
# v = 1
# dt = 0.09
# num_samples = 20k (10k train, 5k val, 5k test)
# t_end = num_samples * dt = 20k * 0.09 = 1800 (+ 50 to develop)
# simulator_dt = 0.0009
# write_every_n = dt / simulator_dt = 100
