#!/bin/bash
# run this script with:
# CUDA_VISIBLE_DEVICES=7 nohup ./scripts/dataset_rpf.sh 2>&1 &

###### 2D RPF
# explore resolution
# ./venv/bin/python main.py --case=Rlx --solver=SPH --viscosity=0.1 --tvf=1.0 --dim=2 --dx=0.05 --nxnynz=20_40_0 --seed=0 --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.1 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.05 --nxnynz=20_40_0 --seed=0 --write-h5 --write-vtk --write-every=100 --t-end=100 --data-path="datasets/2D_RPF_800_100kevery10/" --dt=0.001
# ./venv/bin/python main.py --case=Rlx --solver=SPH --viscosity=0.1 --tvf=1.0 --dim=2 --dx=0.03125 --nxnynz=32_64_0 --seed=0 --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.1 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.03125 --nxnynz=32_64_0 --seed=0 --write-h5 --write-vtk --write-every=200 --t-end=100 --data-path="datasets/2D_RPF_2048_100kevery10/" --dt=0.0005
# ./venv/bin/python main.py --case=Rlx --solver=SPH --viscosity=0.1 --tvf=1.0 --dim=2 --dx=0.025 --nxnynz=40_80_0 --seed=0 --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.1 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.025 --nxnynz=40_80_0 --seed=0 --write-h5 --write-vtk --write-every=200 --t-end=100 --data-path="datasets/2D_RPF_3200_100kevery10/" --dt=0.0005
# ./venv/bin/python main.py --case=Rlx --solver=SPH --viscosity=0.1 --tvf=1.0 --dim=2 --dx=0.0125 --nxnynz=80_160_0 --seed=0 --write-h5 --write-vtk --write-every=10 --r0-noise-factor=0.1 --relax-pbc --data-path=data_relaxed
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.0125 --nxnynz=80_160_0 --seed=0 --write-h5 --write-vtk --write-every=400 --t-end=100 --data-path="datasets/2D_RPF_12800_100kevery10/" --dt=0.00025

# # Long dataset for scaling w.r.t. data size
# ./venv/bin/python main.py --case=Rlx --solver=SPH --viscosity=0.1 --tvf=1.0 --dim=2 --dx=0.025 --nxnynz=40_80_0 --seed=123 --write-h5 --write-every=10 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.05
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.025 --nxnynz=40_80_0 --seed=123 --write-h5 --write-every=100 --t-end=2050 --data-path="datasets/2D_RPF_3200_20kevery100/" --dt=0.0005
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_20kevery100' --dst_dir='/home/atoshev/data/2D_RPF_3200_20kevery100' --split=2_1_1 --skip_first_n_frames=998

# writing every 10th step to explore training on 10th/20th/50th/100th step
# ./venv/bin/python main.py --case=RPF --solver=SPH --viscosity=0.1 --tvf=0.0 --dim=2 --dx=0.025 --nxnynz=40_80_0 --seed=123 --write-h5 --write-every=10 --t-end=1050 --data-path="datasets/2D_RPF_3200_10kevery10/" --dt=0.0005
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery10' --split=2_1_1 --skip_first_n_frames=9980
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery20' --split=2_1_1 --skip_first_n_frames=9980 --slice_every_nth_frame=2
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery30' --split=2_1_1 --skip_first_n_frames=9980 --slice_every_nth_frame=3
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery40' --split=2_1_1 --skip_first_n_frames=9980 --slice_every_nth_frame=4
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery50' --split=2_1_1 --skip_first_n_frames=9980 --slice_every_nth_frame=5
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_RPF_3200_10kevery10' --dst_dir='/home/atoshev/data/2D_RPF_3200_10kevery100' --split=2_1_1 --skip_first_n_frames=9980 --slice_every_nth_frame=10


# ###### 3D RPF
# explore resolution
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.025 --nxnynz=40_80_20 --seed=123 --write-h5 --write-every=100 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.05
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-every=50 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.05
# for i in "0.1 1.0" "0.05 0.5"
# do
#     a=( $i )
#     viscosity="${a[0]}"
#     force="${a[1]}"
#     ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.025 --nxnynz=40_80_20 --seed=123 --write-vtk --write-every=100 --dt=0.0005 --t-end=50 --data-path="data_valid/pf3d_notvf" --g-ext-magnitude=$force --viscosity=$viscosity
#     ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-vtk --write-every=50 --dt=0.001 --t-end=50 --data-path="data_valid/pf3d_notvf" --g-ext-magnitude=$force --viscosity=$viscosity
# done
# CUDA_VISIBLE_DEVICES=7 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.025 --nxnynz=40_80_20 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.0005 --t-end=50 --data-path="data_valid/pf3d_notvf" --g-ext-magnitude=0.5 --viscosity=0.05 > fine_res.out 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=50 --dt=0.001 --t-end=50 --data-path="data_valid/pf3d_notvf" --g-ext-magnitude=0.5 --viscosity=0.05 > coarse_res.out 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.001 --t-end=2050 --data-path="data_valid/pf3d_notvf_more_pb" --g-ext-magnitude=1.0 --viscosity=0.1 --p-bg-factor=0.1 > more_pb.out 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.001 --t-end=2050 --data-path="data_valid/pf3d_notvf_no_pb" --g-ext-magnitude=1.0 --viscosity=0.1 --p-bg-factor=0.0 > no_pb.out 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.001 --t-end=2050 --data-path="data_valid/pf3d_notvf_less_pb" --g-ext-magnitude=1.0 --viscosity=0.1 --p-bg-factor=0.02 > less_pb.out 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.001 --t-end=2050 --data-path="data_valid/pf3d_notvf_high_visc" --g-ext-magnitude=2.0 --viscosity=0.2 --p-bg-factor=0.02 > high_visc.out 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-vtk --write-every=100 --dt=0.001 --t-end=2050 --data-path="data_valid/pf3d_notvf_low_visc" --g-ext-magnitude=0.5 --viscosity=0.05 --p-bg-factor=0.02 > low_visc.out 2>&1 &

# used to generate 3D dataset
# ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-every=50 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.05
# ./venv/bin/python main.py --case=RPF --solver=SPH --tvf=0.0 --dim=3 --dx=0.05 --nxnynz=20_40_10 --seed=123 --write-h5 --write-every=100 --dt=0.001 --t-end=2050 --data-path="datasets/3D_RPF_8000_10kevery100" --viscosity=0.1 --p-bg-factor=0.02
# ./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/3D_RPF_8000_10kevery100' --dst_dir='/home/atoshev/data/3D_RPF_8000_10kevery100' --split=2_1_1 --skip_first_n_frames=498

# s=v * t
# dx = 0.05
# dt_sim = 0.001
# target: s = 2*dx = 0.1
# vmax = 1.0
# target dt = s/vmax = 0.1/1.0 = 0.1
# subsample_n = dt_target/dt_sim = 0.1/0.001 = 100
# target num samples = 20000
# target time horizon: 20000 * 0.1 = 2000 (+50 for equilibriation)
