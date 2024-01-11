#!/bin/bash
# Generate validation data and validate 2D TGV
# with number of particles per direction nx = [20, 50, 100]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
#"0.05 20_20_0 10" "0.02 50_50_0 25" "0.01 100_100_0 50"


#./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.05 --nxnynz=20_20_0 --seed=123 --write-h5 --write-every=1 --r0-noise-factor=0.25 --relax-pbc --data-path=data_relaxed --p-bg-factor=0.02 --gpu=0
#./venv/bin/python main.py --case=TGV --solver=SPH --tvf=1.0 --dim=2 --dx=0.05 --nxnynz=20_20_0 --t-end=10 --seed=123 --write-h5 --write-every=10 --data-path="data_Riemann/tgv2d_tvf/" --gpu=0
#./venv/bin/python main.py --case=TGV --solver=RIE --dim=2 --dx=0.01 --nxnynz=100_100_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=50 --data-path="data_Riemann/tgv2d/" --gpu=0
#./venv/bin/python main.py --case=TGV --solver=RIE --dim=2 --dx=0.01 --nxnynz=100_100_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=50 --data-path="data_Riemann/tgv2d/" --gpu=0
#./venv/bin/python main.py --case=TGV --solver=RIE --dim=2 --dx=0.005 --nxnynz=200_200_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=100 --data-path="data_Riemann/tgv2d/" --gpu=0
#./venv/bin/python main.py --case=TGV --solver=RIE --dim=2 --dx=0.01 --nxnynz=100_100_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=100 --data-path="data_Riemann/tgv2d_lim/" --gpu=0 --is-limiter --eta-limiter=3


./venv/bin/python main.py --case=TGV --solver=RIE2 --dim=2 --dx=0.05 --nxnynz=20_20_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=50 --data-path="data_Riemann/tgv2d/" --gpu=0
#./venv/bin/python main.py --case=TGV --solver=RIE2 --density-evolution --dim=2 --dx=0.01 --nxnynz=100_100_0 --Vmax=1.0 --t-end=10 --seed=123 --write-h5 --write-every=50 --data-path="data_Riemann/tgv2d_lim/" --gpu=0 --is-limiter --eta-limiter=3


CUDA_VISIBLE_DEVICES=0 ./venv/bin/python riemann_plot/riemann_tgv_plot.py --case=2D_TGV --src_dir="data_Riemann/tgv2d/"
#CUDA_VISIBLE_DEVICES=0 ./venv/bin/python riemann_plot/riemann_tgv_plot.py --case=2D_TGV --src_dir="data_Riemann/tgv2d_lim/"