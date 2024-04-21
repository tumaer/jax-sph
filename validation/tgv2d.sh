#!/bin/bash
# Generate validation data and validate 2D TGV
# with number of particles per direction nx = [20, 50, 100]
# Reference result from:
# "A Transport Velocty [...]", Adami 2012

# Generate data
for i in "0.02 25" "0.01 50" 
do
    a=( $i )
    dx=${a[0]}
    write_every=${a[1]}
    python main.py config=cases/tgv.yaml case.mode=rlx solver.tvf=1.0 case.dim=2 case.dx=$dx seed=123 case.r0_noise_factor=0.25 io.data_path=data_relaxed/ eos.p_bg_factor=0.02
    python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=1.0 case.dim=2 case.dx=$dx solver.t_end=5 io.write_every=$write_every case.state0_path=data_relaxed/tgv_2_${dx}_123.h5 io.data_path=data_valid/tgv2d_tvf/
    python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=0.0 case.dim=2 case.dx=$dx solver.t_end=5 io.write_every=$write_every case.state0_path=data_relaxed/tgv_2_${dx}_123.h5 io.data_path=data_valid/tgv2d_notvf/
    python main.py config=cases/tgv.yaml solver.name=RIE solver.tvf=0.0 case.dim=2 case.dx=$dx solver.t_end=5 io.write_every=$write_every case.state0_path=data_relaxed/tgv_2_${dx}_123.h5 io.data_path=data_valid/tgv2d_Rie/ solver.density_evolution=True
done

# Run validation script
CUDA_VISIBLE_DEVICES=0 python validation/validate.py --case=2D_TGV --src_dir=data_valid/