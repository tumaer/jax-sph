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
    dx=${a[0]}
    write_every=${a[1]}
    python main.py config=cases/tgv.yaml case.mode=rlx solver.tvf=1.0 case.dim=3 case.dx=$dx seed=123 case.r0_noise_factor=0.25 io.data_path=data_relaxed/ eos.p_bg_factor=0.02
    python main.py config=cases/tgv.yaml solver.name=SPH case.viscosity=0.02 solver.tvf=1.0 case.dim=3 case.dx=$dx solver.t_end=10 io.write_every=$write_every case.state0_path=data_relaxed/tgv_3_${dx}_123.h5 io.data_path=data_valid/tgv3d_tvf/ case.r0_type=cartesian
    python main.py config=cases/tgv.yaml solver.name=SPH case.viscosity=0.02 solver.tvf=0.0 case.dim=3 case.dx=$dx solver.t_end=10 io.write_every=$write_every case.state0_path=data_relaxed/tgv_3_${dx}_123.h5 io.data_path=data_valid/tgv3d_notvf/ case.r0_type=cartesian
    python main.py config=cases/tgv.yaml solver.name=RIE case.viscosity=0.02 solver.tvf=0.0 case.dim=3 case.dx=$dx solver.t_end=10 io.write_every=$write_every case.state0_path=data_relaxed/tgv_3_${dx}_123.h5 io.data_path=data_valid/tgv3d_Rie/ solver.density_evolution=True case.r0_type=cartesian
done

# Run validation script
python validation/validate.py --case=3D_TGV --src_dir=data_valid/

