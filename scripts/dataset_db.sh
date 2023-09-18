#!/bin/bash
# CUDA_VISIBLE_DEVICES=6 nohup ./scripts/dataset_db.sh >> db_dataset.out 2>&1 &

##### 2D dataset

for seed in {0..114}
do
    echo "Run with seed = $seed"
    ./venv/bin/python main.py --case=Rlx --solver=SPH --tvf=1.0 --dim=2 --dx=0.02 --nxnynz=100_50_0 --seed=$seed --write-h5 --write-every=100 --r0-noise-factor=0.25 --data-path=data_relaxed --p-bg-factor=0.0
    ./venv/bin/python main.py --case=DB --solver=SPH --tvf=0.0 --dim=2 --dx=0.02 --t-end=12 --dt=0.0003 --viscosity=0.00005 --write-every=100 --write-h5 --seed=$seed --data-path=datasets/2D_DB_5740_20kevery100 --density-evolution --artificial-alpha=0.1
done
./venv/bin/python scripts/dataset_gen.py --src_dir='datasets/2D_DB_5740_20kevery100' --dst_dir='/home/atoshev/data/2D_DB_5740_20kevery100' --split=2_1_1
# 15 blowing up runs were removed from the dataset
# use `count_files_db.py` to detect defect runs

# 100x50=5000 water particles
# 106x274 outer box, i.e. 106x274 - 100x268 = 2244 wall particles
# with only one wall layer, wall particles are 2*(100+270) = 740

# t_end = 12
# dt_sample = 0.0003*100 = 0.03
# num_samples = 12/0.03 = 400
# for 20k training+val+test samples, we need 20k/400 = 50 seeds
# because the dataset is complex, we run 100 seeds
