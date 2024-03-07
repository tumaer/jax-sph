# JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework

JAX-SPH [(Toshev et al., 2024)](https://openreview.net/forum?id=8X5PXVmsHW) is a modular JAX-based weakly compressible SPH framework, which implements the following SPH routines:
- Standard SPH [(Adami et al., 2012)](https://www.sciencedirect.com/science/article/pii/S002199911200229X)
- Transport velocity SPH [(Adami et al., 2013)](https://www.sciencedirect.com/science/article/pii/S002199911300096X)
- Riemann SPH [(Zhang et al., 2017)](https://www.sciencedirect.com/science/article/abs/pii/S0021999117300438)

## Installation
Currently, the code can only be installed by cloning this repository. We recommend using a Poetry or `python3-venv` environment.

### Using Poetry (recommended)
```bash
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```
Later, you just need to `source .venv/bin/activate` to activate the environment.

### Using `python3-venv`
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . # to install jax_sph in interactive mode
```

Later, you just need to `source venv/bin/activate` to activate the environment.

### GPU Support
If you want to use a CUDA GPU, you first need a running Nvidia driver. And then just follow the instructions [here](https://jax.readthedocs.io/en/latest/installation.html). The whole process could look like this:
```bash
source .venv/bin/activate
pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Getting Started

In the following, a quick setup guide for different cases is presented.

### Running an SPH Simulation
- Standard SPH 2D Taylor Green vortex
```bash
python main.py --case=TGV --solver=SPH --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data/tgv2d_notvf/"
 ```
- Transport velocity SPH 2D Taylor Green vortex
```bash
python main.py --case=TGV --tvf=1.0 --solver=SPH --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data/tgv2d_notvf/"
 ```
- Riemann SPH 2D Taylor Green vortex
```bash
python main.py --case=TGV --tvf=1.0 --solver=RIE --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data/tgv2d_notvf/"
 ```
-  Thermal diffusion
```bash
python main.py --case=HT --solver=SPH --density-evolution --heat-conduction --dim=2 --dx=0.02 --t-end=1.5 --write-h5 --write-vtk --r0-noise-factor=0.05 --outlet-temperature-derivative --data-path="data/therm_diff/"
```

### Solver-in-the-Loop
To train and test our Solver-in-the-Loop model, run the script in [./experiments/sitl.py](./experiments/sitl.py). This file relies on [LagrangeBench](https://github.com/tumaer/lagrangebench), which can be installed by `pip install lagrangebench`. For more information on the training and inference setup, visit the LagrangeBench website.

### Inverse Problem
The presented inverse problem of finding the initial state of a 100-step-long SPH simulation can be fully reproduced using the notebook [./experiments/inverse.ipynb](./experiments/inverse.ipynb).

### Gradient Validation
The presented validation of the gradients through the solver can be fully reproduced using the notebook [./experiments/grads.ipynb](./experiments/grads.ipynb)

## Development and Contribution
If you wish to contribute, please run
```bash
pre-commit install
```

upon installation to automate the code linting and formatting checks.

## Citation

The main reference for this code is `toshev2024jaxsph`. If you refer to the code used for dataset generation in LagrangeBench, please cite `toshev2023lagrangebench` directly.

```bibtex
@inproceedings{toshev2024jaxsph,
title      = {JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework},
author     = {Artur Toshev and Harish Ramachandran and Jonas A. Erbesdobler and Gianluca Galletti and Johannes Brandstetter and Nikolaus A. Adams},
booktitle  = {ICLR 2024 Workshop on AI4DifferentialEquations In Science},
year       = {2024},
url        = {https://openreview.net/forum?id=8X5PXVmsHW}
}
```
```bibtex
@inproceedings{toshev2023lagrangebench,
title      = {LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite},
author     = {Artur P. Toshev and Gianluca Galletti and Fabian Fritz and Stefan Adami and Nikolaus A. Adams},
year       = {2023},
url        = {https://arxiv.org/abs/2309.16342},
booktitle  = {37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks},
}
```
