<div align="center">

# JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework

[![Paper](http://img.shields.io/badge/paper-arxiv.2403.04750-B31B1B.svg)](https://arxiv.org/abs/2403.04750)
[![Docs](https://img.shields.io/readthedocs/jax-sph/latest)](https://jax-sph.readthedocs.io/en/latest/index.html)
[![PyPI - Version](https://img.shields.io/pypi/v/jax-sph)](https://pypi.org/project/jax-sph/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/tutorial.ipynb)
[![Discord](https://img.shields.io/badge/Discord-%235865F2?logo=discord&logoColor=white)](https://discord.gg/Ds8jRZ78hU)

[![Tests](https://github.com/tumaer/jax-sph/actions/workflows/tests.yml/badge.svg)](https://github.com/tumaer/jax-sph/actions/workflows/tests.yml)
[![CodeCov](https://codecov.io/gh/tumaer/jax-sph/graph/badge.svg?token=ULMGSY71R1)](https://codecov.io/gh/tumaer/jax-sph)
[![License](https://img.shields.io/pypi/l/jax-sph)](https://github.com/tumaer/jax-sph/blob/main/LICENSE)

![HT_T.gif](https://s9.gifyu.com/images/SUwUD.gif)

</div>

## Table of Contents

1. [**Installation**](#installation)
1. [**Getting Started**](#getting-started)
1. [**Setting up a case**](#setting-up-a-case)
1. [**Contributing**](#contributing)
1. [**Citation**](#citation)
1. [**Acknowledgements**](#acknowledgements)

## Installation

### Standalone library
Install the `jax-sph` library from PyPI as

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install jax-sph
```

By default `jax-sph` is installed without GPU support. If you have a CUDA-capable GPU, follow the instructions in the [GPU support](#gpu-support) section.

### Clone
We recommend using a `poetry` or `python3-venv` environment.

**Using Poetry**
```bash
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```
Later, you just need to `source .venv/bin/activate` to activate the environment.

**Using `python3-venv`**
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
pip install -U "jax[cuda12]==0.4.29"
```

## Getting Started

In the following, a quick setup guide for different cases is presented.

### Running an SPH Simulation
- Standard SPH 2D Taylor Green vortex
```bash
python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=0.0
 ```
- Transport velocity SPH 2D Taylor Green vortex
```bash
python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=1.0
 ```
- Riemann SPH 2D Taylor Green vortex
```bash
python main.py config=cases/tgv.yaml solver.name=RIE solver.tvf=0.0
 ```
-  Thermal diffusion
```bash
python main.py config=cases/ht.yaml
```

### Notebooks
We provide various notebooks demonstrating how to use JAX-SPH:
- [`tutorial.ipynb`](notebooks/tutorial.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/tutorial.ipynb), with a general overview of JAX-SPH and an example how to run the channel flow with hot bottom wall.
- [`iclr24_grads.ipynb`](notebooks/iclr24_grads.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/iclr24_grads.ipynb), with a validation of the gradients through the solver.
- [`iclr24_inverse.ipynb`](notebooks/iclr24_inverse.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/iclr24_inverse.ipynb), solving the inverse problem of finding the initial state of a 100-step-long SPH simulation.
- [`iclr24_sitl.ipynb`](notebooks/iclr24_sitl.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/iclr24_sitl.ipynb), including training and testing a Solver-in-the-Loop model using the [LagrangeBench](https://github.com/tumaer/lagrangebench) library.
- [`neighbors.ipynb`](notebooks/neighbors.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/neighbors.ipynb), explaining the difference between the three neighbor search implementations and comparing their performance.
- [`kernel_plots.ipynb`](notebooks/kernel_plots.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tumaer/jax-sph/blob/main/notebooks/kernel_plots.ipynb), visualizing the SPH kernels.

## Setting up a Case
To set up a case, just add a `my_case.py` and a `my_case.yaml` file to the `cases/` directory. Every *.py case should inherit from `SimulationSetup` in `jax_sph/case_setup.py` or another case, and every *.yaml config file should either contain a complete set of parameters (see `jax_sph/defaults.py`) or extend `JAX_SPH_DEFAULTS`. Running a case in relaxation mode `case.mode=rlx` overwrites certain parts of the selected case. Passed CLI arguments overwrite any argument.

## Contributing
If you wish to contribute, please run
```bash
pre-commit install
```

upon installation to automate the code linting and formatting checks.

## Citation

The main reference for this code is the ICLR'24 workshop paper `toshev2024jax`. If you refer to the code used for dataset generation in LagrangeBench, please cite `toshev2024lagrangebench` directly.

```bibtex
@article{toshev2024jax,
  title={JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework},
  author={Toshev, Artur P and Ramachandran, Harish and Erbesdobler, Jonas A and Galletti, Gianluca and Brandstetter, Johannes and Adams, Nikolaus A},
  journal={arXiv preprint arXiv:2403.04750},
  year={2024}
}
```

```bibtex
@article{toshev2024lagrangebench,
  title={Lagrangebench: A lagrangian fluid mechanics benchmarking suite},
  author={Toshev, Artur and Galletti, Gianluca and Fritz, Fabian and Adami, Stefan and Adams, Nikolaus},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

## Acknowledgements

The initial idea for JAX-SPH is due to Fabian Fritz and Ludger Paehler, which has led to the first validated JAX implementation of the 3D Taylor-Green vortex simulated with the transport velocity SPH formulation. Since then, Artur Toshev has maintained the codebase and extended it in various ways. The people who have provided useful feedback include, but are not limited to: Stefan Adami, Xiangyu Hu, Fabian Fritz, Christopher Zöller, Fabian Thiery, Johannes Brandstetter, and Ludger Paehler. Special thanks to Nikolaus Adams, who has supervised the project from the beginning.

### Contributors
- [Artur Toshev](https://github.com/arturtoshev) - developed and maintains the codebase; selected the SPH algorithms and validated most of them; designed the simulation cases and ML experiments.
- [Fabian Fritz](https://github.com/orgs/tumaer/people/fritzio) - provided the first validated transport velocity SPH implementation of the 3D Taylor-Green vortex in JAX.
- [Jonas Erbesdobler](https://github.com/JonasErbesdobler) - implemented Riemann SPH; improved and added solver validation scripts; contributed to refactoring the codebase.
- [Harish Ramachandran](https://github.com/harish6696) - implemented thermal diffusion and the inverse problem; helped in the initial phase of Solver-in-the-Loop.
- [Gianluca Galletti](https://github.com/gerkone) - validated the gradients through the solver; implemented and tuned Solver-in-the-Loop.
