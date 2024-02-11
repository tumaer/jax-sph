# JAX-SPH: A Differentiable Smoothed Particle Hydrodynamics Framework

JAX-SPH is a differentiable weakly compressible SPH solver, that is currently under development at the Chair of Aerodynamics and Fluid Mechanics of the Technical University of Munich (TUM). The solver's framework utilizes Jax and depends on functions from the Jax-md library. The main references used for our solver are [1], [2] and [3]. 

The advantage of our solver is that you can combine all of the implemented SPH terms to your liking, or extract / solve only certain terms of the weakly compressible Navier-Stokes equations. Our solver includes the following SPH diskretizations:

- Standard SPH [1]
- Transport velocity formulation SPH [1]
- Riemann SPH [3]

Not only is it possible to combine, e.g. Riemann SPH with Transport velocity on top, but also to take the density evolution of Riemann SPH and pair it with Standard SPH for example.

## Installation

### Using Poetry (recommended)

```bash
poetry config virtualenvs.in-project true
poetry install
source .venv/bin/activate
```

Later, you just need to `source .venv/bin/activate` to activate the environment.

### Using Pip

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . # to install jax_sph in interactive mode
```

Later, you just need to `source venv/bin/activate` to activate the environment.

### GPU Support

If you want to use a CUDA GPU, you first need a running Nvidia driver. And then just follow the instruction [here](https://jax.readthedocs.io/en/latest/installation.html). The whole process could look like:

```bash
source .venv/bin/activate
pip install --upgrade "jax[cuda12_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## SPH Solver Overview

### Standard SPH
Standard SPH denotes the standard formulation, so to say the bare minimum of a SPH discretization, where the particel's position updates with their actual advection speed. While that being true, it does still include the possibility to use density summation, density evolution, the pressure term and the viscosity term of the momentum euation. However, as mentioned earlier, in our solver framework it is possible to interchange parts of the different solvers as you wish. For further informations, see [1][2].
### Transport Velocity SPH
The transport velocity term adds the so-called transport velocity that replaces the physical advection speed of the particles, leading to a less dissipative solver. Similar as before, every term of the weakyl compressible Navier-Stokes equations is included. However, in order to correct the physical velocity gradients, a correction term is added. On top, usually artifical velocity is added. For more details, see [1][2].
### Riemann SPH
Riemann SPH introduces a ondimensional Riemann problemm between every particle interaction counteracting the need of artificial viscosity. This leads to a vastly different formulation of the mass conservation's and the momentum equation's discretization. For a more detailed explanation and the derivation of the Riemann SPH discretization, see [3]. However, it is still possible to add every other term, e.g. transport velocity, on top of Riemann SPH.

## Getting Started
In the following a quick setup guide for differen cases is presented.


### Running a SPH Simulation
- Standard SPH 2D Taylor Green Vortex 
```bash
./venv/bin/python main.py --case=TGV --solver=STD --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data_valid/tgv2d_notvf/"
 ```

- Transport velocity formulation SPH 2D Taylor Green Vortex
```bash
./venv/bin/python main.py --case=TGV --tvf=1.0 --solver=STD --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data_valid/tgv2d_notvf/"
 ```
- Riemann SPH 2D Taylor Green Vortex
```bash
./venv/bin/python main.py --case=TGV --tvf=1.0 --solver=RIE --dim=2 --dx=0.02 --nxnynz=50_50_0 --t-end=5 --seed=123 --write-h5 --write-every=25 --data-path="data_valid/tgv2d_notvf/"
 ```
-  Thermal Diffusion
```bash
./venv/bin/python main.py --case=HT --solver=SPH --density-evolution --heat-conduction --dim=2 --dx=0.02 --t-end=1.5 --write-h5 --write-vtk --r0-noise-factor=0.05 --outlet-temperature-derivative --data-path="data_valid/therm_diff/"
```

### Solver in the Loop


### Inverse Problem

### Gradient Validation
In oder to see the differentiability of our solver, please take a look at the following notebook.\
[Gradient validation notebook](experiments/grads.ipynb)

## Development and Contribution

As stated in the introduction, our code base is still under development and will be expanded with new features in the future.
If you wish to contribute, please run

```bash
pre-commit install
```

upon istallation to automate the code linting and formatting checks. 

<!-- ## Citation
If you wish to use our code or parts of our code in your reasearch, please cite the solver using the following .bib,

```
@misc{jaxsph2024,
 author = {},
 booktitle = {},
 publisher = {},
 title = {},
 url = {},
 volume = {},
 year = {2024}
}
``` -->

## References

* [1] - "A generalized wall boundary condition for smoothed particle hydrodynamics", Adami, Hu & Adams, 2012
* [2] - "A transport-velocity formulation for smoothed particle hydrodynamics", Adami, Hu & Adams, 2013
* [3] - "A weakly compressible SPH method based on a low-dissipation Riemann solver", Zhang, Hu, Adams, 2017