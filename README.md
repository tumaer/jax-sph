# SPH Dataset using JAX-MD

SPH solver in JAX. The main references are [1] and [2].

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

## Usage and Contribution

See the example scripts in `scripts/` on how to use the code.

If you want to contribute, you should run

```bash
pre-commit install
```

upon istallation to automate the code linting and formatting checks.

## References

* [1] - "A generalized wall boundary condition for smoothed particle hydrodynamics", Adami, Hu & Adams, 2012
* [2] - "A transport-velocity formulation for smoothed particle hydrodynamics", Adami, Hu & Adams, 2013
