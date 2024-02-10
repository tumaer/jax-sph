# SPH Dataset using JAX-MD

SPH solver in JAX. The main references are [1] and [2].
## TODOs

**Artur**
- [x] write trajectory to 1) `.h5` and 2) `.vtk` files to visualize with ParaView
- [x] add tags to main.py
- [x] implement vanilla SPH integrator as in [1]
- [x] implement LDC in 2D - based on [1]
- [x] implement LDC in 3D
- [x] dt condition based on external force
- [x] implement Poiseuille flow
- [ ] implement cube of water (CW) flying in random directions and hitting a wall (same simulation as DM)
- [x] check why the system size does not scale to >20k particles in 3D dam break! -> allocating space for neighbors computation
- [ ] result validation:
    - [ ] TGV - analytical solution
        - [x] kinetic energy over 6 sec.
        - [x] velocity
        - [x] oscillations in the beginning?
		- [ ] force to get a driven TGV? To avoid oscillations and
    - [x] PF - analytical solution for velocity in x
    - [x] LDC - reference solution - see [2]
- [x] compare u vs v in transport velocity formulation
- [x] Dam break:
	- [x] implement artificial viscosity
	- [x] configure case as in generalized BC paper
- [x] How is artificial viscosity different than just normal one? -> artif. visc is applied to only among water particles. Fow wall viscosity, use normal viscosity.
- [x] run relaxation simulations to obtain multiple initial conditions for
	- [x] TGV
	- [x] Dam Break
- [ ] extract dt computation and apply it at every iteration step as a sanity check! Especially relevant for RPF and other driven flows / flows with external force.
- [ ] neighbors backends
	- [x] jax-md scan
	- [x] matscipy
	- [ ] validate these implementations
	- [ ] time these implementations

## Notes

**Dam break**
- [x] How do you initialize the particles? With noise?
- [x] Particles jump up in the beginning? -> use cartesian grid initialization. Alternatively, start a simulation from the positions of previous runs. Remove random noise! It introduces acoustic waves in the beginning!
- [x] Inviscid dam break means either 1) free-slip at the wall + physical viscosity or artificial viscosity + no physical viscosity. If inviscid flow, then artificial viscosity is equivalent to setting physical one, except for wall interactions.
- [x] role of exponent gamma in EoS?
- [ ] Dam break particles fly away from the wall in the front of the wave.



### Reference data
'Case: name of file'

- TGV: tgv_analytical_sol.py
	- it has the analytic velocities u and v as per Adami2012
	- the array 'y_axes' has the u_max values
- LDC: ldc_data.py
	- for three Reynolds Numbers: 100, 1000, 10000, reference data is available
	- u_Re_100, u_Re_1000, u_Re_10000 and v_Re_100, v_Re_1000, v_Re_10000 are the respective u and v velocities for each of the Re values
	- requires 2 csv files: ldc_data_u_vel.csv and ldc_data_v_vel.csv (obtained the tabular data from Ghia et al (https://www.sciencedirect.com/science/article/pii/0021999182900584))
- RPF: poiseuille_data.py
	- got digitized graph of analytic solution at t = 2, 10, 20 and 100 seconds
	- requires 4 csv files: rpf_t_2.csv, rpf_t_10.csv, rpf_t_20.csv, rpf_t_100.csv
- Dambreak: dambreak_data.py
	- Obtained the reference data for time evolution of front and height from Martin and Moyce (https://royalsocietypublishing.org/doi/10.1098/rsta.1952.0006)
	- Requires 1 csv file: Buchner_pressureProfile.csv, it is for comparing temporal pressure profile as per Fig 15 in Adami2012


### Final goal

This is the TODO list concerning different cases. Everything has to be in 3D; use 2D only for debugging.

- [x] TGV:
	- [x] increase Reynolds number to turbulence
	- [x] Vanilla SPH
	- [x] Transport velocity
- [x] Reverse Poiseuille Flow
	- [x] Vanilla SPH
	- [x] Transport velocity
- [x] LDC:
	- [x] Vanilla SPH
	- [x] Transport velocity - for dataset, run 10 Reynolds numbers between 100-1000. Time to stationary solution ~20s
- [ ] Dab Break:
	- [x] Vanilla SPH
	- [ ] Transport velocity
- [ ] Rayleigh-Taylor instability
	- [ ] Vanilla SPH
- [ ] Drop in shear flow
	- [ ] Vanilla SPH


## SPH Algorithm

1. shift velocity (dt)
2. shift transport velocity (0.5 dt)
3. if free surface -> shift density (dt) # TODO: for now this is in the model
4. shift position (dt)
5. update cell list
6. (model) if not free surface -> density summation
7. (model) compute primitives
8. (model) apply wall boundary conditions
9. (model) update RHS
10. (model) if free surface -> density correction

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
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
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
