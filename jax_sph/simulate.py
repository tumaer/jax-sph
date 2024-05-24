"""Core simulation loop."""

import os
import time

import numpy as np
from jax import jit
from jax_md.partition import Sparse
from omegaconf import DictConfig, OmegaConf

from cases import select_case
from jax_sph import partition
from jax_sph.case_setup import set_relaxation
from jax_sph.integrator import si_euler
from jax_sph.io_state import io_setup, write_state
from jax_sph.solver import WCSPH
from jax_sph.utils import Tag, get_ekin, get_val_max


def simulate(cfg: DictConfig):
    """Core simulation loop.

    This function is the main entry point for the simulation and does the following:

    1. Initialize the 1) case, 2) solver, 3) neighbor list, 4) integrator.
    2. Run the simulation loop and optionally write states.
    """

    # Case setup
    # get the config file name, e.g. "cases/cw.py" -> "cw"
    cfg.case.name = os.path.splitext(os.path.basename(cfg.config))[0]
    Case = select_case(cfg.case.name)
    # if in relaxation mode, wrap the case with the relaxation case
    if cfg.case.mode == "rlx":
        case = set_relaxation(Case, cfg)
    elif cfg.case.mode == "sim":
        case = Case(cfg)
    (
        cfg,
        box_size,
        state,
        g_ext_fn,
        bc_fn,
        eos_fn,
        key,
        displacement_fn,
        shift_fn,
    ) = case.initialize()

    # Solver setup
    solver = WCSPH(
        displacement_fn,
        eos_fn,
        g_ext_fn,
        cfg.case.dx,
        cfg.case.dim,
        cfg.solver.dt,
        cfg.case.c_ref,
        cfg.solver.eta_limiter,
        cfg.solver.name,
        cfg.kernel.name,
        cfg.solver.is_bc_trick,
        cfg.solver.density_evolution,
        cfg.solver.artificial_alpha,
        cfg.solver.free_slip,
        cfg.solver.density_renormalize,
        cfg.solver.heat_conduction,
    )
    forward = solver.forward_wrapper()

    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=solver._kernel_fn.cutoff,
        backend=cfg.nl.backend,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=state["r"].shape[0],
        num_partitions=cfg.nl.num_partitions,
        pbc=np.array(cfg.case.pbc),
    )
    num_particles = (state["tag"] != Tag.PAD_VALUE).sum()
    neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)

    # Instantiate advance function for our use case
    advance = si_euler(cfg.solver.tvf, forward, shift_fn, bc_fn)

    advance = advance if cfg.no_jit else jit(advance)  # TODO: is this even needed?

    print("#" * 79, "\nStarting a JAX-SPH run with the following configs:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 79)

    # create data directory and dump config.yaml
    dir = io_setup(cfg)

    # compile kernel and initialize accelerations
    _state, _neighbors = advance(0.0, state, neighbors)
    _state["v"].block_until_ready()

    digits = len(str(cfg.solver.sequence_length))
    start = time.time()
    for step in range(cfg.solver.sequence_length + 2):
        write_state(step - 1, state, dir, cfg)

        state_, neighbors_ = advance(cfg.solver.dt, state, neighbors)

        # Check whether the edge list is too small and if so, create longer one
        if neighbors_.did_buffer_overflow:
            edges_ = neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {step}")
            neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)
            print(f"To list {neighbors.idx.shape}")

            # To run the loop N times even if sometimes did_buffer_overflow > 0
            # we directly rerun the advance step here
            state, neighbors = advance(cfg.solver.dt, state, neighbors)
        else:
            state, neighbors = state_, neighbors_

        # update the progress bar
        if step % cfg.io.write_every == 0:
            t_ = (step + 1) * cfg.solver.dt
            ekin_ = get_ekin(state, cfg.case.dx)
            u_max_ = get_val_max(state, "u")
            T_max_ = get_val_max(state, "T")

            msg = f"{str(step).zfill(digits)}/{cfg.solver.sequence_length}"
            msg += f", t={t_:.4f}, Ekin={ekin_:.7f}, u_max={u_max_:.4f}"
            msg += f", T_max={T_max_:.4f}" if cfg.solver.heat_conduction else ""
            print(msg)

    print(f"time: {time.time() - start:.2f} s")
