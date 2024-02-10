"""Core simulation loop"""

import time

import numpy as np
from jax import jit
from jax_md import space
from jax_md.partition import Sparse

from cases import select_case
from jax_sph import partition
from jax_sph.integrator import kick_drift_kick_RIE
from jax_sph.integrator import si_euler
from jax_sph.io_state import io_setup, write_state
from jax_sph.solver.sph_tvf import SPHTVF
from jax_sph.solver.sph_riemann import SPHRIEMANN
from jax_sph.solver.ut import UTSimulator
from jax_sph.utils import get_ekin, get_val_max


def simulate(args):
    # Case setup
    case = select_case(args.case)
    args, box_size, state, g_ext_fn, bc_fn, eos_fn, key = case(args).initialize()

    displacement_fn, shift_fn = space.periodic(side=box_size)

    if args.kernel == "QSK":
        r_cut = 3
    elif args.kernel == "WC2K":
        r_cut = 2.6

    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=r_cut * args.dx,
        backend=args.nl_backend,
        dr_threshold=r_cut * args.dx * 0.25,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=state["r"].shape[0],
        num_partitions=args.num_partitions,
        pbc=np.array(args.periodic_boundary_conditions),
    )
    num_particles = (state["tag"] != -1).sum()
    neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)

    # Solver setup
    if args.solver == "SPH":
        model = SPHTVF(
            displacement_fn,
            eos_fn,
            g_ext_fn,
            args.dx,
            args.dim,
            args.dt,
            args.is_bc_trick,
            args.density_evolution,
            args.artificial_alpha,
            args.free_slip,
            args.density_renormalize,
        )
    elif args.solver == "RIE":
        model = SPHRIEMANN(
            displacement_fn,
            eos_fn,
            g_ext_fn,
            args.dx,
            args.dim,
            args.dt,
            args.Vmax,
            args.eta_limiter,
            args.is_limiter,
            args.density_evolution,
            args.is_bc_trick,
            args.free_slip,
        )
    elif args.solver == "UT":
        model = UTSimulator(g_ext_fn)

    # Instantiate advance function for our use case
    if args.solver == "RIE":
        advance = kick_drift_kick_RIE(args.tvf, model, shift_fn, bc_fn)
    else:    
        advance = si_euler(args.tvf, model, shift_fn, bc_fn)

    advance = advance if args.no_jit else jit(advance)

    # create data directory and dump args.txt
    dir = io_setup(args)

    # compile kernel and initialize accelerations
    _state, _neighbors = advance(0.0, state, neighbors)
    _state["v"].block_until_ready()

    start = time.time()
    for step in range(args.sequence_length + 2):
        # TODO: writing for the first time is not at zero. Why?
        write_state(step - 1, args.sequence_length, state, dir, args)

        state_, neighbors_ = advance(args.dt, state, neighbors)

        # Check whether the edge list is too small and if so, create longer one
        if neighbors_.did_buffer_overflow:
            edges_ = neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {step}")
            neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)
            print(f"To list {neighbors.idx.shape}")

            # To run the loop N times even if sometimes did_buffer_overflow > 0
            # we directly rerun the advance step here
            state, neighbors = advance(args.dt, state, neighbors)
        else:
            state, neighbors = state_, neighbors_

        # update the progress bar
        if step % args.write_every == 0:
            t_ = (step + 1) * args.dt
            ekin_ = get_ekin(state, args.dx)
            u_max_ = get_val_max(state, "u")
            print(
                f"{step} / {args.sequence_length}, t = {t_:.4f} Ekin = {ekin_:.7f} "
                f"u_max = {u_max_:.4f}"
            )

    print(f"time: {time.time() - start:.2f} s")
