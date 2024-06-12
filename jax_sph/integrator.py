"""Integrator schemes."""

from typing import Callable, Dict

from jax_sph.utils import Tag


def si_euler(
    tvf: float, model: Callable, shift_fn: Callable, bc_fn: Callable, nw_fn: Callable
):
    """Semi-implicit Euler integrator including Transport Velocity.

    The integrator advances the state of the system following the steps:

        1. Update u and v with dudt and dvdt
        2. Update position r with velocity v
        3. Update neighbor list
        4. Compute accelerations - SPH model call
        5. Impose boundary conditions as defined by the case.
    """

    def advance(dt: float, state: Dict, neighbors):
        """Call to integrator."""

        # 1. Twice 1/2dt integration of u and v
        state["u"] += 1.0 * dt * state["dudt"]
        state["v"] = state["u"] + tvf * 0.5 * dt * state["dvdt"]

        # 2. Integrate position with velocity v
        state["r"] = shift_fn(state["r"], 1.0 * dt * state["v"])

        # recompute wall normals if needed
        if nw_fn is not None:
            state["nw"] = nw_fn(state["r"])

        # 3. Update neighbor list

        # The displacment and shift function from JAX MD are used for computing the
        # relative particle distance and moving particles across the priodic domain
        #
        # dr = displacement_fn(r1, r2) = r1 - r2
        # respecting PBC, i.e. returns |dr_i| < box_size / 2
        #
        # r = shift_fn(r, dr) = r + dr
        # respecting PBC, i.e. new r in [0, box_size]

        num_particles = (state["tag"] != Tag.PAD_VALUE).sum()
        neighbors = neighbors.update(state["r"], num_particles=num_particles)

        # 4. Compute accelerations
        state = model(state, neighbors)

        # 5. Impose boundary conditions on dummy particles (if applicable)
        state = bc_fn(state)

        return state, neighbors

    return advance
