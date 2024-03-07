"""Integrator schemes"""


def si_euler(tvf, model, shift_fn, bc_fn):
    """Semi-implicit Euler integrator including tvf"""

    def advance(dt, state, neighbors):
        # 1. Twice 1/2dt integration of u and v
        state["u"] += 1.0 * dt * state["dudt"]
        state["v"] = state["u"] + tvf * 0.5 * dt * state["dvdt"]

        # 2. Integrate position with velocity v
        state["r"] = shift_fn(state["r"], 1.0 * dt * state["v"])

        # 3. Update neighbors list

        # The displacment and shift function from JAX MD are used for computing the
        # relative particle distance and moving particles across the priodic domain
        #
        # dr = displacement_fn(r1, r2) = r1 - r2
        # respecting PBC, i.e. returns |dr_i| < box_size / 2
        #
        # r = shift_fn(r, dr) = r + dr
        # respecting PBC, i.e. new r in [0, box_size]

        num_particles = (state["tag"] != -1).sum()
        neighbors = neighbors.update(state["r"], num_particles=num_particles)

        # 4. Compute accelerations
        state = model(state, neighbors)

        # 5. Impose boundary conditions on dummy particles (if applicable)
        state = bc_fn(state)

        return state, neighbors

    return advance
