"""Integrator schemes"""
from jax import numpy as jnp

def si_euler(tvf, model, shift_fn, bc_fn):
    """Semi-implicit Euler integrator"""

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

def si_euler_RIE(model, shift_fn, bc_fn):
    """Semi-implicit Euler integrator"""

    def advance(dt, state, neighbors):
        # 1. Twice 1/2dt integration of u and v
        state["v"] += 1.0 * dt * state["dvdt"]


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




def kick_drift_kick_RIE2(model, shift_fn, bc_fn):

    def advance(dt, state, neighbors):
     
        # 1. 1/2dt integration of v
        v_05 = state["v"] + 0.5 * dt * state["dvdt"]
        state["v"] = v_05

        # 2. Integrate position with velocity v
        state["r"] = shift_fn(state["r"], 1.0 * dt * v_05)

        # 3. Update neighbors list
        num_particles = (state["tag"] != -1).sum()
        neighbors = neighbors.update(state["r"], num_particles=num_particles)

        # 4. Compute derivatives in "drift state"
        state = model(state, neighbors)
        
        # 6 fully integrate v
        state["v"] = v_05 + 0.5 * dt * state["dvdt"]

        # 7. Impose boundary conditions on dummy particles (if applicable)
        state = bc_fn(state)


        return state, neighbors

    return advance


'''
def kick_drift_kick(model, shift_fn):

    def advance(dt, state, neighbors):
        # 1. 1/2dt integration of v
        v_05 = state["v"] + 0.5 * dt * state["dvdt"]
        state["v"] = v_05

        rho_n = state["rho"]

        # 2. Integrate position with velocity v
        state["r"] = shift_fn(state["r"], 1.0 * dt * v_05)

        # 3. Update neighbors list
        num_particles = (state["tag"] != -1).sum()
        neighbors = neighbors.update(state["r"], num_particles=num_particles)

        # 4. Compute derivatives in "drift state"
        state = model(state, neighbors)

        # 5. Integrate density 
        state["rho"] = rho_n + dt * state["drhodt"]

        # 6 fully integrate v
        state["v"] = v_05 + 0.5 * dt * state["dvdt"]


        return state, neighbors

    return advance
'''
