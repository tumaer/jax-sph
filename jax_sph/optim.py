
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
import os
import copy
import optax

from io_state import read_h5
from jax_sph.io_state import io_setup
from jax_md import space
from jax_sph.solver.sph_tvf import SPHTVF
from jax_sph import partition
from jax_sph.integrator import si_euler
from jax_sph.args import Args
from jax_md.partition import Sparse
from cases import select_case
from jax_sph.utils import get_ekin, get_val_max
import matplotlib.pyplot as plt
from jax.experimental import io_callback

#Optimization Algorithm:
# Step 1: Define Loss Function which takes in the random initial position and other solver parameters and runs the forward simulation
# Step 2: Load the target ground truth position from the paraview file
# Step 3: Inside the loss function, compute the mse between the target and the predicted position  
# Step 4: Return the loss and the state computed from forward pass.
# Step 5: Define the optimization loop, where the initial state is updated using the gradients of the loss function: init_state[r] = init_state[r] - learning_rate * grads[-1]


def optimization_case_setup(args, offset_init):
    
    case = select_case(args.case)
    args, box_size, state, g_ext_fn, bc_fn, eos_fn, key = case(args).initialize()

    #Offset the fluid particle from the target_init_state
    mask = state["tag"] == 0
    fluid_min = jnp.min(state["r"][mask], axis=0)
    state["r"] = state["r"].at[mask].set(state['r'][mask]- fluid_min + offset_init + 0.15)
    
    
    displacement_fn, shift_fn = space.periodic(side=box_size)
    
    # Initialize a neighbors list for looping through the local neighborhood
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=3 * args.dx,
        backend=args.nl_backend,
        capacity_multiplier=2.0,
        mask_self=False,
        format=Sparse,
        num_particles_max=state["r"].shape[0],
        num_partitions=args.num_partitions,
        pbc=np.array(args.periodic_boundary_conditions),
    )
    num_particles = (state["tag"] != -1).sum()
    neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)

    # Solver setup
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
        
    # Instantiate advance function for our use case
    advance = si_euler(args.tvf, model, shift_fn, bc_fn)
    advance = advance if args.no_jit else jit(advance)
    
    # compile kernel and initialize accelerations
    _state, _neighbors = advance(0.0, state, neighbors)
    _state["v"].block_until_ready()
    
    return advance, state, neighbors, neighbor_fn, num_particles


def forward_simulation(advance, state, neighbors, neighbor_fn, num_particles,args):
    
    for step in range(args.sequence_length):
        
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
        # if step % args.write_every == 0:
        #     t_ = (step + 1) * args.dt
        #     ekin_ = get_ekin(state, args.dx)
        #     u_max_ = get_val_max(state, "u")
        #     print(
        #         f"{step} / {args.sequence_length}, t = {t_:.4f} Ekin = {ekin_:.7f} "
        #         f"u_max = {u_max_:.4f}"
        #     )
    #returns the final state
    return state

# Loss function:
def loss_fn_wrapper(advance,args):
    def loss_fn(state, neighbors, neighbor_fn, num_particles, target_position):
        # Forward Simulation
        state = forward_simulation(advance, state, neighbors, neighbor_fn, num_particles, args)
        mask = state["tag"] == 0
        fluid_position = state["r"][mask]
        
        loss = jnp.sqrt(jnp.mean((fluid_position - target_position)**2))
        return loss
    
    return loss_fn

def ploting(r_plt, init_state, target_position):
    fig = plt.figure(figsize=(5, 5))
    plt.scatter(r_plt[:,0],r_plt[:,1], label='updated', s=5)
    plt.scatter(target_position[:,0], target_position[:,1], label='target', s=5)
    plt.scatter(init_state["r"][:,0], init_state["r"][:,1], label='init', s=5)
    plt.legend()
    plt.axis('equal')
    plt.show()
    print('done')
 
if __name__ == "__main__":
    num_optimization_steps =4
    learning_rate = 0.8
    momentum_parameter = 0.9
    grads=[]
    
    #Load the target state
    target_dir_path = "target_traj/"
    target_filename = 'traj_50.h5'
    file_path_h5 = os.path.join(target_dir_path, target_filename)
    target_state = read_h5(file_path_h5)
    
    args = Args().args
    
    offset_init = jnp.array([0.1, 0.2])
    
    advance, init_state , neighbors, neighbor_fn, num_particles=optimization_case_setup(args, offset_init)
    state = copy.deepcopy(init_state) #init_state required for plotting
    mask = state["tag"] == 0
    target_position =target_state["r"][mask] #Extract only the fluid particles
    
    #Initialize GD + momentum
    velocity = jnp.zeros_like(state["r"][mask])
    
    #Adams
    # lr_scheduler = optax.exponential_decay(
    #     init_value=1,  #0.0005
    #     transition_steps=50, #1,00,000
    #     decay_rate=0.1, #0.1
    #     end_value=0.001, #1e-6
    # )
    # #optimizer = optax.adam(learning_rate=0.1)
    # optimizer=optax.adam(learning_rate=lr_scheduler)
    # opt_state = optimizer.init(state["r"][mask])

    #Main Optimization Loop:
    loss_fn = loss_fn_wrapper(advance,args)
    for optimization_step in range(num_optimization_steps):
        
        loss, grad = jax.value_and_grad(fun=loss_fn, allow_int=True)(state, neighbors, neighbor_fn, num_particles,target_position)
        
        grads.append(grad)
        
        mask = state["tag"] == 0
        
        #Naive Gradient Descent
        #state["r"] = state["r"].at[mask].set(state['r'][mask] - learning_rate * grads[-1]['r'][mask])
        
        #Gradient Descent with Momentum
        velocity = momentum_parameter * velocity - learning_rate * grads[-1]['r'][mask]
        state["r"] = state["r"].at[mask].set(state['r'][mask] + velocity)
        
        ##Adams
        # updates, optimizer_state = optimizer.update(grads[-1]['r'][mask], opt_state)
        # state['r'] = state["r"].at[mask].set(optax.apply_updates(state['r'][mask], updates))

        print(f"\n Step {optimization_step}, Loss: {loss}")
  
   
    jax.debug.callback(ploting, jnp.asarray(state["r"]), init_state, target_position)