
import jax
import jax.numpy as jnp
#Optimization loop to be written here
# Step 1: Define Loss Function which takes in the random initial position and other solver parameters and runs the forward simulation
# Step 2: Load the target ground truth position from the paraview file
# Step 3: Inside the loss function, compute the mse between the target and the predicted position  
# Step 4: Return the loss and the state computed from forward pass.
# Step 5: Define the optimization loop, where the initial state is updated using the gradients of the loss function: init_state[r] = init_state[r] - learning_rate * grads[-1]
# #(loss, state), grads = value_and_grad(loss_fn)(state):


num_optimization_steps =10
learning_rate = 0.01
grads=[]

# Loss function:
def loss_fn(init_state):
    # Forward Simulation
    state = forward_simulation(init_state)
    # Compute the loss
    loss = jnp.mean((state['position'] - target_position)**2)
    return loss, state

#Optimization Loop:
for optimization_step in range(num_optimization_steps):
    (loss, _), grad = jax.value_and_grad(loss_fn)(init_state)
    
    grads.append(grad)
    
    init_state =  init_state - learning_rate * grads[-1]
    
    print(f"Step {optimization_step}, Loss: {loss}")