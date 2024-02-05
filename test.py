import jax.numpy as jnp
from jax import ops, vmap
from jax_md import space
import numpy as np


#a = jnp.array([1, 0, 0, 0, 1, 1, 0, 1, 1, 0])
#c = jnp.array([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0],[2, 0, 0, 2, 2, 4, 3, 3, 2, 1]] )
#print(a)
#mask_wall = a == 1
#mask_wall = mask_wall.reshape((1,10))
#b = jnp.where(a == 0, 1, 0)
#print(mask_wall)
#print(a[a == 1])


# a = jnp.array([[-1, 1], [-1, -11], [1, -1], [1, 1]])
# #print(jnp.linalg.norm(a, ord=2, axis=1))
# print(jnp.where(a < 0, 0, a))

a = 1 - jnp.array([[1 , 2, 3],[ 1, 2,5]])
print(jnp.prod(a, axis=1))
