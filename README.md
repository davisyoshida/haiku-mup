# MUP for Haiku

This is a (very preliminary) port of Yang and Hu et al.'s [μP repo](https://github.com/microsoft/mup) to Haiku and JAX. It's not feature complete, and I'm very open to suggestions on improving the usability.

## Learning rate demo
These plots show the evolution of the optimal learning rate for a 3-hidden-layer MLP on MNIST, trained for 10 epochs (5 trials per lr/width combination).

With standard parameterization, the learning rate optimum continues changing as the width increases:

<img src="https://github.com/davisyoshida/haiku-mup/blob/master/figures/mlp_sp.png?raw=True" width="500" />


With μP, the learning rate optimum stabilizes as width increases:

<img src="https://github.com/davisyoshida/haiku-mup/blob/master/figures/mlp.png?raw=True" width="500" />

## Usage
```python
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
from optax import adam, chain

from hk_mup import apply_mup, Mup, Readout, get_shapes

class MyModel(hk.Module):
    def __init__(self, width, n_classes=10):
        super().__init__(name='model')
        self.width = width
        self.n_classes = n_classes
        
    def __call__(self, x):
        x = hk.Linear(self.width)(x)
        x = jax.nn.relu(x)
        return Readout(2)(x) # 1. Replace output layer with Readout layer

def fn(x, width=100):
    with apply_mup(): # 2. Modify parameter creation with apply_mup()
        return MyModel(width)(x)

init_input = jnp.zeros(123)
base_model = hk.transform(partial(fn, width=1))
base_params = hk.init(fn, jax.random.PRNGKey(0), init_input)
base_shapes = get_shapes(base_params) # 3. Get shapes from base model
del base_params


model = hk.transform(fn)

mup = Mup()
with mup.init_context(): # 4. Use a `Mup` instance to initialize your model
    params = model.init(jax.random.PRNGKey(0), init_input)

model = mup.retransform_model_fn(fn) # 5. Get final model with `retransform_model_fn`

optimizer = optax.adam(3e-4)
optimizer = mup.wrap_optimizer(optimizer, adam=True) # 6. Use wrap_optimizer to get layer specific learning rates

# Now the model can be trained as normal
```

### Summary
1. Replace output layers with `Readout` layers
2. Modify parameter creation with the apply_mup() context manager
3. Get shapes from base model size
4. Use a `Mup` instance to initialize your model with Mup.init_context
5. Get final model with `Mup.retransform_model_fn`
6. Wrap optimizer with `Mup.wrap_optimizer`
