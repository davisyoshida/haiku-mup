import haiku as hk
import jax.numpy as jnp

class ConstantStdInit(hk.initializers.Initializer):
    """Initializer which wraps another initializer and scales down the variance by `div`"""
    def __init__(self, base_initializer, div):
        self.base = base_initializer
        self.div = div

    def __call__(self, shape, dtype):
        if dtype.kind not in ('f', 'V'):
            raise ValueError('Attempted to initialize non-float tensor with infinite shape')

        tensor = self.base(shape, dtype)
        return tensor / (self.div ** 0.5)
