import haiku as hk
import jax.numpy as jnp

from .ctx import mup_context, MupMode

class Readout(hk.Linear):
    """Wrapper around hk.Linear. Used by Mup to set different learning rate."""
    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, name=name if name is not None else 'readout', **kwargs)

    def __call__(self, inputs, **kwargs):
        mup_ctx = mup_context.get()
        if mup_ctx is not None:
            if mup_ctx.mode is None:
                raise ValueError(
                    'Called Readout outside of Mup context. Must be used inside Mup.init_base, Mup.init_target,'
                    'or after calling Mup.wrap_model.'
                )
            if mup_ctx.mode == MupMode.apply:
                inputs /= mup_ctx.readout_mults[self.name]

        result = super().__call__(inputs, **kwargs)

        return result

class SharedEmbed(hk.Embed):
    def get_weights(self):
        return self.params_dict()[f'{self.name}/embeddings']

class SharedReadout(hk.Module):
    """Readout layer for use with shared embeddings. Manually pass the weight matrix when calling."""
    def __init__(self, use_bias=False, bias_init=None, name=None):
        if bias_init and not use_bias:
            raise ValueError('Received value for bias_init but use_bias is False')

        self.use_bias = use_bias
        self.bias_init = bias_init
        super().__init__(name=name if name is not None else 'shared_readout')

    def __call__(self, weight, x):
        if len(weight.shape) != 2 or weight.shape[-1] != x.shape[-1]:
            raise ValueError("weight shape should be (output_size, h), final dimension of x should shape should be h")

        out = x @ weight.T

        mup_ctx = mup_context.get()
        if mup_ctx is not None:
            if mup_ctx.mode is None:
                raise ValueError(
                    'Called SharedReadout outside of Mup context. Must be used inside Mup.init_base, Mup.init_target,'
                    'or after calling Mup.wrap_model.'
                )
            if mup_ctx.mode == MupMode.apply:
                out /= mup_ctx.shared_readout_mult

        if self.use_bias:
            bias_init = self.bias_init if self.bias_init is not None else hk.initializers.Constant(0)
            bias = hk.get_parameter('b', [weight.shape[0]], dtype=weight.dtype, init=bias_init)
            out += jnp.broadcast_to(bias, out.shape)

        return out
