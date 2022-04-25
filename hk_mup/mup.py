from collections import defaultdict
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import partial, wraps
from dataclasses import dataclass

import haiku as hk
import jax
import optax

from .init import ConstantStdInit
from .module import Readout

def get_shapes(params):
    return jax.tree_map(lambda p: p.shape, params)

_mup_context = ContextVar('mup_context', default=None)

class Mup:
    def __init__(self):
        self.base_shapes = None

        self.readout_mults = {}
        self._lrs = defaultdict(dict)

    @contextmanager
    def init_context(self, base_shapes):
        self.base_shapes = base_shapes
        token = _mup_context.set(self)
        if len(self._lrs):
            raise ValueError('Attempted to re-use mup context')
        try:
            yield
        finally:
            _mup_context.reset(token)
            self.base_shapes = None

    def wrap_optimizer(self, optimizer, adam=True):
        if not adam:
            raise NotImplementedError()

        if not self._lrs:
            raise ValueError('Attempted to wrap optimizer before initializing network')

        def init_fn(params):
            del params
            return optax.EmptyState()

        def update_fn(updates, state, params=None):
            del params
            updates = jax.tree_map(
                lambda update, scale: update * scale,
                updates,
                dict(self._lrs)
            )

            return updates, state

        return optax.GradientTransformation(init_fn, update_fn)

    def retransform_model_fn(self, f):
        @wraps(f)
        def fn(*args, **kwargs):
            with hk.custom_getter(partial(_mup_getter, mup_ctx=self)):
                return f(*args, **kwargs)

        return hk.transform(fn)

    def set_lr(self, full_name, value):
        parent, name = full_name.rsplit('/')
        self._lrs[parent][name] = value

@contextmanager
def apply_mup():
    ctx = _mup_context.get()
    is_init = ctx is not None

    with ExitStack() as stack:
        if is_init:
            creator = partial(_mup_creator, mup_ctx=ctx)
            stack.enter_context(hk.custom_creator(creator))

        yield

def _get_inf_ratios(mup_ctx, full_name, shape):
    parent, name = full_name.rsplit('/')
    base = mup_ctx.base_shapes[parent][name]
    n_inf = sum(a != b for a, b in zip(base, shape))
    if n_inf > 2:
        raise ValueError(f'At most two infinite dimensions supported. Found {n_inf} in {full_name}')

    inf_ratios = [b / a for a, b in zip(base, shape) if a != b]
    return n_inf, inf_ratios

def _mup_creator(next_creator, shape, dtype, init, context, *, mup_ctx):
    if mup_ctx is None:
        raise ValueError('Attempted to use `use_mup()` outside of `apply_mup` context')

    n_inf, inf_ratios = _get_inf_ratios(mup_ctx, context.full_name, shape)
    parent, _ = context.full_name.rsplit('/')

    width_mult = 1 if n_inf == 0 else inf_ratios[0]
    is_readout_w = isinstance(context.module, Readout) and n_inf == 1
    if n_inf == 2:
        mup_ctx.set_lr(context.full_name, 1 / width_mult)
    elif is_readout_w:
        init = ConstantStdInit(init, div=1 / width_mult)
        mup_ctx.set_lr(context.full_name, 1.)
        mup_ctx.readout_mults[parent] = width_mult
    else:
        mup_ctx.set_lr(context.full_name, 1.)

    return next_creator(shape, dtype, init)

def _mup_getter(next_getter, value, context, *, mup_ctx):
    val = next_getter(value)
    if not isinstance(context.module, Readout):
        return val

    parent = context.full_name.rsplit('/')[0]
    width_mult = mup_ctx.readout_mults.get(parent)
    if not width_mult:
        return val

    return val / width_mult
