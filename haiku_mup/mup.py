from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from dataclasses import dataclass

import haiku as hk
import jax
import optax

from .ctx import mup_context, MupMode
from .init import ConstantStdInit
from .module import Readout, SharedEmbed

def get_shapes(params):
    return jax.tree_map(lambda p: p.shape, params)

class Mup:
    """Class which tracks infinite shapes, and applies per-parameter learning rates/multipliers"""
    def __init__(self):
        self.base_shapes = None

        self.readout_mults = {}
        self._shared_readout_mult = None
        self._adam_lrs = defaultdict(dict)
        self._sgd_lrs = defaultdict(dict)

        self._mode = None

    @property
    def mode(self):
        return self._mode

    @property
    def shared_readout_mult(self):
        if self._shared_readout_mult is None:
            raise ValueError('Attempted to use shared output layer without an input embedding layer')

        return self._shared_readout_mult

    @shared_readout_mult.setter
    def shared_readout_mult(self, value):
        if self._shared_readout_mult is not None:
            raise ValueError('Currently only one SharedEmbed module is supported.')
        self._shared_readout_mult = value

    @contextmanager
    def init_base(self):
        self.base_shapes = {}
        token = mup_context.set(self)
        self._mode = MupMode.base
        try:
            yield
        finally:
            mup_context.reset(token)
            self._mode = None

    @contextmanager
    def init_target(self):
        """A context manager which uses a custom Haiku creator to track parameter initialization
        to rescale initialization and determine learning rates/weight multipliers"""
        token = mup_context.set(self)
        self._mode = MupMode.target
        if len(self._adam_lrs):
            raise ValueError('Attempted to re-use mup context')
        try:
            yield
        finally:
            mup_context.reset(token)
            self._mode = None

    @contextmanager
    def _apply_ctx(self):
        token = mup_context.set(self)
        self._mode = MupMode.apply
        try:
            yield
        finally:
            mup_context.reset(token)
            self._mode = None

    def _mup_creator(self, next_creator, shape, dtype, init, context):
        if self._mode == MupMode.base:
            self.base_shapes[context.full_name] = shape
            return next_creator(shape, dtype, init)
        if self._mode != MupMode.target:
            raise ValueError('Using _mup_creator outside of Mup context managers.')

        n_inf, inf_ratios = self._get_inf_ratios(context.full_name, shape)
        full_name = context.full_name
        parent, name = full_name.rsplit('/', maxsplit=1)

        width_mult = 1 if n_inf == 0 else inf_ratios[0]
        if n_inf == 2:
            fanin_fanout_ratio = width_mult / inf_ratios[1]
            self._set_lrs(
                full_name,
                sgd_lr=1 / fanin_fanout_ratio,
                adam_lr=1 / width_mult
            )
        elif n_inf == 1:
            self._set_lrs(
                full_name,
                sgd_lr=width_mult,
                adam_lr=1.
            )
        else:
            self._set_lrs(
                full_name,
                sgd_lr=1.,
                adam_lr=1.
            )

        new_init = init

        is_readout_w = isinstance(context.module, Readout) and name == 'w'
        if is_readout_w:
            new_init = ConstantStdInit(init, div=1 / width_mult)
            self.readout_mults[parent] = width_mult

        if isinstance(context.module, SharedEmbed):
            self.shared_readout_mult = width_mult

        return next_creator(shape, dtype, new_init)

    def wrap_model(self, model):
        @wraps(model.apply)
        def mup_apply(*args, **kwargs):
            with self._apply_ctx():
                return model.apply(*args, **kwargs)
        return hk.Transformed(init=model.init, apply=mup_apply)

    def wrap_optimizer(self, optimizer, adam=True):
        """Apply the per-parameter learning rates computed by `init_context` to an Optax optimizer."""
        if not self._adam_lrs:
            raise ValueError('Attempted to wrap optimizer before initializing network. Did you forget to use init_base/init_target/apply_mup?')

        def init_fn(params):
            del params
            return optax.EmptyState()

        def update_fn(updates, state, params=None):
            del params
            updates = jax.tree_map(
                lambda update, scale: update * scale,
                updates,
                dict(self._adam_lrs if adam else self._sgd_lrs)
            )

            return updates, state

        return optax.chain(
            optimizer,
            optax.GradientTransformation(init_fn, update_fn)
        )

    def _set_lrs(self, full_name, sgd_lr, adam_lr):
        parent, name = full_name.rsplit('/', maxsplit=1)
        self._sgd_lrs[parent][name] = sgd_lr
        self._adam_lrs[parent][name] = adam_lr

    def _get_inf_ratios(self, full_name, shape):
        base = self.base_shapes[full_name]
        n_inf = sum(a != b for a, b in zip(base, shape))
        if n_inf > 2:
            raise ValueError(f'At most two infinite dimensions supported. Found {n_inf} in {full_name}')

        inf_ratios = [b / a for a, b in zip(base, shape) if a != b]
        return n_inf, inf_ratios


@contextmanager
def apply_mup():
    ctx = mup_context.get()
    is_init = ctx is not None

    if is_init:
        with hk.custom_creator(ctx._mup_creator):
            yield
    else:
        yield
