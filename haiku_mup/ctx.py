from contextvars import ContextVar
from enum import Enum

class MupMode(Enum):
    base = 1
    target = 2
    apply = 3

mup_context = ContextVar('mup_context', default=None)
