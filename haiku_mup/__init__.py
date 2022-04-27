""""A simple port of μP  to Haiku/JAX."""

from .mup import get_shapes, Mup, apply_mup
from .module import Readout, SharedEmbed, SharedReadout

__version__ = "0.1.2"
