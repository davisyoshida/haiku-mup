import haiku as hk

class Readout(hk.Linear):
    """Wrapper around hk.Linear. Used by Mup to set different learning rate."""
    def __init__(self, *args, name=None, **kwargs):
        if name is None:
            name = 'readout'
        super().__init__(*args, name=name, **kwargs)
