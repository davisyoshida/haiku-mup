import haiku as hk

class Readout(hk.Linear):
    def __init__(self, *args, name=None, **kwargs):
        if name is None:
            name = 'readout'
        self.width_mult = None
        super().__init__(*args, name=name, **kwargs)
