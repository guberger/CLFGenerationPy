import numpy as np

class GenericSystem:
    def __init__(self, field) -> None:
        self.field = field
    
    def flow(self, states, inputs):
        return self.field(states, inputs)

class AffineMap:
    def __init__(self, A, b) -> None:
        self.A = A
        self.b = b
    
    def val(self, states):
        return np.dot(self.A, states) + self.b

class AffineSystem:
    def __init__(self, aff_maps):
        self.aff_maps = aff_maps

    def flow(self, states, inputs):
        vals = np.array([f.val(states) for f in self.aff_maps])
        return np.dot(inputs, vals)

class MultiSystem:
    def __init__(self, syss) -> None:
        self.syss = syss
    
    def flow(self, mode, states, inputs):
        return self.syss[mode].flow(states, inputs)

def open_loop(sys, finputs):
    field = lambda states, t : sys.flow(states, finputs(t))
    return GenericSystem(field)