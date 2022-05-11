import numpy as np

class GenericField:
    def __init__(self, F) -> None:
        self.F = F
    
    def flow(self, states, inputs):
        return self.F(states, inputs)

class AffineMap:
    def __init__(self, A, b) -> None:
        self.A = A
        self.b = b
    
    def val(self, states):
        return np.dot(self.A, states) + self.b

class AffineField:
    def __init__(self, aff_maps):
        self.aff_maps = aff_maps

    def flow(self, states, inputs):
        vals = np.array([f.val(states) for f in self.aff_maps])
        return np.dot(inputs, vals)

def open_loop(field, finputs):
    F = lambda states, t : field.flow(states, finputs(t))
    return GenericField(F)