import numpy as np

class GenericSystem:
    def __init__(self, field) -> None:
        self.field = field
    
    def flow(self, vars, inps):
        return self.field(vars, inps)

class AffineMap:
    def __init__(self, A, b) -> None:
        self.A = A
        self.b = b
    
    def val(self, vars):
        return np.dot(self.A, vars) + self.b

class AffineSystem:
    def __init__(self, aff_maps):
        self.aff_maps = aff_maps

    def flow(self, vars, inps):
        vals = np.array([f.val(vars) for f in self.aff_maps])
        return np.dot(inps, vals)


def open_loop(sys, finps):
    field = lambda vars, t : sys.flow(vars, finps(t))
    return GenericSystem(field)