import numpy as np

class GenericSystem:
    def __init__(self, field):
        self.field = field
        return None
    
    def flow(self, vars, inps):
        return self.field(vars, inps)

class AffineMap:
    def __init__(self, A, b):
        self.A = A
        self.b = b
    
    def val(self, vars):
        return np.dot(self.A, vars) + self.b

class AffineSystem:
    def __init__(self, aff_maps):
        self.aff_maps = aff_maps

    def flow(self, vars, inps):
        d = np.zeros(len(vars))
        for i in range(len(inps)):
            d = d + inps[i]*(self.aff_maps[i].val(vars))
        return d

def open_loop(sys, finps):
    field = lambda vars, t : sys.flow(vars, finps(t))
    return GenericSystem(field)