import numpy as np

class Halfspace:
    def __init__(self, a, beta) -> None:
        self.a = a
        self.beta = beta

    def __repr__(self):
        return "<Halfspace a:%s, beta:%s>" % (self.a, self.beta)

    def contains(self, vars):
        return np.dot(self.a, vars) + self.beta <= 0

class Polyhedron:
    def __init__(self, nvar) -> None:
        self.nvar = nvar
        self.halfspaces = []

    def add_halfspace(self, h):
        self.halfspaces.append(h)

    def contains(p, vars):
        for h in p.halfspaces:
            if not h.contains(vars):
                return False
        return True