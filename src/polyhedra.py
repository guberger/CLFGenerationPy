import numpy as np
import gurobipy as gpy
from gurobipy import GRB

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

class ChebyshevError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def chebyshev_center(p, rmax, *, output_flag=True):
    model = gpy.Model('Chebyshev center')
    model.setParam('OutputFlag', output_flag)
    vars_ = model.addVars(p.nvar, lb=-rmax, ub=+rmax, name='x')
    vars = np.array(vars_.values())
    r = model.addVar(lb=-float('inf'), ub=rmax, name='r')

    for h in p.halfspaces:
        a = h.a
        beta = h.beta
        na = np.linalg.norm(a)
        model.addConstr(np.dot(a, vars) + beta + na*r <= 0)
    
    model.setObjective(r, GRB.MAXIMIZE)
    model.optimize()

    if model.Status != 2:
        raise ChebyshevError('Status = %d.' % model.status)

    vars_opt = np.array([var.X for var in vars])
    r_opt = model.getObjective().getValue()
    return vars_opt, r_opt