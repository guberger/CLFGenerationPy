import numpy as np
import gurobipy as gpy
from gurobipy import GRB

class Halfspace:
    def __init__(self, a, beta) -> None:
        self.a = a
        self.beta = beta

    def __repr__(self):
        return "<Halfspace a:%s, beta:%s>" % (self.a, self.beta)

class ChebyshevError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Polyhedron:
    def __init__(self, nvar, rmax) -> None:
        self.nvar = nvar
        self.rmax = rmax
        self.halfspaces = []

    def add_halfspace(self, h):
        self.halfspaces.append(h)

    def compute_chebyshev_center(self, *, output_flag=True):
        model = gpy.Model('Chebyshev center')
        model.setParam('OutputFlag', output_flag)
        rmax = self.rmax
        vars_ = model.addVars(self.nvar, lb=-rmax, ub=+rmax, name='x')
        vars = np.array(vars_.values())
        r = model.addVar(lb=-float('inf'), ub=rmax, name='r')

        for h in self.halfspaces:
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


