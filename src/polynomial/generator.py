import numpy as np
from gurobipy import gurobipy, GRB
from src import symbolics

class Constraint:
    def __init__(self, a, beta) -> None:
        self.a = a
        self.beta = beta

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class CoeffsGenerator:
    def __init__(self, syms, expr_vals, exprs_dirs) -> None:
        self.syms = syms
        ncoeff = len(expr_vals)
        assert len(exprs_dirs) == ncoeff
        self.ncoeff = ncoeff
        self.expr_vals = expr_vals
        self.exprs_dirs = exprs_dirs
        self.rmax = 2
        self.constraints = []
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            self.constraints.append(Constraint(+a, -1))
            self.constraints.append(Constraint(-a, -1))

    def add_constraint_pos(self, states):
        a = np.array([
            symbolics.evalf_expr(expr_val, self.syms, states)
            for expr_val in self.expr_vals
        ])
        self.constraints.append(Constraint(-a, 0))

    def add_constraint_lie(self, states, derivs):
        a = np.array([
            symbolics.evalf_expr(np.dot(exprs_dir, derivs), self.syms, states)
            for exprs_dir in self.exprs_dirs
        ])
        self.constraints.append(Constraint(+a, 0))

    def compute_coeffs_robust(self, *, output_flag=True):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        coeffs_ = model.addVars(
            self.ncoeff, lb=-self.rmax, ub=+self.rmax, name='c'
        )
        coeffs = np.array(coeffs_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for h in self.constraints:
            a = h.a
            beta = h.beta
            na = np.linalg.norm(a)
            model.addConstr(np.dot(a, coeffs) + beta + na*r <= 0)

        model.setObjective(r, GRB.MAXIMIZE)
        model.optimize()

        if model.Status != 2:
            raise GeneratorError(
                'Chebyshev center status = %d.' % model.status
            )

        coeffs_opt = np.array([var.X for var in coeffs])
        r_opt = model.getObjective().getValue()
        return coeffs_opt, r_opt