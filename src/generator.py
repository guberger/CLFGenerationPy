import numpy as np
from src import symbolics
from src import polyhedra

class CoeffsGenerator:
    def __init__(self, syms, expr_vals, exprs_dirs) -> None:
        self.syms = syms
        ncoeff = len(expr_vals)
        assert len(exprs_dirs) == ncoeff
        self.expr_vals = expr_vals
        self.exprs_dirs = exprs_dirs
        self.rmax = 2
        p = polyhedra.Polyhedron(ncoeff)
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            p.add_halfspace(polyhedra.Halfspace(+a, -1))
            p.add_halfspace(polyhedra.Halfspace(-a, -1))
        self.p = p

    def add_constraint_pos(self, states):
        a = np.array([
            symbolics.evalf_expr(expr_val, self.syms, states)
            for expr_val in self.expr_vals
        ])
        self.p.add_halfspace(polyhedra.Halfspace(-a, 0))

    def add_constraint_lie(self, states, derivs):
        a = np.array([
            symbolics.evalf_expr(np.dot(exprs_dir, derivs), self.syms, states)
            for exprs_dir in self.exprs_dirs
        ])
        self.p.add_halfspace(polyhedra.Halfspace(+a, 0))

    def compute_coeffs_robust(self, *, output_flag=True):
        return polyhedra.chebyshev_center(
            self.p, self.rmax, output_flag=output_flag
        )