import numpy as np
from src import symbolics
from src import polyhedra

class FlowExpr:
    def __init__(self, expr_val, expr_dir) -> None:
        self.val = expr_val
        self.dir = expr_dir

class CoeffsGenerator:
    def __init__(self, syms, fexprs) -> None:
        self.syms = syms
        self.fexprs = fexprs
        ncoeff = len(fexprs)
        self.rmax = 2
        p = polyhedra.Polyhedron(ncoeff)
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            p.add_halfspace(polyhedra.Halfspace(+a, -1))
            p.add_halfspace(polyhedra.Halfspace(-a, -1))
        self.p = p

    def add_constraint_pos(self, states):
        a = np.array([
            symbolics.evalf_expr(fexpr.val, self.syms, states)
            for fexpr in self.fexprs
        ])
        self.p.add_halfspace(polyhedra.Halfspace(-a, 0))

    def add_constraint_lie(self, states, derivs):
        a = np.array([
            symbolics.evalf_expr(np.dot(fexpr.dir, derivs), self.syms, states)
            for fexpr in self.fexprs
        ])
        self.p.add_halfspace(polyhedra.Halfspace(+a, 0))

    def compute_coeffs_robust(self, *, output_flag=True):
        return polyhedra.chebyshev_center(
            self.p, self.rmax, output_flag=output_flag
        )