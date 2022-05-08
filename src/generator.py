from ast import expr
import numpy as np
import src.symbolics as symbolics
import src.polyhedra as polyhedra

class FlowExpr:
    def __init__(self, expr_val, expr_dir) -> None:
        self.val = expr_val
        self.dir = expr_dir

class CoeffsGenerator:
    def __init__(self, syms, fexprs) -> None:
        self.syms = syms
        self.fexprs = fexprs
        ncoeff = len(fexprs)
        rmax = 2
        p = polyhedra.Polyhedron(ncoeff, rmax)
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            p.add_halfspace(polyhedra.Halfspace(+a, -1))
            p.add_halfspace(polyhedra.Halfspace(-a, -1))
        self.p = p

    def add_constraint_pos(self, vars):
        a = np.array([
            symbolics.evalf_expr(fexpr.val, self.syms, vars)
            for fexpr in self.fexprs
        ])
        self.p.add_halfspace(polyhedra.Halfspace(-a, 0))

    def add_constraint_lie(self, vars, dvars):
        a = np.array([
            symbolics.evalf_expr(np.dot(fexpr.dir, dvars), self.syms, vars)
            for fexpr in self.fexprs
        ])
        self.p.add_halfspace(polyhedra.Halfspace(+a, 0))

    def compute_coeffs_robust(self, *, output_flag=True):
        return self.p.compute_chebyshev_center(output_flag=output_flag)


