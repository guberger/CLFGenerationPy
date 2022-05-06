from cmath import exp
import numpy as np
import sympy as sp

def evalf_expr(expr, syms, vals):
    valuation = {sym:val for sym, val in zip(syms, vals)}
    return expr.evalf(subs=valuation)

def diff_expr(expr, syms):
    return np.array([sp.diff(expr, sym) for sym in syms])

class Template:
    def __init__(self, syms, exprs) -> None:
        self.syms = np.array(syms)
        self.exprs = np.array(exprs)

    def build_expr(self, coeffs):
        return np.dot(coeffs, self.exprs)
    

        