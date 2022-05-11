import unittest
import numpy as np
import sympy as sp
from src.symbolics import evalf_expr
from src.polyhedra import Halfspace, Polyhedron
from src.polynomial.verifier import VerifierSimple, VerifierParam

class TestVerifierSimple(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = np.array(sp.symbols('x y'))
        x, y = syms
        p = Polyhedron(2)
        for a1, a2 in zip((-1, +1), (-1, +1)):
            p.add_halfspace(Halfspace(np.array([a1, a2]), -1))

        rmin = 0.1
        verif = VerifierSimple(syms, p, rmin)

        self.syms = syms
        self.p = p

        self.expr1 = (x + y)**2
        self.res1, self.vars1 = verif.check_expr(self.expr1)
        self.expr2 = (x + y)**2 + (x - y)**2
        self.res2, self.vars2 = verif.check_expr(self.expr2)

    def test_check(self):
        self.assertFalse(self.res1)
        val_expr1 = evalf_expr(self.expr1, self.syms, self.vars1)
        self.assertTrue(self.p.contains(self.vars2))
        self.assertLessEqual(round(val_expr1, 1), 0)
        self.assertTrue(self.res2)

class TestVerifierParam(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = np.array(sp.symbols('x y'))
        syms_param = np.array(sp.symbols('u v'))
        x, y = syms
        p = Polyhedron(2)
        for a1, a2 in zip((-1, +1), (-1, +1)):
            p.add_halfspace(Halfspace(np.array([a1, a2]), -1))

        rmin = 0.1

        u, v = syms_param
        p_param = Polyhedron(2)
        p_param.add_halfspace(Halfspace(np.array([-1, 0]), 0))
        p_param.add_halfspace(Halfspace(np.array([0, -1]), 0))
        p_param.add_halfspace(Halfspace(np.array([+1, +1]), -1))

        verif = VerifierParam(syms, p, rmin, syms_param, p_param)

        self.syms = syms
        self.syms_param = syms_param
        self.expr1 = (x + y)**2 - 0.25 + u*v
        self.res1, self.vars1 = verif.check_expr(self.expr1)
        self.expr2 = (x + u*y)**2 + (x - v*y)**2
        self.res2, self.vars2 = verif.check_expr(self.expr2)

    def test_check(self):
        self.assertFalse(self.res1)
        self.assertTrue(self.res2)