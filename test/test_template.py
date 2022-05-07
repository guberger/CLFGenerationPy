import unittest
from math import pi
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src import template

class TestTemplate(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = np.array(sp.symbols('x a'))
        self.syms = syms
        expr = syms[1]*sp.sin(syms[0])

        nstep = 200
        xdom = np.linspace(0.0, 3*pi, nstep)
        adom = [1, 2, 3]

        for a in adom:
            points = [template.evalf_expr(
                expr, syms, np.array([x, a]))
                for x in xdom]
            plt.plot(xdom, points)
        
        Dexpr = np.dot(template.diff_expr(expr, syms), np.array([1, 1]))

        for a in adom:
            points = [template.evalf_expr(
                Dexpr, syms, np.array([x, a]))
                for x in xdom]
            plt.plot(xdom, points)

        exprs = np.array([sp.sin(syms[0]), 0.5*syms[1]])
        temp = template.Template(syms, exprs)
        coeffs = np.array([1, 2])
        Vexpr = temp.build_expr(coeffs)

        for a in adom:
            points = [template.evalf_expr(
                Vexpr, syms, np.array([x, a]))
                for x in xdom]
            plt.plot(xdom, points)

        plt.savefig('./figs/plot_symfunc.png')
        plt.close()

        self.Dexpr = Dexpr
        self.Vexpr = Vexpr

    def test_diff(self):
        x, a = self.syms
        self.assertEqual(self.Dexpr, a*sp.cos(x) + sp.sin(x))

    def test_build(self):
        x, a = self.syms
        self.assertEqual(self.Vexpr, sp.sin(x) + 1.0*a)