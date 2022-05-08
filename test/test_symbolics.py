import unittest
from math import pi
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src import symbolics

class TestSymbolics(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = sp.symbols('x a')
        self.syms = syms
        exprs = [
            syms[1]*sp.sin(syms[0]),
            1/(syms[1] + syms[0]**2)
        ]

        nstep = 200
        xdom = np.linspace(0.0, 3*pi, nstep)
        adom = [1, 2, 3]

        for a in adom:
            for expr in exprs:
                points = [
                    symbolics.evalf_expr(expr, syms, np.array([x, a]))
                    for x in xdom
                ]
                plt.plot(xdom, points)
        
        dexprs = [symbolics.diff_expr(expr, syms) for expr in exprs]

        for a in adom:
            for dexpr in dexprs:
                points = [
                    symbolics.evalf_expr(
                        np.dot(dexpr, np.array([1, 1])),
                        syms,
                        np.array([x, a])
                    )
                    for x in xdom
                ]
                plt.plot(xdom, points)

        plt.savefig('./figs/plot_symfunc.png')
        plt.close()

        self.dexprs = dexprs

    def test_diff(self):
        x, a = self.syms
        self.assertEqual(self.dexprs[0][0], a*sp.cos(x))
        self.assertEqual(self.dexprs[0][1], sp.sin(x))
        self.assertEqual(self.dexprs[1][0], -2*x/(a + x**2)**2)
        self.assertEqual(self.dexprs[1][1], -1/(a + x**2)**2)