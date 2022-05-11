import unittest
from math import sqrt
import numpy as np
import sympy as sp
from src import generator

class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = sp.symbols('x y')
        x, y = syms
        expr_vals = [x**2, y**2]
        exprs_dirs = [np.array([2*x, 0]), np.array([0, 2*y])]
        cg = generator.CoeffsGenerator(syms, expr_vals, exprs_dirs)

        wits_pos = [
            np.array([+1., 0.]),
            np.array([-1., 0.]),
            np.array([0., +1.]),
            np.array([0., -1.]),
        ]

        for states in wits_pos:
            cg.add_constraint_pos(states)

        A = np.array([
            [-0.5, -1.],
            [+1., -0.5]
        ])

        wits_lie_states = [
            np.array([+1., +1.]),
            np.array([+1., -1.]),
            np.array([-1., +1.]),
            np.array([-1., -1.]),
        ]
        wits_lie = [(states, A @ states) for states in wits_lie_states]

        for states, derivs in wits_lie:
            cg.add_constraint_lie(states, derivs)

        self.cg = cg

        coeffs, r = cg.compute_coeffs_robust(output_flag=False)
        self.coeffs = coeffs
        self.r = r

    def test_coeffs_robust(self):
        self.assertAlmostEqual(self.coeffs[0], 1 - self.r)
        self.assertAlmostEqual(self.coeffs[1], 1 - self.r)
        self.assertAlmostEqual(self.r, 1/(1 + sqrt(10)/2))
        self.assertTrue(self.cg.p.contains(self.coeffs))