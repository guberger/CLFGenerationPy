import unittest
from math import sqrt
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src import polyhedra
from src import generator

class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = sp.symbols('x y')
        x, y = syms
        exprs_val = [x**2, y**2]
        exprs_dir = [np.array([2*x, 0]), np.array([0, 2*y])]
        fexprs = [
            generator.FlowExpr(expr_val, expr_dir)
            for expr_val, expr_dir in zip(exprs_val, exprs_dir)
        ]
        cg = generator.CoeffsGenerator(syms, fexprs)

        wits_pos = [
            np.array([+1., 0.]),
            np.array([-1., 0.]),
            np.array([0., +1.]),
            np.array([0., -1.]),
        ]

        for vars in wits_pos:
            cg.add_constraint_pos(vars)

        A = np.array([
            [-0.5, -1.],
            [+1., -0.5]
        ])

        wits_lie_vars = [
            np.array([+1., +1.]),
            np.array([+1., -1.]),
            np.array([-1., +1.]),
            np.array([-1., -1.]),
        ]
        wits_lie = [(vars, A @ vars) for vars in wits_lie_vars]

        for vars, dvars in wits_lie:
            cg.add_constraint_lie(vars, dvars)

        self.cg = cg

        coeffs, r = cg.compute_coeffs_robust(output_flag=False)
        self.coeffs = coeffs
        self.r = r

    def test_coeffs_robust(self):
        self.assertAlmostEqual(self.coeffs[0], 1 - self.r)
        self.assertAlmostEqual(self.coeffs[1], 1 - self.r)
        self.assertAlmostEqual(self.r, 1/(1 + sqrt(10)/2))