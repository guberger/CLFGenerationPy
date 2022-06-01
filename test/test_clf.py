from math import pi, sqrt
import unittest
import numpy as np
import sympy as sp
import gurobipy
from src.polyhedra import Halfspace, Polyhedron
from src.symbolics import evalf_expr
from src.systems import System
from src.clf import \
    Constraint, Generator, \
    VerifierSimple, VerifierParam, \
    Learner, LearnerError

class TestGenerator(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        syms = sp.symbols('x y')
        x, y = syms
        expr_vals = [x**2, y**2]
        exprs_dirs = [np.array([2*x, 0]), np.array([0, 2*y])]
        gen = Generator(syms, expr_vals, exprs_dirs)

        wits_pos = [
            np.array([+1., 0.]),
            np.array([-1., 0.]),
            np.array([0., +1.]),
            np.array([0., -1.]),
        ]

        for states in wits_pos:
            gen.add_constraint_pos(states)

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
            gen.add_constraint_lie(states, derivs)

        self.gen = gen

        coeffs, r = gen.compute_coeffs(output_flag=False)
        self.coeffs = coeffs
        self.r = r

    def test_coeffs(self):
        self.assertAlmostEqual(self.coeffs[0], 1 - self.r)
        self.assertAlmostEqual(self.coeffs[1], 1 - self.r)
        self.assertAlmostEqual(self.r, 1/(1 + sqrt(10)/2))


class TestChebyshev(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        gurobipy.Model('')
        self.rmax = 100
        
    def _test_chebyshev(self, nvar):
        expr_vals = np.ones(nvar)
        exprs_dirs = np.ones(nvar)
        gen = Generator([], expr_vals, exprs_dirs)
        gen.rmax = self.rmax

        gen.constraints.append(Constraint(np.ones(nvar), -1))

        for i in range(nvar):
            a = np.array([1. if j == i else 0. for j in range(nvar)])
            gen.constraints.append(Constraint(-a, 0))

        vars, rad = gen.compute_coeffs(output_flag=False)

        self.assertAlmostEqual(rad, sqrt(1/nvar)/(1 + sqrt(nvar)))
        self.assertAlmostEqual(sum(vars), rad*nvar)

    def test_chebyshev(self):
        for nvar in range(1, 11):
            self._test_chebyshev(nvar)

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

class TestLearner(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_small(self):
        x, y = syms_state = np.array(sp.symbols('x y'))
        exprs_term = [x, y, x**2, x*y, y**2]
        u, v = syms_input = np.array(sp.symbols('u, v'))

        exprs_field = np.array([-u*x, 2*x/pi + v])
        dom_state = Polyhedron(2)
        dom_state.add_halfspace(Halfspace(np.array([-1, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([+1, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([0, -1]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, +1]), -1))
        dom_input = Polyhedron(2)
        dom_input.add_halfspace(Halfspace(np.array([-1, 0]), +0.9))
        dom_input.add_halfspace(Halfspace(np.array([+1, 0]), -1.1))
        dom_input.add_halfspace(Halfspace(np.array([0, -1]), -2))
        dom_input.add_halfspace(Halfspace(np.array([0, +1]), -2))
        system = System(
            syms_state, syms_input, exprs_field, dom_state, dom_input
        )

        lear = Learner(system, exprs_term)

        exprs_uopt = np.array([1, -2*x/pi - y])
        demo_func = lambda states : \
            np.array([
                evalf_expr(expr_uopt, syms_state, states)
                for expr_uopt in exprs_uopt
            ])

        rmin = 0.1

        lear.learn_CLF(rmin, demo_func, 1e-2)

    def test_medium(self):
        x1, x2, x3, x4 = syms_state = np.array(sp.symbols('x1 x2 x3 x4'))
        exprs_term = [
            x1, x2, x3, x4,
            x1**2, x1*x2, x1*x3, x1*x4,
            x2**2, x2*x3, x2*x4,
            x3**2, x3*x4,
            x4**2
        ]
        u1, u2 = syms_input = np.array(sp.symbols('u1 u2'))

        exprs_field = np.array([-u1*x1, -x2, -x3, 2*x1/pi + u2])
        dom_state = Polyhedron(4)
        dom_state.add_halfspace(Halfspace(np.array([-1, 0, 0, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([+1, 0, 0, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([0, -1, 0, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, +1, 0, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, -1, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, +1, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, 0, -1]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, 0, +1]), -1))
        dom_input = Polyhedron(2)
        dom_input.add_halfspace(Halfspace(np.array([-1, 0]), +0.9))
        dom_input.add_halfspace(Halfspace(np.array([+1, 0]), -1.1))
        dom_input.add_halfspace(Halfspace(np.array([0, -1]), -2))
        dom_input.add_halfspace(Halfspace(np.array([0, +1]), -2))
        system = System(
            syms_state, syms_input, exprs_field, dom_state, dom_input
        )

        lear = Learner(system, exprs_term)
        lear.iter_max = 10

        exprs_uopt = np.array([1, -2*x1/pi - x4])
        demo_func = lambda states : \
            np.array([
                evalf_expr(expr_uopt, syms_state, states)
                for expr_uopt in exprs_uopt
            ])

        rmin = 0.1

        self.assertRaises(
            LearnerError,
            lear.learn_CLF,
            rmin, demo_func, 1e-2,
        )