from math import pi
import unittest
import numpy as np
import sympy as sp
from src import polyhedra
from src.polyhedra import Halfspace
from src import symbolics
from src import learner

class TestLearner(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_small(self):
        x, y = syms_state = np.array(sp.symbols('x y'))
        exprs_term = [x, y, x**2, x*y, y**2]
        u, v = syms_input = np.array(sp.symbols('u, v'))

        exprs_field = np.array([-u*x, 2*x/pi + v])
        dom_state = polyhedra.Polyhedron(2)
        dom_state.add_halfspace(Halfspace(np.array([-1, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([+1, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([0, -1]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, +1]), -1))
        dom_input = polyhedra.Polyhedron(2)
        dom_input.add_halfspace(Halfspace(np.array([-1, 0]), +0.9))
        dom_input.add_halfspace(Halfspace(np.array([+1, 0]), -1.1))
        dom_input.add_halfspace(Halfspace(np.array([0, -1]), -2))
        dom_input.add_halfspace(Halfspace(np.array([0, +1]), -2))
        system = learner.System(exprs_field, dom_state, dom_input)

        prob = learner.LearningProblem(
            syms_state, syms_input, [system], exprs_term
        )

        exprs_uopt = np.array([1, -2*x/pi - y])
        demo_func = lambda states : \
            np.array([
                symbolics.evalf_expr(expr_uopt, syms_state, states)
                for expr_uopt in exprs_uopt
            ])

        rmin = 0.1

        prob.learn_CLF(rmin, demo_func, 1e-2)

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
        dom_state = polyhedra.Polyhedron(4)
        dom_state.add_halfspace(Halfspace(np.array([-1, 0, 0, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([+1, 0, 0, 0]), -pi/2))
        dom_state.add_halfspace(Halfspace(np.array([0, -1, 0, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, +1, 0, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, -1, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, +1, 0]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, 0, -1]), -1))
        dom_state.add_halfspace(Halfspace(np.array([0, 0, 0, +1]), -1))
        dom_input = polyhedra.Polyhedron(2)
        dom_input.add_halfspace(Halfspace(np.array([-1, 0]), +0.9))
        dom_input.add_halfspace(Halfspace(np.array([+1, 0]), -1.1))
        dom_input.add_halfspace(Halfspace(np.array([0, -1]), -2))
        dom_input.add_halfspace(Halfspace(np.array([0, +1]), -2))
        system = learner.System(exprs_field, dom_state, dom_input)

        prob = learner.LearningProblem(
            syms_state, syms_input, [system], exprs_term
        )

        exprs_uopt = np.array([1, -2*x1/pi - x4])
        demo_func = lambda states : \
            np.array([
                symbolics.evalf_expr(expr_uopt, syms_state, states)
                for expr_uopt in exprs_uopt
            ])

        rmin = 0.1

        prob.learn_CLF(rmin, demo_func, 1e-2)