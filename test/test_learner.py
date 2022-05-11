from math import pi
import unittest
import numpy as np
import sympy as sp
from src import polyhedra
from src import symbolics
from src import learner

class TestLearner(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        x, y = syms_state = np.array(sp.symbols('x y'))
        exprs_term = [x, y, x**2, x*y, y**2]
        u, v = syms_input = np.array(sp.symbols('u, v'))

        exprs_field = np.array([-u*x, 2*x/pi + v])
        dom_state = polyhedra.Polyhedron(2)
        dom_state.add_halfspace(polyhedra.Halfspace(np.array([-1, 0]), -pi/2))
        dom_state.add_halfspace(polyhedra.Halfspace(np.array([+1, 0]), -pi/2))
        dom_state.add_halfspace(polyhedra.Halfspace(np.array([0, -1]), -1))
        dom_state.add_halfspace(polyhedra.Halfspace(np.array([0, +1]), -1))
        dom_input = polyhedra.Polyhedron(2)
        dom_input.add_halfspace(polyhedra.Halfspace(np.array([-1, 0]), +0.9))
        dom_input.add_halfspace(polyhedra.Halfspace(np.array([+1, 0]), -1.1))
        dom_input.add_halfspace(polyhedra.Halfspace(np.array([0, -1]), -2))
        dom_input.add_halfspace(polyhedra.Halfspace(np.array([0, +1]), -2))
        system = learner.System(exprs_field, dom_state, dom_input)

        prob = learner.LearningProblem(
            syms_state, syms_input, [system], exprs_term
        )

        expr_uopt = -2*x/pi - y
        demo_func = lambda states : \
            np.array([
                1.0,
                symbolics.evalf_expr(expr_uopt, syms_state, states)
            ])

        rmin = 0.1

        prob.learn_CLF(rmin, demo_func, 1e-2)

    def test_coeffs_robust(self):
        pass