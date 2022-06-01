import unittest
from math import cos, sin
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from src.polyhedra import Polyhedron
from src.systems import System, trajectory
from src.symbolics import evalf_expr

class TestSystems(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        x, y = syms_state = np.array(sp.symbols('x y'))
        u, v = syms_input = np.array(sp.symbols('u, v'))
        exprs_field = np.array([-u, 2*x + v])
        dom_state = Polyhedron(2)
        dom_input = Polyhedron(2)
        system = System(
            syms_state, syms_input, exprs_field, dom_state, dom_input
        )

        t, = sym_time = np.array([sp.symbols('t')])
        exprs_input = np.array([sp.cos(t), sp.S(1)])

        nstep = 200
        dom_time = np.linspace(0.0, 20.0, nstep)
        states_init = np.zeros(2)

        inputs_list = [
            [
                evalf_expr(expr_input, sym_time, (time,))
                for expr_input in exprs_input
            ]
            for time in dom_time
        ]
        states_list = trajectory(
            system, sym_time, exprs_input, states_init, dom_time, nsub=3
        )

        _, ax_ = plt.subplots(4, 1)

        for k in range(2):
            input_list = [input[k] for input in inputs_list]
            ax_[k].plot(dom_time, input_list)
            state_list = [state[k] for state in states_list]
            ax_[2 + k].plot(dom_time, state_list)
       
        plt.savefig('./figs/plot_test_systems.png')
        plt.close()

        self.dom_time = dom_time
        self.states_list = states_list

    def test_deviation(self):
        states_ref = [
            np.array([-sin(time), 2*cos(time) + time - 2])
            for time in self.dom_time
        ]
        devs = [
            np.linalg.norm(states - states_ref)
            for states, states_ref in zip(self.states_list, states_ref)
        ]
        self.assertAlmostEqual(max(devs), 0)