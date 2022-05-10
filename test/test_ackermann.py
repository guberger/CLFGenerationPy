import unittest
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from src import ackermann
from src import dynamics
from src import systems

class TestAckermann(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        _, ax_ = plt.subplots(3, 2)

        nstep = 200
        tdom = np.linspace(0.0, 20.0, nstep)

        finputs = lambda t : np.array([
            0.1*(1 + 0.5*cos(t)),
            0.1*(1 + 0.5*sin(t))
        ])

        traj_inputs = [finputs(t) for t in tdom]

        for k in range(2):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i, j].plot(tdom, [inputs[k] for inputs in traj_inputs])

        L = 0.33
        states_init = np.zeros(4)

        nom_sys = ackermann.make_system(L)
        nom_cosys = systems.open_loop(nom_sys, finputs)
        traj_conom = dynamics.trajectory(nom_cosys, states_init, tdom)
        finputs_augm = lambda t : np.concatenate((np.array([1]), finputs(t)))

        for k in range(4):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i + 1, j].plot(tdom, [states[k] for states in traj_conom])

        traj_coloc = []
        states = np.array(states_init)

        for k in range(nstep):
            traj_coloc.append(states)
            if k == nstep - 1:
                break
            t0, t1 = tdom[k:k+2]
            loc_sys = ackermann.make_local_affine_system(states, L)
            loc_cosys = systems.open_loop(loc_sys, finputs_augm)
            states = dynamics._rk4_mult(states, loc_cosys, t0, t1, 5)
        
        for k in range(4):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i + 1, j].plot(tdom, [states[k] for states in traj_coloc])

        plt.savefig('./figs/plot_trajectories.png')
        plt.close()

        self.traj_conom = traj_conom
        self.traj_coloc = traj_coloc

    def test_deviation(self):
        devs = [
            np.linalg.norm(states_nom - states_loc)
            for states_nom, states_loc in zip(self.traj_conom, self.traj_coloc)
        ]
        self.assertLess(max(devs), 0.002)