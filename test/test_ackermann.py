import unittest
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from src.ackermann import make_field, make_local_affine_field
from src.dynamics import trajectory, _rk4_mult
from src.fields import open_loop

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

        nom_field = make_field(L)
        nom_cofield = open_loop(nom_field, finputs)
        traj_conom = trajectory(nom_cofield, states_init, tdom)
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
            loc_field = make_local_affine_field(states, L)
            F = open_loop(loc_field, finputs_augm).flow
            states = _rk4_mult(states, F, t0, t1, 5)
        
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