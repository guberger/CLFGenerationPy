# importing sys
import unittest
from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from src import ackermann
from src import dynamics
from src import systems

class TestAckermann(unittest.TestCase):
    def test_plot_trajectories(self):
        fig, ax_ = plt.subplots(3, 2)

        nstep = 200
        tdom = np.linspace(0.0, 20.0, nstep)

        finps = lambda t : np.array([
            0.1*(1 + 0.5*cos(t)),
            0.1*(1 + 0.5*sin(t))])

        traj_inps = [finps(t) for t in tdom]

        for k in range(2):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i, j].plot(tdom, [inps[k] for inps in traj_inps])

        L = 0.33

        nom_sys = ackermann.make_system(L)
        nom_cosys = systems.open_loop(nom_sys, finps)
        varsinit = np.array([0, 0, 0, 0])
        traj_conom = dynamics.trajectory(nom_cosys, varsinit, tdom)
        finps_augm = lambda t : np.concatenate((np.array([1]), finps(t)))

        for k in range(4):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i + 1, j].plot(tdom, [vars[k] for vars in traj_conom])

        traj_coloc = []
        vars = np.array(varsinit)

        for k in range(nstep):
            traj_coloc.append(vars)
            if k == nstep - 1:
                break
            t0, t1 = tdom[k:k+2]
            loc_sys = ackermann.make_local_affine_system(vars, L)
            loc_cosys = systems.open_loop(loc_sys, finps_augm)
            vars = dynamics._rk4_mult(vars, loc_cosys, t0, t1, 5)
        
        for k in range(4):
            i, j = np.unravel_index(k, ax_.shape)
            ax_[i + 1, j].plot(tdom, [vars[k] for vars in traj_coloc])

        plt.savefig('./figs/plot_trajectories.png')
        plt.close()