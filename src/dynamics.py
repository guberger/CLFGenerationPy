import numpy as np

def _rk4_single(states, sys, t, dt):
    derivs1 = sys.flow(states, t)
    statest = states + derivs1*(dt/2)
    derivs2 = sys.flow(statest, t + dt/2)
    statest = states + derivs2*(dt/2)
    derivs3 = sys.flow(statest, t + dt/2)
    statest = states + derivs3*dt
    derivs4 = sys.flow(statest, t + dt)
    return states + (derivs1 + derivs2*2 + derivs3*2 + derivs4)*(dt/6)

def _rk4_mult(states, sys, t0, t1, nsub):
    dt = (t1 - t0)/nsub
    for i in range(nsub):
        t = t0 + i*dt
        states = _rk4_single(states, sys, t, dt)
    return states

def trajectory(flow, states_init, tdom, *, nsub=1):
    nstep = len(tdom)
    traj = []
    states = np.array(states_init) # makes copy
    for k in range(nstep):
        traj.append(states)
        if k == nstep - 1:
            break
        t0, t1 = tdom[k:k+2]
        states = _rk4_mult(states, flow, t0, t1, nsub)
    return traj