import numpy as np

def _rk4_single(vars, sys, t, dt):
    dvars1 = sys.flow(vars, t)
    varst = vars + dvars1*(dt/2)
    dvars2 = sys.flow(varst, t + dt/2)
    varst = vars + dvars2*(dt/2)
    dvars3 = sys.flow(varst, t + dt/2)
    varst = vars + dvars3*dt
    dvars4 = sys.flow(varst, t + dt)
    return vars + (dvars1 + dvars2*2 + dvars3*2 + dvars4)*(dt/6)

def _rk4_mult(vars, sys, t0, t1, nsub):
    dt = (t1 - t0)/nsub
    for i in range(nsub):
        t = t0 + i*dt
        vars = _rk4_single(vars, sys, t, dt)
    return vars

def trajectory(flow, vars_init, tdom, nsub=1):
    nstep = len(tdom)
    traj = []
    vars = np.array(vars_init) # makes copy
    for k in range(nstep):
        traj.append(vars)
        if k == nstep - 1:
            break
        t0, t1 = tdom[k:k+2]
        vars = _rk4_mult(vars, flow, t0, t1, nsub)
    return traj