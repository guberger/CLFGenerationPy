import numpy as np
from src.symbolics import evalf_expr

class System:
    def __init__(
            self, syms_state, syms_input, exprs_field, dom_state, dom_input
        ) -> None:
        self.syms_state = syms_state
        self.syms_input = syms_input
        self.exprs_field = exprs_field
        self.dom_state = dom_state
        self.dom_input = dom_input

def _rk4_single(states, F, time, dtime):
    derivs1 = F(states, time)
    statest = states + derivs1*(dtime/2)
    derivs2 = F(statest, time + dtime/2)
    statest = states + derivs2*(dtime/2)
    derivs3 = F(statest, time + dtime/2)
    statest = states + derivs3*dtime
    derivs4 = F(statest, time + dtime)
    return states + (derivs1 + derivs2*2 + derivs3*2 + derivs4)*(dtime/6)

def _rk4_mult(states, F, time0, time1, nsub):
    dtime = (time1 - time0)/nsub
    for i in range(nsub):
        time = time0 + i*dtime
        states = _rk4_single(states, F, time, dtime)
    return states

def trajectory(system, sym_time, exprs_input, states_init, dom_time, *, nsub=1):
    nstep = len(dom_time)
    syms_state = system.syms_state
    syms_input = system.syms_input
    exprs_field = system.exprs_field
    U = lambda time : np.array([
        evalf_expr(expr_input, sym_time, (time,)) for expr_input in exprs_input
    ])
    syms = np.concatenate((syms_state, syms_input))
    F = lambda states, time : np.array([
        evalf_expr(expr_field, syms, np.concatenate((states, U(time))))
        for expr_field in exprs_field
    ])
    states_list = []
    states = np.array(states_init) # makes copy
    for k in range(nstep):
        states_list.append(states)
        if k == nstep - 1:
            break
        time0, time1 = dom_time[k:k+2]
        states = _rk4_mult(states, F, time0, time1, nsub)
    return states_list