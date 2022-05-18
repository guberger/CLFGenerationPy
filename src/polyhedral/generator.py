import numpy as np
from gurobipy import gurobipy, GRB

class Witness:
    def __init__(self, states, derivs) -> None:
        self.states = states
        self.derivs = derivs

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class VecsGenerator:
    def __init__(self, nvar, Gs, eta) -> None:
        self.nvar = nvar
        self.Gs = Gs
        self.eta = eta
        self.witnesses = []
        self.optvals = {G: -float('inf') for G in Gs}

    def add_witness(self, states, derivs):
        self.witnesses.append(Witness(states, derivs))

    def _compute_coeffs_robust(self, G, output_flag=True):
        nvec = len(self.witnesses)
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        vec_s = [
            model.addVars(
                self.nvar, lb=-1, ub=+1, name='c'
            )
            for i in range(nvec)]
        vecs = [np.array(vec_.values()) for vec_ in vec_s]
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for i in range(nvec):
            witness = self.witnesses[i]
            states_ = witness.states
            norm_state = np.linalg.norm(states_)
            states = states_/norm_state
            vec = vecs[i]
            @constraint(model, dot(dx, c) + ndx*δ ≤ 0)
            model.addConstr(np.dot(derivs, vec) +  >= 0)
            if self.eta > 0:
                model.addConstr(np.dot(states, vec) >= self.eta)
            derivs = witness.derivs/norm_state
            for j in range(nvec):
                if j == i:
                    continue
                vec2 = vecs[j]
                model.addConstr(np.dot(states, vec - vec2) >= 0)
            


    nxt = norm(xt)
    x = xt/nxt
    c = coeffs[i]
    if ϵ > 0
        @constraint(model, dot(x, c) ≥ ϵ)
    end
    for j = 1:M
        j == i && continue
        d = coeffs[j]
        @constraint(model, dot(x, c - d) ≥ 0)
    end
    for dxt in flow.grads
        dx = dxt/nxt
        ndx = norm(dx)
        @constraint(model, dot(dx, c) + ndx*δ ≤ 0)
        for j = 1:M
            j == i && continue
            d = coeffs[j]
            @constraint(model, dot(dx, d) - G*dot(x, c - d) + ndx*δ ≤ 0)
        end
    end
    return nothing
        model.setObjective(r, GRB.MAXIMIZE)
        model.optimize()

        if model.Status != 2:
            raise GeneratorError(
                'Chebyshev center status = %d.' % model.status
            )

        coeffs_opt = np.array([var.X for var in coeffs])
        r_opt = model.getObjective().getValue()
        return coeffs_opt, r_opt
        
        p = polyhedra.Polyhedron(ncoeff)
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            p.add_halfspace(polyhedra.Halfspace(+a, -1))
            p.add_halfspace(polyhedra.Halfspace(-a, -1))
        self.p = p



        return polyhedra.chebyshev_center(
            self.p, self.rmax, output_flag=output_flag
        )

        xt = flow.point
