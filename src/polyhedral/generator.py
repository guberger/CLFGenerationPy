import numpy as np
from src import polyhedra

class Witness:
    def __init__(self, states, derivs) -> None:
        self.states = states
        self.derivs = derivs

class CoeffsGenerator:
    def __init__(self, G0, Gmax, alpha, eta) -> None:
        self.G0 = G0
        self.Gmax = Gmax
        self.alpha = alpha
        self.eta = eta
        self.witnesses = []
        self.rmax = 2

    def add_witness(self, states, derivs):
        self.witnesses.append(Witness(states, derivs))

    def _compute_coeffs_robust(self, G, rmin, output_flag=True):
        ncoeff = len(self.witnesses)
        
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