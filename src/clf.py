import numpy as np
from gurobipy import gurobipy, GRB
import z3
from src.symbolics import evalf_expr, diff_expr
from src.z3utils import \
    create_z3syms_from_spsyms, \
    convert_spexpr_to_z3expr, \
    get_vars_from_z3model

## Generator

class Constraint:
    def __init__(self, a, b) -> None:
        self.a = a
        self.b = b

class GeneratorError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Generator:
    def __init__(self, syms, expr_vals, exprs_dirs) -> None:
        self.syms = syms
        ncoeff = len(expr_vals)
        assert len(exprs_dirs) == ncoeff
        self.ncoeff = ncoeff
        self.expr_vals = expr_vals
        self.exprs_dirs = exprs_dirs
        self.rmax = 2
        self.constraints = []
        for i in range(ncoeff):
            a = np.array([1 if j == i else 0 for j in range(ncoeff)])
            self.constraints.append(Constraint(+a, -1))
            self.constraints.append(Constraint(-a, -1))

    def add_constraint_pos(self, states):
        a = np.array([
            evalf_expr(expr_val, self.syms, states)
            for expr_val in self.expr_vals
        ])
        self.constraints.append(Constraint(-a, 0))

    def add_constraint_lie(self, states, derivs):
        a = np.array([
            evalf_expr(np.dot(exprs_dir, derivs), self.syms, states)
            for exprs_dir in self.exprs_dirs
        ])
        self.constraints.append(Constraint(+a, 0))

    def compute_coeffs(self, *, output_flag=True):
        model = gurobipy.Model('Robust coeffs')
        model.setParam('OutputFlag', output_flag)
        coeffs_ = model.addVars(
            self.ncoeff, lb=-self.rmax, ub=+self.rmax, name='c'
        )
        coeffs = np.array(coeffs_.values())
        r = model.addVar(lb=-float('inf'), ub=self.rmax, name='r')

        for h in self.constraints:
            a = h.a
            b = h.b
            na = np.linalg.norm(a)
            model.addConstr(np.dot(a, coeffs) + b + na*r <= 0)

        model.setObjective(r, GRB.MAXIMIZE)
        model.optimize()

        if model.Status != 2:
            raise GeneratorError(
                'Chebyshev center status = %d.' % model.status
            )

        coeffs_opt = np.array([var.X for var in coeffs])
        r_opt = model.getObjective().getValue()
        return coeffs_opt, r_opt

## Verifier

class VerifierSimple:
    def __init__(self, syms, p, rmin) -> None:
        self.spsyms = syms
        self.p = p
        self.rmin = rmin

    def check_expr(self, expr):
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        z3syms, syms_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms)

        for h in self.p.halfspaces:
            con = np.dot(h.a, z3syms) + h.beta <= 0
            solver.add(con)

        if self.rmin != None:
            solver.add(np.dot(z3syms, z3syms) >= self.rmin*self.rmin)

        z3expr = convert_spexpr_to_z3expr(syms_map, expr)
        solver.add(z3expr <= 0)

        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            vars_ = get_vars_from_z3model(syms_map, model)
            vars = np.array([vars_[sym.name] for sym in self.spsyms])
            return False, vars
        else:
            return True, np.zeros(len(self.spsyms))

class VerifierParam:
    def __init__(self, syms, p, rmin, syms_param, p_param) -> None:
        self.spsyms = syms
        self.p = p
        self.rmin = rmin
        self.spsyms_param = syms_param
        self.p_param = p_param

    def check_expr(self, expr):
        ctx = z3.Context()
        solver = z3.Solver(ctx=ctx)
        z3syms, syms_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms)
        z3syms_param, syms_param_map = \
            create_z3syms_from_spsyms(ctx, self.spsyms_param)

        for h in self.p.halfspaces:
            con = np.dot(h.a, z3syms) + h.beta <= 0
            solver.add(con)

        if self.rmin != None:
            solver.add(np.dot(z3syms, z3syms) >= self.rmin*self.rmin)

        syms_all_map = syms_map | syms_param_map
        z3expr = convert_spexpr_to_z3expr(syms_all_map, expr)

        cons_param_ = []
        for h in self.p_param.halfspaces:
            con = np.dot(h.a, z3syms_param) + h.beta <= 0
            cons_param_.append(con)

        cons_param = z3.And(cons_param_)
        fml = z3.Implies(cons_param, z3expr <= 0)
        solver.add(z3.ForAll(z3syms_param, fml))

        res = solver.check()
        if res == z3.sat:
            model = solver.model()
            vars_ = get_vars_from_z3model(syms_map, model)
            vars = np.array([vars_[sym.name] for sym in self.spsyms])
            return False, vars
        else:
            return True, np.zeros(len(self.spsyms))

## Learner

class LearnerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)    

class Learner:
    def __init__(self, system, exprs_term) -> None:
        self.system = system
        self.exprs_term = exprs_term
        self.iter_max = 1_000

    def learn_CLF(self, rmin, demo_func, eps):
        system = self.system
        syms_state = system.syms_state
        syms_input = system.syms_input
        expr_vals = self.exprs_term
        exprs_dirs = [
            diff_expr(expr_val, syms_state)
            for expr_val in expr_vals
        ]
        gen = Generator(syms_state, expr_vals, exprs_dirs)

        iter = 0

        while True:
            iter = iter + 1
            if iter > self.iter_max:
                raise LearnerError('Max iter excedeed: ' + str(iter))

            coeffs, r = gen.compute_coeffs(output_flag=False)
            print('\nIter %5d:\n%s\n%s' % (iter, coeffs, r))

            if r < eps:
                raise LearnerError('Radius too small: ' + str(r))

            Vexpr = np.dot(expr_vals, coeffs)
            dVexprs = diff_expr(Vexpr, syms_state)
            res = True

            print('Verify pos...', end='', flush=True)
            verif = VerifierSimple(
                syms_state, system.dom_state, rmin
            )
            res, states = verif.check_expr(Vexpr)
            if not res:
                print(' CE found: %s' % states)
                gen.add_constraint_pos(states)
                continue
            else:
                print(' No CE found')
            
            print('Verify lie...', end='', flush=True)
            verif = VerifierParam(
                syms_state, system.dom_state, rmin,
                syms_input, system.dom_input
            )
            dVfexpr = -np.dot(dVexprs, system.exprs_field)
            res, states = verif.check_expr(dVfexpr)
            if not res:
                print(' CE found: %s' % states)
                inputs = demo_func(states)
                syms = np.concatenate((syms_state, syms_input))
                vars = np.concatenate((states, inputs))
                derivs = np.array([
                    evalf_expr(expr_field, syms, vars)
                    for expr_field in system.exprs_field
                ])                        
                gen.add_constraint_lie(states, derivs)
            else:
                print(' No CE found')
                print('Valid CLF: terminated')
                return coeffs