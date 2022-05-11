import numpy as np
from src import symbolics
from src.polynomial import generator
from src.polynomial import verifier

class System:
    def __init__(self, exprs_field, dom_state, dom_input) -> None:
        self.exprs_field = exprs_field
        self.dom_state = dom_state
        self.dom_input = dom_input

class LearnerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class LearningProblem:
    def __init__(self, syms_state, syms_input, systems, exprs_term) -> None:
        self.syms_state = syms_state
        self.syms_input = syms_input
        self.systems = systems
        self.exprs_term = exprs_term
        self.iter_max = 1_00

    def learn_CLF(self, rmin, demo_func, eps):
        syms_state = self.syms_state
        syms_input = self.syms_input
        expr_vals = self.exprs_term
        exprs_dirs = [
            symbolics.diff_expr(expr_val, syms_state)
            for expr_val in expr_vals]
        cg = generator.CoeffsGenerator(syms_state, expr_vals, exprs_dirs)

        iter = 0

        while True:
            iter = iter + 1
            if iter > self.iter_max:
                raise LearnerError('Max iter excedeed: ' + str(iter))

            coeffs, r = cg.compute_coeffs_robust(output_flag=False)
            print('\nIter %5d:\n%s\n%s' % (iter, coeffs, r))

            if r < eps:
                raise LearnerError('Radius too small: ' + str(r))

            Vexpr = np.dot(expr_vals, coeffs)
            dVexprs = symbolics.diff_expr(Vexpr, syms_state)
            res = True

            print('Verify pos...', end='', flush=True)
            for system in self.systems:
                verif = verifier.VerifierSimple(
                    syms_state, system.dom_state, rmin
                )
                res, states = verif.check_expr(Vexpr)
                if not res:
                    print(' CE found: %s' % states)
                    cg.add_constraint_pos(states)
                    break

            if not res:
                continue
            else:
                print(' No CE found')
            
            print('Verify lie...', end='', flush=True)
            for system in self.systems:
                verif = verifier.VerifierParam(
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
                        symbolics.evalf_expr(expr_field, syms, vars)
                        for expr_field in system.exprs_field
                    ])                        
                    cg.add_constraint_lie(states, derivs)
                    break
            
            if res:
                print(' No CE found')
                print('Valid CLF: terminated')
                return coeffs