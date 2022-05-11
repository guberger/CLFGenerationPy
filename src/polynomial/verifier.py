import numpy as np
import z3
from src.z3utils import \
    create_z3syms_from_spsyms, \
    convert_spexpr_to_z3expr, \
    get_vars_from_z3model

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