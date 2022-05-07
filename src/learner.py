import numpy as np
import gurobipy as gpy
from gurobipy import GRB
import src.template as template

class Witness:
    def __init__(self, type, vars, mode, inps) -> None:
        self.type = type
        self.vars = vars
        self.mode = mode
        self.inps = inps

class WitnessError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class LearningProblem:
    def __init__(self, temp, msys, witnesses) -> None:
        self.temp = temp
        self.msys = msys
        self.witnesses = witnesses

    def add_witness(self, witness):
        self.witnesses.append(witness)

    def compute_coeffs(self, *, output_flag=True):
        model = gpy.Model('LearningProblem')
        model.setParam('OutputFlag', output_flag)
        syms = self.temp.syms
        exprs = self.temp.exprs
        nexpr = len(exprs)
        dexprs = np.array([template.diff_expr(expr, syms) for expr in exprs])
        coeffs_ = model.addVars(nexpr, lb=-1, ub=1, name='c')
        coeffs = np.array(coeffs_.values())
        radius = model.addVar(lb=-float('inf'), ub=1, name='r')

        for witness in self.witnesses:
            if witness.type == 'pos':
                vars = witness.vars
                a = np.array([
                    template.evalf_expr(expr, syms, vars)
                    for expr in exprs])
                model.addConstr(np.dot(a, coeffs) >= 0)
            elif witness.type == 'lie':
                vars = witness.vars
                sys = self.msys.syss[witness.mode]
                inps = witness.inps
                dvars = sys.flow(vars, inps)
                dfuncs = np.array([
                    template.evalf_expr(dexpr, syms, vars)
                    for dexpr in dexprs])
                a = np.array([np.dot(dfunc, dvars) for dfunc in dfuncs])
                model.addConstr(np.dot(a, coeffs) <= 0)
            else:
                raise WindowsError('Type not defined')
        
        model.setObjective(obj, GRB.MINIMIZE)


