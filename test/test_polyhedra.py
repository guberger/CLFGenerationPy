import unittest
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import gurobipy
from src import polyhedra

class TestPolyhedra(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        gurobipy.Model('')
        self.rmax = 100

    def _test_polyhedron(self, nvar):
        p = polyhedra.Polyhedron(nvar, self.rmax)

        h = polyhedra.Halfspace(np.ones(nvar), -1)
        p.add_halfspace(h)

        for i in range(nvar):
            a = np.zeros(nvar)
            a[i] = -1
            h = polyhedra.Halfspace(a, 0)
            p.add_halfspace(h)

        _, rad = p.compute_chebyshev_center(output_flag=False)

        self.assertAlmostEqual(rad, sqrt(1/nvar)/(1 + sqrt(nvar)))

    def test_polyhedron(self):
        for nvar in range(1, 11):
            self._test_polyhedron(nvar)