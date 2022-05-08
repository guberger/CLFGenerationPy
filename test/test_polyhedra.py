import unittest
from math import sqrt
import numpy as np
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
            a = np.array([1. if j == i else 0. for j in range(nvar)])
            p.add_halfspace(polyhedra.Halfspace(-a, 0))

        _, rad = p.compute_chebyshev_center(output_flag=False)

        self.assertAlmostEqual(rad, sqrt(1/nvar)/(1 + sqrt(nvar)))

    def test_polyhedron(self):
        for nvar in range(1, 11):
            self._test_polyhedron(nvar)