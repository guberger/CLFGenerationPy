import numpy as np
from math import cos, sin
import src.systems as sys

def _field(vars, inps, L):
    v, alpha = vars[2:4]
    u1, u2 = inps
    return np.array([v*cos(alpha), v*sin(alpha), u1, v/L*u2])

def make_system(L):
    field = lambda vars, inps : _field(vars, inps, L)
    return sys.GenericSystem(field)

def _local_affine_maps(vars, L):
    v, alpha = vars[2:4]
    A0 = np.array([
        [0, 0, cos(alpha), -v*sin(alpha)],
        [0, 0, sin(alpha), +v*cos(alpha)],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    b0 = np.array([v*cos(alpha), v*sin(alpha), 0, 0]) - np.dot(A0, vars)
    A1 = np.zeros((4, 4))
    b1 = np.array([0, 0, 1, 0])
    A2 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1/L, 0]
    ])
    b2 = np.zeros(4)
    return [
        sys.AffineMap(A0, b0),
        sys.AffineMap(A1, b1),
        sys.AffineMap(A2, b2)
    ]

def make_local_affine_system(vars, L):
    aff_maps = _local_affine_maps(vars, L)
    return sys.AffineSystem(aff_maps)