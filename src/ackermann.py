import numpy as np
from math import cos, sin
from src.fields import GenericField, AffineMap, AffineField

def _F(states, inputs, L):
    v, alpha = states[2:4]
    u1, u2 = inputs
    return np.array([v*cos(alpha), v*sin(alpha), u1, v/L*u2])

def make_field(L):
    F = lambda states, inputs : _F(states, inputs, L)
    return GenericField(F)

def _aff_maps(states, L):
    v, alpha = states[2:4]
    A0 = np.array([
        [0, 0, cos(alpha), -v*sin(alpha)],
        [0, 0, sin(alpha), +v*cos(alpha)],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    b0 = np.array([v*cos(alpha), v*sin(alpha), 0, 0]) - np.dot(A0, states)
    A1 = np.zeros((4, 4))
    b1 = np.array([0., 0., 1., 0.])
    A2 = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1/L, 0]
    ])
    b2 = np.zeros(4)
    return [
        AffineMap(A0, b0),
        AffineMap(A1, b1),
        AffineMap(A2, b2)
    ]

def make_local_affine_field(states, L):
    aff_maps = _aff_maps(states, L)
    return AffineField(aff_maps)