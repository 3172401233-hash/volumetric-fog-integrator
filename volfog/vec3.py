"""Minimal 3-component vector helpers (tuples of three floats)."""

import math


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def neg(a):
    return (-a[0], -a[1], -a[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    n = length(a)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return scale(a, 1.0 / n)
