"""Medium volumes: axis-aligned boxes with RGB coefficients."""

from dataclasses import dataclass

from .phase import ISOTROPIC

INF = float("inf")


@dataclass(frozen=True)
class Medium:
    """An axis-aligned box of participating medium.

    sigma_a / sigma_s: RGB absorption and scattering coefficients (1/length).
    emission: RGB emitted radiance per unit length.
    phase: phase function callable, defaults to isotropic.
    """

    min_corner: tuple
    max_corner: tuple
    sigma_a: tuple
    sigma_s: tuple
    emission: tuple = (0.0, 0.0, 0.0)
    phase: object = None

    def phase_function(self):
        return self.phase if self.phase is not None else ISOTROPIC


def intersect_box(origin, direction, min_corner, max_corner):
    """Slab intersection of a ray with an AABB.

    Returns (t_enter, t_exit) along the ray, or None if the ray misses.
    The interval may start before t=0 when the origin is inside the box.
    """
    t_enter, t_exit = -INF, INF
    for axis in range(3):
        o = origin[axis]
        d = direction[axis]
        lo = min_corner[axis]
        hi = max_corner[axis]
        if d == 0.0:
            if o < lo or o > hi:
                return None
            continue
        ta = (lo - o) / d
        tb = (hi - o) / d
        if ta > tb:
            ta, tb = tb, ta
        if ta > t_enter:
            t_enter = ta
        if tb < t_exit:
            t_exit = tb
        if t_enter > t_exit:
            return None
    return (t_enter, t_exit)
