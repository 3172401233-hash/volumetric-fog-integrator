"""Single-scattering volume integrator.

The camera ray is clipped to [0, t_max] and split at every medium
boundary and every light-visibility edge, producing elementary segments
in which the set of overlapping media and the visibility state are
constant. Inside one elementary segment the combined extinction
coefficient is constant, so transmittance and emission are integrated
analytically (Beer-Lambert). Directional-light inscattering is also
analytic; point-light inscattering (1/r^2 falloff and a varying phase
angle) is integrated with adaptive Simpson quadrature.

Overlapping media are combined by summing their coefficients once per
overlap region (each medium contributes exactly once; nothing is
double-counted and no last-write-wins). Pass overlap="raise" to reject
overlaps instead.
"""

import math
from dataclasses import dataclass

from .light import DirectionalLight, PointLight
from .medium import intersect_box
from .phase import ISOTROPIC
from .vec3 import dot, length, normalize, scale, sub

INF = float("inf")


@dataclass(frozen=True)
class RayResult:
    """Result of integrating one camera ray.

    radiance: RGB inscattered radiance reaching the camera.
    transmittance: RGB transmittance of the full ray (multiply by any
        background/surface radiance behind the medium).
    error: RGB estimate of the absolute quadrature error in radiance.
    subdivisions: total number of quadrature subintervals actually used.
    """

    radiance: tuple
    transmittance: tuple
    error: tuple
    subdivisions: int


def _int_factor(sigma_t, length_):
    """Integral of exp(-sigma_t * x) over x in [0, length_]."""
    if sigma_t == 0.0:
        return length_
    x = sigma_t * length_
    if abs(x) < 1e-8:
        return length_ * (1.0 - 0.5 * x)
    return (1.0 - math.exp(-x)) / sigma_t


def _simpson(fa, fb, fc, a, b):
    return (b - a) * (fa + 4.0 * fc + fb) / 6.0


def _adaptive_simpson(f, a, b, tol, max_depth):
    """Adaptive Simpson quadrature. Returns (value, error, n_intervals)."""
    c = 0.5 * (a + b)
    fa, fb, fc = f(a), f(b), f(c)
    whole = _simpson(fa, fb, fc, a, b)
    return _simpson_rec(f, a, b, fa, fb, fc, whole, tol, max_depth)


def _simpson_rec(f, a, b, fa, fb, fc, whole, tol, depth):
    c = 0.5 * (a + b)
    d = 0.5 * (a + c)
    e = 0.5 * (c + b)
    fd, fe = f(d), f(e)
    left = _simpson(fa, fc, fd, a, c)
    right = _simpson(fc, fb, fe, c, b)
    delta = left + right - whole
    if depth <= 0 or abs(delta) <= 15.0 * tol:
        return left + right + delta / 15.0, abs(delta) / 15.0, 2
    lv, le, ln = _simpson_rec(f, a, c, fa, fc, fd, left, 0.5 * tol, depth - 1)
    rv, re, rn = _simpson_rec(f, c, b, fc, fb, fe, right, 0.5 * tol, depth - 1)
    return lv + rv, le + re, ln + rn


def integrate_ray(origin, direction, media, lights, t_max,
                  tol=1e-6, max_depth=20, overlap="sum"):
    """Integrate transmittance and single scattering along a camera ray.

    origin, direction: ray; direction need not be unit length (t values,
        including t_max and visibility intervals, are in units of the
        given direction vector).
    media: iterable of Medium (axis-aligned boxes).
    lights: iterable of DirectionalLight / PointLight.
    t_max: far clip distance along the ray.
    tol: absolute error tolerance per adaptive quadrature call.
    max_depth: maximum recursion depth of the adaptive quadrature; each
        call uses at most 2**(max_depth + 1) subintervals.
    overlap: "sum" combines overlapping media by summing coefficients;
        "raise" raises ValueError on any overlap.
    """
    if overlap not in ("sum", "raise"):
        raise ValueError("overlap must be 'sum' or 'raise'")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    dir_len = length(direction)
    if dir_len == 0.0:
        raise ValueError("ray direction must be non-zero")
    d = scale(direction, 1.0 / dir_len)
    t_far = t_max * dir_len
    view_to_camera = scale(d, -1.0)

    cuts = [0.0, t_far]
    spans = []
    for m in media:
        hit = intersect_box(origin, d, m.min_corner, m.max_corner)
        if hit is None:
            continue
        t0 = max(hit[0], 0.0)
        t1 = min(hit[1], t_far)
        if t1 > t0:
            spans.append((t0, t1, m))
            cuts.append(t0)
            cuts.append(t1)
    for light in lights:
        for v0, v1 in light.visibility:
            cuts.append(min(max(v0 * dir_len, 0.0), t_far))
            cuts.append(min(max(v1 * dir_len, 0.0), t_far))
    cuts = sorted(set(cuts))

    radiance = [0.0, 0.0, 0.0]
    transmittance = [1.0, 1.0, 1.0]
    error = [0.0, 0.0, 0.0]
    subdivisions = 0

    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if not b > a:
            continue
        mid = 0.5 * (a + b)
        active = [m for (t0, t1, m) in spans if t0 <= mid <= t1]
        if not active:
            continue
        if len(active) > 1 and overlap == "raise":
            raise ValueError(
                "overlapping media on segment [%r, %r]" % (a, b))

        sigma_a = [sum(m.sigma_a[k] for m in active) for k in range(3)]
        sigma_s = [sum(m.sigma_s[k] for m in active) for k in range(3)]
        emission = [sum(m.emission[k] for m in active) for k in range(3)]
        phase = _combined_phase(active)
        seg_len = b - a

        seg_rad = [0.0, 0.0, 0.0]
        seg_err = [0.0, 0.0, 0.0]
        seg_trans = [0.0, 0.0, 0.0]
        for k in range(3):
            sigma_t = sigma_a[k] + sigma_s[k]
            seg_trans[k] = math.exp(-sigma_t * seg_len)
            seg_rad[k] += emission[k] * _int_factor(sigma_t, seg_len)

        for light in lights:
            for v0, v1 in light.visibility:
                c = max(a, v0 * dir_len)
                e = min(b, v1 * dir_len)
                if not e > c:
                    continue
                if isinstance(light, DirectionalLight):
                    to_light = normalize(scale(light.direction, -1.0))
                    cos_theta = dot(view_to_camera, to_light)
                    p = phase(cos_theta)
                    for k in range(3):
                        sigma_t = sigma_a[k] + sigma_s[k]
                        seg_rad[k] += (sigma_s[k] * p * light.radiance[k]
                                       * math.exp(-sigma_t * (c - a))
                                       * _int_factor(sigma_t, e - c))
                elif isinstance(light, PointLight):
                    for k in range(3):
                        sigma_t = sigma_a[k] + sigma_s[k]

                        def integrand(t, k=k, sigma_t=sigma_t, light=light):
                            x = (origin[0] + t * d[0],
                                 origin[1] + t * d[1],
                                 origin[2] + t * d[2])
                            offset = sub(x, light.position)
                            r2 = dot(offset, offset)
                            if r2 == 0.0:
                                return 0.0
                            travel = scale(offset, 1.0 / math.sqrt(r2))
                            p = phase(dot(travel, view_to_camera))
                            return (sigma_s[k]
                                    * math.exp(-sigma_t * (t - a))
                                    * p * light.intensity[k] / r2)

                        value, err, n = _adaptive_simpson(
                            integrand, c, e, tol, max_depth)
                        seg_rad[k] += value
                        seg_err[k] += err
                        subdivisions += n
                else:
                    raise TypeError("unsupported light type: %r" % (light,))

        for k in range(3):
            radiance[k] += transmittance[k] * seg_rad[k]
            error[k] += transmittance[k] * seg_err[k]
            transmittance[k] *= seg_trans[k]

    return RayResult(tuple(radiance), tuple(transmittance),
                     tuple(error), subdivisions)


def _combined_phase(active):
    """Sigma_s-weighted mix of the active media's phase functions."""
    if len(active) == 1:
        return active[0].phase_function()
    phases = [m.phase_function() for m in active]
    weights = [sum(m.sigma_s) for m in active]
    total = sum(weights)
    if total <= 0.0:
        return ISOTROPIC

    def phase(cos_theta):
        num = 0.0
        for w, ph in zip(weights, phases):
            if w > 0.0:
                num += w * ph(cos_theta)
        return num / total

    return phase
