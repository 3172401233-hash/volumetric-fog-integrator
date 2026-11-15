"""Phase functions describing how a medium scatters light.

All phase functions take ``cos_theta`` and return a probability density
normalized over the unit sphere (integrates to 1 over 4*pi steradians).

``cos_theta`` is the cosine of the scattering angle: the angle between the
direction the light is travelling and the direction from the scattering
point towards the camera. ``cos_theta == 1`` means the light continues
straight towards the viewer (forward scattering).
"""

import math

ISO_VALUE = 1.0 / (4.0 * math.pi)


def isotropic():
    """Uniform scattering in every direction."""

    def phase(cos_theta):
        return ISO_VALUE

    return phase


def henyey_greenstein(g):
    """Henyey-Greenstein phase function.

    ``g`` in (-1, 1): 0 is isotropic, positive values scatter forward
    (glow when looking towards the light), negative values backscatter.
    """
    if not -1.0 < g < 1.0:
        raise ValueError("g must be in the open interval (-1, 1)")

    def phase(cos_theta):
        denom = 1.0 + g * g - 2.0 * g * cos_theta
        return (1.0 - g * g) / (4.0 * math.pi * denom ** 1.5)

    return phase


ISOTROPIC = isotropic()
