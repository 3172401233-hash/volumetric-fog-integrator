"""Light samples with visibility intervals along the camera ray."""

from dataclasses import dataclass

INF = float("inf")

_FULL_VISIBILITY = ((0.0, INF),)


@dataclass(frozen=True)
class DirectionalLight:
    """Distant light.

    direction: unit vector the light travels along.
    radiance: RGB radiance arriving at any point (no distance falloff).
    visibility: intervals of camera-ray parameter t where the light is
        not occluded; anything outside these intervals is in shadow.
    """

    direction: tuple
    radiance: tuple
    visibility: tuple = _FULL_VISIBILITY


@dataclass(frozen=True)
class PointLight:
    """Point light with inverse-square falloff.

    position: world position.
    intensity: RGB radiant intensity; irradiance at distance r is
        intensity / r^2.
    visibility: as for DirectionalLight.
    """

    position: tuple
    intensity: tuple
    visibility: tuple = _FULL_VISIBILITY
