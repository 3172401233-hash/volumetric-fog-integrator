import math
import unittest

from volfog import (DirectionalLight, Medium, PointLight,
                    henyey_greenstein, integrate_ray)

ISO = 1.0 / (4.0 * math.pi)


def box(z0, z1, sigma_a, sigma_s, emission=(0.0, 0.0, 0.0), phase=None):
    return Medium(min_corner=(-1.0, -1.0, z0), max_corner=(1.0, 1.0, z1),
                  sigma_a=sigma_a, sigma_s=sigma_s,
                  emission=emission, phase=phase)


def rgb(v):
    return (v, v, v)


class ConstantDensityTest(unittest.TestCase):
    def test_transmittance_matches_beer_lambert(self):
        sigma_a = (0.3, 0.2, 0.1)
        medium = box(-1.0, 1.0, sigma_a, rgb(0.0))
        result = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [], 10.0)
        for k in range(3):
            expected = math.exp(-sigma_a[k] * 2.0)
            self.assertAlmostEqual(result.transmittance[k], expected,
                                   places=14)
            self.assertEqual(result.radiance[k], 0.0)
        self.assertEqual(result.subdivisions, 0)
        self.assertEqual(result.error, (0.0, 0.0, 0.0))

    def test_camera_inside_medium(self):
        medium = box(-1.0, 1.0, rgb(0.4), rgb(0.0))
        result = integrate_ray((0, 0, 0), (0, 0, 1), [medium], [], 10.0)
        for k in range(3):
            self.assertAlmostEqual(result.transmittance[k], math.exp(-0.4),
                                   places=14)

    def test_directional_light_analytic(self):
        sigma_a, sigma_s = rgb(0.1), (0.2, 0.3, 0.4)
        medium = box(-1.0, 1.0, sigma_a, sigma_s)
        light = DirectionalLight(direction=(0, 0, -1), radiance=(1, 2, 3))
        result = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [light], 10.0)
        for k in range(3):
            sigma_t = sigma_a[k] + sigma_s[k]
            expected = (sigma_s[k] * ISO * light.radiance[k]
                        * (1.0 - math.exp(-sigma_t * 2.0)) / sigma_t)
            self.assertAlmostEqual(result.radiance[k], expected, places=14)

    def test_zero_density_is_vacuum(self):
        medium = box(-1.0, 1.0, rgb(0.0), rgb(0.0))
        light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(1.0))
        result = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [light], 10.0)
        self.assertEqual(result.transmittance, (1.0, 1.0, 1.0))
        self.assertEqual(result.radiance, (0.0, 0.0, 0.0))


class ExponentialDecayTest(unittest.TestCase):
    def test_emission_approaches_asymptote_analytically(self):
        sigma_a, sigma_s = rgb(0.3), rgb(0.2)
        sigma_t = 0.5
        for z_far in (0.25, 0.5, 1.0, 2.0):
            medium = box(0.0, z_far, sigma_a, sigma_s, emission=rgb(1.0))
            result = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [], 10.0)
            expected = (1.0 - math.exp(-sigma_t * z_far)) / sigma_t
            for k in range(3):
                self.assertAlmostEqual(result.radiance[k], expected,
                                       places=14)
                self.assertAlmostEqual(result.transmittance[k],
                                       math.exp(-sigma_t * z_far), places=14)


class AdjacentSegmentsTest(unittest.TestCase):
    def test_two_density_segments_match_analytic(self):
        sa1, ss1 = rgb(0.3), rgb(0.1)
        sa2, ss2 = rgb(0.7), rgb(0.5)
        media = [box(-1.0, 0.0, sa1, ss1), box(0.0, 1.0, sa2, ss2)]
        light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(1.0))
        result = integrate_ray((0, 0, -5), (0, 0, 1), media, [light], 10.0)
        st1, st2 = sa1[0] + ss1[0], sa2[0] + ss2[0]
        t1 = math.exp(-st1)
        t2 = math.exp(-st2)
        for k in range(3):
            self.assertAlmostEqual(result.transmittance[k], t1 * t2,
                                   places=14)
            l1 = ss1[k] * ISO * (1.0 - t1) / st1
            l2 = ss2[k] * ISO * (1.0 - t2) / st2
            self.assertAlmostEqual(result.radiance[k], l1 + t1 * l2,
                                   places=14)

    def test_radiance_continuous_across_boundary(self):
        light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(1.0))

        def radiance_at(delta):
            media = [box(-1.0, delta, rgb(0.3), rgb(0.1)),
                     box(delta, 1.0, rgb(0.7), rgb(0.5))]
            return integrate_ray((0, 0, -5), (0, 0, 1), media, [light],
                                 10.0).radiance

        base = radiance_at(0.0)
        for delta in (1e-7, -1e-7):
            shifted = radiance_at(delta)
            for k in range(3):
                self.assertAlmostEqual(base[k], shifted[k], places=6)


class PartialOcclusionTest(unittest.TestCase):
    def test_visibility_interval_limits_inscattering(self):
        sigma_a, sigma_s = rgb(0.2), rgb(0.5)
        medium = box(-1.0, 1.0, sigma_a, sigma_s)
        # Ray enters the box at t=4, exits at t=6; light visible on t in
        # [4.5, 5.5] only (i.e. z in [-0.5, 0.5]).
        light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(2.0),
                                 visibility=((4.5, 5.5),))
        result = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [light], 10.0)
        sigma_t = 0.7
        expected = (sigma_s[0] * ISO * 2.0
                    * math.exp(-sigma_t * 0.5)
                    * (1.0 - math.exp(-sigma_t * 1.0)) / sigma_t)
        for k in range(3):
            self.assertAlmostEqual(result.radiance[k], expected, places=14)

    def test_radiance_continuous_as_shadow_edge_moves(self):
        medium = box(-1.0, 1.0, rgb(0.2), rgb(0.5))

        def radiance_at(edge):
            light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(2.0),
                                     visibility=((4.0, edge),))
            return integrate_ray((0, 0, -5), (0, 0, 1), [medium], [light],
                                 10.0).radiance

        base = radiance_at(5.0)
        for delta in (1e-7, -1e-7):
            shifted = radiance_at(5.0 + delta)
            for k in range(3):
                self.assertAlmostEqual(base[k], shifted[k], places=6)


class ForwardScatteringTest(unittest.TestCase):
    def test_henyey_greenstein_forward_peak(self):
        g = 0.9
        medium = box(-1.0, 1.0, rgb(0.1), rgb(0.5),
                     phase=henyey_greenstein(g))
        # Light travelling towards the camera (looking into the light).
        front = DirectionalLight(direction=(0, 0, -1), radiance=rgb(1.0))
        # Light travelling away from the camera.
        back = DirectionalLight(direction=(0, 0, 1), radiance=rgb(1.0))
        r_front = integrate_ray((0, 0, -5), (0, 0, 1), [medium], [front],
                                10.0).radiance
        r_back = integrate_ray((0, 0, 1), (0, 0, 1), [medium], [back],
                               10.0).radiance
        for k in range(3):
            self.assertGreater(r_front[k], 10.0 * r_back[k])

    def test_zero_g_matches_isotropic(self):
        m_iso = box(-1.0, 1.0, rgb(0.1), rgb(0.5))
        m_hg0 = box(-1.0, 1.0, rgb(0.1), rgb(0.5),
                    phase=henyey_greenstein(0.0))
        light = DirectionalLight(direction=(0.3, 0.4, -0.8660254),
                                 radiance=rgb(1.0))
        r_iso = integrate_ray((0, 0, -5), (0, 0, 1), [m_iso], [light], 10.0)
        r_hg0 = integrate_ray((0, 0, -5), (0, 0, 1), [m_hg0], [light], 10.0)
        for k in range(3):
            self.assertAlmostEqual(r_iso.radiance[k], r_hg0.radiance[k],
                                   places=14)

    def test_invalid_g_rejected(self):
        with self.assertRaises(ValueError):
            henyey_greenstein(1.0)


class OverlapTest(unittest.TestCase):
    def test_overlap_sums_coefficients_once(self):
        media = [box(-1.0, 0.5, rgb(0.3), rgb(0.0)),
                 box(-0.5, 1.0, rgb(0.2), rgb(0.0))]
        result = integrate_ray((0, 0, -5), (0, 0, 1), media, [], 10.0)
        # z in [-1,-0.5]: 0.3; z in [-0.5,0.5]: 0.3+0.2; z in [0.5,1]: 0.2
        expected = math.exp(-(0.3 * 0.5 + 0.5 * 1.0 + 0.2 * 0.5))
        for k in range(3):
            self.assertAlmostEqual(result.transmittance[k], expected,
                                   places=14)

    def test_overlap_raise_mode(self):
        media = [box(-1.0, 0.5, rgb(0.3), rgb(0.0)),
                 box(-0.5, 1.0, rgb(0.2), rgb(0.0))]
        with self.assertRaises(ValueError):
            integrate_ray((0, 0, -5), (0, 0, 1), media, [], 10.0,
                          overlap="raise")

    def test_overlap_not_double_counted(self):
        # Two identical media fully overlapping must equal one medium with
        # doubled coefficients, not one medium applied twice.
        one = box(-1.0, 1.0, rgb(0.3), rgb(0.1))
        doubled = box(-1.0, 1.0, rgb(0.6), rgb(0.2))
        light = DirectionalLight(direction=(0, 0, -1), radiance=rgb(1.0))
        r_two = integrate_ray((0, 0, -5), (0, 0, 1), [one, one], [light],
                              10.0)
        r_dbl = integrate_ray((0, 0, -5), (0, 0, 1), [doubled], [light],
                              10.0)
        for k in range(3):
            self.assertAlmostEqual(r_two.transmittance[k],
                                   r_dbl.transmittance[k], places=14)
            self.assertAlmostEqual(r_two.radiance[k], r_dbl.radiance[k],
                                   places=14)


class PointLightConvergenceTest(unittest.TestCase):
    def setUp(self):
        self.medium = box(-1.0, 1.0, rgb(0.1), rgb(0.8),
                          phase=henyey_greenstein(0.7))
        self.light = PointLight(position=(0.3, 0.2, 0.1),
                                intensity=rgb(10.0))

    def integrate(self, tol, max_depth=30):
        return integrate_ray((0, 0, -5), (0, 0, 1), [self.medium],
                             [self.light], 10.0, tol=tol,
                             max_depth=max_depth)

    def test_convergence_as_tolerance_tightens(self):
        reference = self.integrate(1e-13).radiance
        previous_error = None
        for tol in (1e-2, 1e-4, 1e-6, 1e-8):
            result = self.integrate(tol)
            err = max(abs(result.radiance[k] - reference[k])
                      for k in range(3))
            if previous_error is not None:
                self.assertLess(err, previous_error)
            previous_error = err
        self.assertLess(previous_error, 1e-6)

    def test_subdivisions_tracked_and_budgeted(self):
        loose = self.integrate(1e-2)
        tight = self.integrate(1e-8)
        self.assertGreater(loose.subdivisions, 0)
        self.assertGreater(tight.subdivisions, loose.subdivisions)
        # Each of the 3 RGB quadrature calls uses at most 2**(depth+1)
        # subintervals.
        capped = self.integrate(1e-12, max_depth=3)
        self.assertLessEqual(capped.subdivisions, 3 * 2 ** 4)
        self.assertGreaterEqual(max(capped.error), 0.0)

    def test_determinism(self):
        first = self.integrate(1e-6)
        second = self.integrate(1e-6)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
