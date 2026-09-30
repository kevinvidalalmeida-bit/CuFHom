"""Gradient, periodic geometry, and raster consistency checks; no FFT."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from cufhom.pipeline.sam_contact_energy import objective, contact_energy, relax_contact_energy
from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers
from cufhom.pipeline import rve_generator as raster


class ContactEnergyTests(unittest.TestCase):
    def setUp(self):
        from numba import set_num_threads
        set_num_threads(2)

    def test_gradient_finite_differences(self):
        rng = np.random.default_rng(258)
        x = np.column_stack((rng.random((6, 3))*4, rng.normal(size=(6, 3)))).ravel()
        args = (3., np.full(3, 4.), np.diag([.5, .3, .2]), .7)
        value, analytic, _, _ = objective(x, *args)
        finite = np.empty_like(x)
        for k in range(len(x)):
            plus, minus = x.copy(), x.copy()
            plus[k] += 1e-6; minus[k] -= 1e-6
            finite[k] = (objective(plus, *args)[0]-objective(minus, *args)[0])/2e-6
        self.assertGreater(value, 0.)
        np.testing.assert_allclose(analytic, finite, atol=2e-8, rtol=3e-5)

    def test_self_image_gradient(self):
        x = np.array([[1., 1., 1., 1.5, .08, .02]]).ravel()
        args = (3., np.full(3, 3.), np.eye(3)/3, 0.)
        _, g, maximum, _ = objective(x, *args)
        self.assertGreater(maximum, 0.)
        for k in range(6):
            plus, minus = x.copy(), x.copy()
            plus[k] += 1e-6; minus[k] -= 1e-6
            fd = (objective(plus, *args)[0]-objective(minus, *args)[0])/2e-6
            self.assertAlmostEqual(g[k], fd, places=7)

    def test_maximum_matches_independent_audit(self):
        rng = np.random.default_rng(908)
        centers = rng.random((12, 3))*4
        directions = rng.normal(size=(12, 3)); directions /= np.linalg.norm(directions, axis=1)[:, None]
        for length in (2., 5.):
            table = np.c_[centers, directions, np.full(12, length), np.ones(12)]
            audited = audit_periodic_fibers(table, 4., 1/120)
            energy, gc, gu, maximum = contact_energy(centers, directions, length, 1., np.full(3, 4.))
            self.assertAlmostEqual(maximum, audited['maximum_relative_overlap'], places=11)
            np.testing.assert_allclose(gc.sum(axis=0), 0., atol=1e-12)

    def test_simple_relaxation_and_no_input_mutation(self):
        f = np.array([[2., 2., 2., 1., 0., 0., 2., 1.],
                      [2., 2.5, 2., 1., 0., 0., 2., 1.]])
        before = f.copy()
        out, result = relax_contact_energy(f, 8., np.diag([1., 0., 0.]), maxiter=50)
        np.testing.assert_array_equal(f, before)
        np.testing.assert_array_equal(out[:, 6:], before[:, 6:])
        self.assertTrue(result['contact_and_A2_accepted'])
        self.assertTrue(result['vf_not_certified'])

    def test_energy_periodicity_permutation_and_axis_sign(self):
        rng = np.random.default_rng(582)
        c = rng.uniform(0, 4, (8, 3))
        u = rng.normal(size=(8, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        args = (5., 1., np.full(3, 4.))
        e, gc, gu, maximum = contact_energy(c, u, *args)
        order = rng.permutation(len(c))
        actual = contact_energy(c[order]+[8., -4., 12.], -u[order], *args)
        self.assertAlmostEqual(actual[0], e, places=11)
        self.assertAlmostEqual(actual[3], maximum, places=11)
        np.testing.assert_allclose(actual[1], gc[order], atol=1e-11)
        np.testing.assert_allclose(actual[2], -gu[order], atol=1e-11)

    def test_full_buffer_falls_back_without_truncation(self):
        c = np.array([[4., 4., 4.]])
        u = np.array([[1., 0., 0.]])
        expected = raster._rasterize_fibers_python(c, u, 2., 1., 8, 513)
        def simulate_overflow(*args):
            args[-1][:] = args[5]+1  # actual_counts > per-fiber capacity
        with patch.object(raster, '_count_and_fill_fiber_indices_parallel', simulate_overflow):
            actual = raster._rasterize_fibers_numba(c, u, 2., 1., 8, 513)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_voxel_neighbors_do_not_imply_geometric_overlap(self):
        from cufhom.cli.contacts import voxel_diagnostic
        f = np.array([[2., 2., 2., 1., 0., 0., 2., .8],
                      [2., 3., 2., 1., 0., 0., 2., .8]])
        self.assertTrue(audit_periodic_fibers(f, 8., 0.)['accepted'])
        result = voxel_diagnostic(f, 8., 8)
        self.assertEqual(result['multiply_occupied_sample_points'], 0)
        self.assertGreater(result['unlike_fiber_neighbor_faces'], 0)

    def test_flat_cylinder_membership_excludes_caps_and_boundaries(self):
        from cufhom.cli.contacts import flat_cylinder_interior
        f = np.array([2., 2., 2., 1., 0., 0., 2., 1.])
        points = np.array([[2., 2., 2.], [3.2, 2., 2.], [3., 2., 2.],
                           [2., 2.5, 2.], [10., 2., 2.]])
        # Membership takes points in the fundamental cell, as the raster does.
        np.testing.assert_array_equal(flat_cylinder_interior(points % 8., f, 8.),
                                      [True, False, False, False, True])

    def test_flat_cylinder_vs_cap_only_intersection(self):
        from cufhom.cli.contacts import voxel_diagnostic
        f = np.array([[2., 2., 2., 1., 0., 0., 2., 1.],
                      [4.5, 2., 2., 1., 0., 0., 2., 1.]])
        caps = voxel_diagnostic(f, 8., 32)
        self.assertGreater(caps['multiply_occupied_sample_points'], 0)
        self.assertEqual(caps['flat_cylinder_strict_shared_sample_points'], 0)
        f[1, :3] = [2., 2.5, 2.]
        bodies = voxel_diagnostic(f, 8., 32)
        self.assertGreater(bodies['flat_cylinder_strict_shared_sample_points'], 0)

    def test_periodic_duplicate_voxels_counted_once(self):
        c = np.array([[4., 4., 4.]])
        u = np.array([[1., 0., 0.]])
        for fn in (raster._rasterize_fibers_python, raster._rasterize_fibers_numba):
            phase, _, count, used = fn(c, u, 10., 1., 8, 8**3+1)
            self.assertEqual(count, np.count_nonzero(phase))
            self.assertEqual(used, 1)

    def test_zero_voxel_fiber_still_in_retained_prefix(self):
        c = np.array([[.5, .5, .5], [2., 2., 2.]])
        u = np.array([[1., 0., 0.], [1., 0., 0.]])
        for fn in (raster._rasterize_fibers_python, raster._rasterize_fibers_numba):
            phase, _, count, used = fn(c, u, .1, .05, 8, 8**3+1)
            self.assertEqual(count, 1)
            self.assertEqual(used, 2)

    # Campaign-specific AR16/Vf40 top-up test is intentionally not a standalone
    # unit test: it needs the original CNN campaign inputs and is not run here.


if __name__ == '__main__':
    unittest.main()
