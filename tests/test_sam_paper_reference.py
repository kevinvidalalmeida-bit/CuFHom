"""Bounded implementation checks, not an AR33/Vf60 success certificate."""
import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from cufhom.pipeline.sam_paper_reference import (rotational_inertia, isotropic_a4, fourth_moment,
    a4_error, sphere_exp, objective, relax_paper_reference, ContactNeighbors)
from cufhom.pipeline.sam_staged_insertion import Settings, validate_settings, certificate, generate_staged


class PaperReferenceTests(unittest.TestCase):
    def test_inertia_formula_and_units(self):
        from fractions import Fraction as F
        expected = (F(33**3,12)+F(33**2,6)+F(3*33,16)+F(1,15))/F(101,3)
        self.assertAlmostEqual(rotational_inertia(33., 1.), float(expected))
        self.assertAlmostEqual(rotational_inertia(66., 2.), 4*rotational_inertia(33., 1.))
        self.assertAlmostEqual(rotational_inertia(1e5, 1.)/1e10, 1/12, places=5)

    def test_a4_isotropic_contraction_and_rotation(self):
        a = isotropic_a4()
        self.assertAlmostEqual(a[0, 0, 0, 0], 1/5)
        self.assertAlmostEqual(a[0, 0, 1, 1], 1/15)
        np.testing.assert_allclose(np.einsum('ijkk->ij', a), np.eye(3)/3)
        q, _ = np.linalg.qr(np.random.default_rng(98).normal(size=(3, 3)))
        rotated = np.einsum('ia,jb,kc,ld,abcd->ijkl', q, q, q, q, a)
        np.testing.assert_allclose(rotated, a, atol=2e-16)

    def test_a2_isotropic_does_not_imply_a4(self):
        self.assertGreater(a4_error(np.eye(3)), .5)
        table = np.c_[np.array([[2,2,2], [6,2,2], [2,6,2]]), np.eye(3),
                      np.ones(3), np.full(3, .2)]
        self.assertTrue(certificate(table, 10, np.eye(3)/3, .001, .001)['accepted'])
        self.assertFalse(certificate(table, 10, np.eye(3)/3, .001, .001, .001)['accepted'])

    def test_a4_tangent_gradient(self):
        rng = np.random.default_rng(75)
        c = rng.uniform(0, 5, (7, 3))
        u = rng.normal(size=(7, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        dc = rng.normal(size=c.shape); du = rng.normal(size=u.shape)
        du -= u*np.sum(u*du, axis=1)[:, None]
        args = (3., np.full(3, 5.), 1.)
        _, gc, gu, _, _ = objective(c, u, *args)
        h = 1e-6
        fd = (objective(c+h*dc, sphere_exp(u, h*du), *args)[0]
              -objective(c-h*dc, sphere_exp(u, -h*du), *args)[0])/(2*h)
        self.assertAlmostEqual(fd, float(np.sum(gc*dc)+np.sum(gu*du)), places=7)

    def test_neighbors_match_dense_including_rejected_trial_and_return(self):
        rng = np.random.default_rng(829)
        c = rng.uniform(0, 8, (20, 3))
        u = rng.normal(size=(20, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        # Include a self-intersecting periodic long-fiber case too.
        for length in (2., 9.):
            cache = ContactNeighbors(length, np.full(3, 8.))
            for shift, rotation in ((0., 0.), (.01, .01), (2., .6), (0., 0.)):
                c1 = np.ascontiguousarray((c+shift*rng.normal(size=c.shape)) % 8)
                u1 = sphere_exp(u, rotation*rng.normal(size=u.shape))
                dense = objective(c1, u1, length, np.full(3, 8.))
                sparse = objective(c1, u1, length, np.full(3, 8.), neighbors=cache)
                for a, b in zip(dense, sparse):
                    np.testing.assert_array_equal(a, b)
            self.assertGreater(cache.rebuilds, 1)

    def test_exp_preserves_unit_vectors_and_sign(self):
        u = np.eye(3); tangent = np.roll(u, 1, axis=1)*.7
        updated = sphere_exp(u, tangent)
        np.testing.assert_allclose(np.linalg.norm(updated, axis=1), 1.)
        np.testing.assert_allclose(sphere_exp(-u, -tangent), -updated)

    def test_general_a2_and_periodic_self_image_guard(self):
        with self.assertRaisesRegex(ValueError, 'self-image'):
            validate_settings(Settings(length=33, box=16, solver='paper-a4'), np.diag([1.,0.,0.]))
        validate_settings(Settings(length=2, box=8, solver='paper-a4'), np.diag([.5,.3,.2]))

    def test_solver_dense_cache_equivalence_and_monotone_energy(self):
        rng = np.random.default_rng(917)
        u = rng.normal(size=(12, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        f = np.c_[rng.uniform(0, 9, (12, 3)), u, np.full(12, 3.), np.ones(12)]
        original = f.copy()
        kwargs = dict(maxiter=30, a4_tolerance=1e-5)
        dense, info = relax_paper_reference(f, 9., np.eye(3)/3, use_neighbors=False, **kwargs)
        cached, cached_info = relax_paper_reference(f, 9., np.eye(3)/3, **kwargs)
        np.testing.assert_array_equal(f, original)
        np.testing.assert_array_equal(dense, cached)
        np.testing.assert_array_equal(cached[:, 6:], original[:, 6:])
        self.assertEqual(info['iterations'], cached_info['iterations'])
        self.assertTrue(np.all(np.diff([r['energy'] for r in info['history']]) <= 0))

    def test_stage_rejects_false_optimizer_success_when_a4_fails(self):
        def false_success(trial, *args, **kwargs):
            f = trial.copy()
            f[:, :3] = np.array([[2,2,2],[6,2,2],[2,6,2],[2,2,6],[6,6,2],[6,2,6]])
            f[:, 3:6] = np.vstack([np.eye(3), np.eye(3)])
            return f, dict(iterations=1, evaluations=1)
        s = Settings(box=10, length=1, target_count=6, initial_count=6, candidates=1,
                     attempts=1, max_stages=1, solver='paper-a4')
        with patch('cufhom.pipeline.sam_paper_reference.relax_paper_reference', false_success):
            f, summary = generate_staged(s, np.eye(3)/3)
        self.assertEqual(len(f), 0)
        self.assertEqual(summary['status'], 'bootstrap_failed')


if __name__ == '__main__':
    unittest.main()
