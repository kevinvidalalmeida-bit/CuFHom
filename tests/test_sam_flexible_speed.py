"""Conservative spatial rejection must retain the exhaustive reference result."""
import sys
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
from cufhom.pipeline.sam_contact_energy import contact_energy
from cufhom.pipeline.sam_flexible import FlexibleSettings, chain_audit, decode, relax_chains


class FlexibleSpeedTests(unittest.TestCase):
    def test_contact_energy_bitwise_multiple_images_and_incremental(self):
        rng = np.random.default_rng(935)
        for box, length in ((16., 4.), (4., 12.), (5., 2.)):
            c = rng.uniform(-2*box, 2*box, (24, 3))
            u = rng.normal(size=c.shape); u /= np.linalg.norm(u, axis=1)[:, None]
            for first in (0, 20, 24):
                for segments in (1, 4):
                    results = [contact_energy(c, u, length, 1., np.full(3, box),
                                               first_new=first, segments_per_fiber=segments,
                                               conservative_cull=mode) for mode in (False, True)]
                    for ref, fast in zip(*results): np.testing.assert_array_equal(ref, fast)

    def test_tangency_and_degenerate_axes(self):
        u = np.tile([1., 0., 0.], (4, 1))
        for gap in (-1e-11, 0., 1e-11):
            c = np.array([[0., 0., 0.], [0., 1.+gap, 0.], [0., 0., 0.], [8., 0., 0.]])
            results = [contact_energy(c, u, 8., 1., np.full(3, 8.),
                                       conservative_cull=mode) for mode in (False, True)]
            for ref, fast in zip(*results): np.testing.assert_array_equal(ref, fast)

    def test_audit_exhaustive_equivalence_with_self_contacts(self):
        rng = np.random.default_rng(748)
        for length, box, ns in ((16., 16., 4), (12., 3., 2), (4., 20., 4)):
            q = rng.normal(size=(8, ns+1, 3)); q[:, 0] *= box
            checks = [chain_audit(q, length, 1., box, np.eye(3)/3,
                                  overlap_tolerance=1/120, a2_tolerance=.035,
                                  max_turn_degrees=30., conservative_cull=mode) for mode in (False, True)]
            self.assertEqual(*checks)

    def test_empty_contact_candidate_set_still_checks_A2_and_turn(self):
        q = np.array([[[10., 10., 10.], [1., 0., 0.], [1., 0., 0.]],
                      [[30., 30., 30.], [1., 0., 0.], [1., 0., 0.]]])
        ref, fast = [chain_audit(q, 2., .1, 100., np.eye(3)/3,
                                  overlap_tolerance=1/120, a2_tolerance=.035,
                                  max_turn_degrees=30., conservative_cull=mode) for mode in (False, True)]
        self.assertEqual(ref, fast)
        self.assertFalse(fast['accepted'])
        self.assertEqual(fast['maximum_relative_overlap'], 0.)

    def test_relaxation_bitwise_same_iterates_budget_and_output(self):
        rng = np.random.default_rng(729)
        s = FlexibleSettings(box=8., length=4., diameter=.7, segments=2, relax_iterations=15)
        q = rng.normal(size=(5, 3, 3)); q[:, 0] = rng.uniform(0, 8, (5, 3))
        q[:, 1:] /= np.linalg.norm(q[:, 1:], axis=2)[:, :, None]
        results = [relax_chains(q, replace(s, conservative_cull=mode), np.eye(3)/3)
                   for mode in (False, True)]
        np.testing.assert_array_equal(results[0][0], results[1][0])
        self.assertEqual(results[0][1:], results[1][1:])


if __name__ == '__main__': unittest.main()
