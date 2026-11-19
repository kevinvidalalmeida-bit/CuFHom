"""Conservative skin cache versus full periodic force graph."""

import unittest

import numpy as np
from cufhom.geometry.contacts import (
    ContactNeighbors,
    a4_objective,
    build_aabb_pairs,
    endpoint_motion_bound,
)


class NeighborTests(unittest.TestCase):
    def test_cached_forces_equal_all_pairs_after_translation_and_rotation(self):
        rng = np.random.default_rng(31)
        box = np.full(3, 16.0)
        from numba import set_num_threads

        set_num_threads(2)
        for length in [3.0, 8.0, 16.0, 25.0]:
            c = rng.uniform(0, 16, (90, 3))
            u = rng.normal(size=c.shape)
            u /= np.linalg.norm(u, axis=1)[:, None]
            skin = 1.0
            nc = (c + rng.uniform(-0.03, 0.03, c.shape)) % 16
            nu = u + rng.uniform(-0.001, 0.001, u.shape)
            nu /= np.linalg.norm(nu, axis=1)[:, None]
            self.assertLess(endpoint_motion_bound(nc, nu, c, u, box, length), skin / 2)
            cache = ContactNeighbors(length, box, skin=skin)
            a4_objective(c, u, length, box, neighbors=cache)
            cached = a4_objective(nc, nu, length, box, neighbors=cache)
            dense = a4_objective(nc, nu, length, box)
            for actual, expected in zip(cached, dense):
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)

    def test_rotation_alone_requires_rebuild(self):
        c = np.zeros((2, 3))
        u = np.array([[1.0, 0, 0], [0, 1.0, 0]])
        angle = 0.15
        v = np.array([[np.cos(angle), np.sin(angle), 0], [0, 1.0, 0]])
        self.assertGreater(
            endpoint_motion_bound(c, v, c, u, np.full(3, 16.0), 16.0), 0.5
        )

    def test_periodic_wrap_does_not_overestimate_translation(self):
        c = np.array([[15.99, 0, 0]])
        u = np.array([[1.0, 0, 0]])
        new = np.array([[0.01, 0, 0]])
        self.assertAlmostEqual(
            endpoint_motion_bound(new, u, c, u, np.full(3, 16.0), 8.0), 0.02
        )

    def test_no_duplicates_and_sorted_pairs(self):
        rng = np.random.default_rng(19)
        c = rng.uniform(0, 16, (35, 3))
        u = np.tile([1.0, 0, 0], (35, 1))
        pairs = build_aabb_pairs(c, u, np.full(3, 16.0), 8.0, 1.0, 0.5)
        self.assertEqual(len(np.unique(pairs, axis=0)), len(pairs))
        self.assertTrue(np.all(pairs[:, 0] < pairs[:, 1]))
        self.assertEqual(pairs.tolist(), sorted(pairs.tolist()))


if __name__ == "__main__":
    unittest.main()
