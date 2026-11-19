import unittest
from unittest.mock import patch

import numpy as np
from cufhom.geometry.contacts import contact_energy
from cufhom.geometry.flexible import (
    FlexibleSettings,
    chain_audit,
    chain_objective,
    decode,
    generate_flexible,
)
from cufhom.geometry.straight import Settings, generate_staged


class StagedTests(unittest.TestCase):
    def test_rigid_rollback_preserves_accepted_input(self):
        initial = np.array(
            [
                [2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0],
                [2.0, 4.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0],
            ]
        )
        before = initial.copy()
        checkpoints = []
        s = Settings(
            box=8.0, length=2.0, target_count=3, batch_size=1, attempts=2, candidates=2
        )

        def fail(trial, *args, **kwargs):
            bad = trial.copy()
            bad[-1] = bad[0]
            return bad, dict(iterations=1, evaluations=1)

        with patch("cufhom.geometry.straight.relax_contact_energy", fail):
            final, info = generate_staged(
                s,
                np.diag([1.0, 0.0, 0.0]),
                initial=initial,
                checkpoint=lambda *args: checkpoints.append(args),
            )
        np.testing.assert_array_equal(initial, before)
        np.testing.assert_array_equal(final, before)
        self.assertEqual(info["status"], "stalled")
        self.assertEqual(len(checkpoints), 0)
        self.assertEqual(sum(e["event"] == "rollback" for e in info["events"]), 2)

    def test_incremental_contact_energy(self):
        rng = np.random.default_rng(789)
        c = rng.random((5, 3)) * 3
        u = rng.normal(size=(5, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        args = (2.0, 1.0, np.full(3, 4.0))
        full = contact_energy(c, u, *args)[0]
        old = contact_energy(c[:3], u[:3], *args)[0]
        added = contact_energy(c, u, *args, first_new=3)[0]
        self.assertAlmostEqual(full, old + added, places=12)

    def test_flexible_gradients_including_bending_and_turn(self):
        rng = np.random.default_rng(183)
        s = FlexibleSettings(
            box=5.0, length=4.0, segments=3, diameter=1.0, bending_weight=0.3
        )
        q = rng.normal(size=(3, 4, 3))
        q[:, 0] = rng.uniform(0, 5, (3, 3))
        x = q.ravel()
        target = np.diag([0.5, 0.3, 0.2])
        _, g = chain_objective(x, 3, s, target)
        fd = np.empty_like(x)
        for k in range(len(x)):
            p, m = x.copy(), x.copy()
            p[k] += 1e-6
            m[k] -= 1e-6
            fd[k] = (
                chain_objective(p, 3, s, target)[0]
                - chain_objective(m, 3, s, target)[0]
            ) / 2e-6
        np.testing.assert_allclose(g, fd, rtol=2e-5, atol=5e-8)

    def test_connectivity_and_length_by_construction(self):
        rng = np.random.default_rng(193)
        q = rng.normal(size=(4, 5, 3))
        vertices, c, u, _ = decode(q, 16.0)
        np.testing.assert_allclose(
            np.linalg.norm(np.diff(vertices, axis=1), axis=2), 4.0
        )
        np.testing.assert_allclose(
            c[:, :-1] + 2 * u[:, :-1], c[:, 1:] - 2 * u[:, 1:], atol=1e-14
        )

    def test_connected_neighbor_exemption_only_original_image(self):
        q = np.array([[[2.0, 2.0, 2.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        target = np.diag([1.0, 0.0, 0.0])

        def check(box):
            return chain_audit(
                q,
                4.0,
                1.0,
                box,
                target,
                overlap_tolerance=1 / 120,
                a2_tolerance=0.035,
                max_turn_degrees=30.0,
            )

        self.assertTrue(check(8.0)["accepted"])
        self.assertFalse(check(4.0)["accepted"])
        self.assertGreater(check(4.0)["self_violating_image_contacts"], 0)

    def test_flexible_kernel_matches_independent_audit(self):
        rng = np.random.default_rng(348)
        q = rng.normal(size=(3, 4, 3))
        q[:, 0] = rng.uniform(0, 5, (3, 3))
        _, c, u, _ = decode(q, 6.0)
        result = contact_energy(
            c.reshape(-1, 3).copy(),
            u.reshape(-1, 3).copy(),
            2.0,
            1.0,
            np.full(3, 5.0),
            segments_per_fiber=3,
        )
        check = chain_audit(
            q,
            6.0,
            1.0,
            5.0,
            np.eye(3) / 3,
            overlap_tolerance=1 / 120,
            a2_tolerance=0.035,
            max_turn_degrees=30.0,
        )
        self.assertAlmostEqual(result[3], check["maximum_relative_overlap"], places=11)

    def test_flexible_nonadjacent_self_crossing_rejected(self):
        q = np.array(
            [
                [
                    [2.0, 2.0, 2.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                ]
            ]
        )
        result = chain_audit(
            q,
            8.0,
            1.0,
            16.0,
            np.diag([0.5, 0.5, 0.0]),
            overlap_tolerance=1 / 120,
            a2_tolerance=0.035,
            max_turn_degrees=100.0,
        )
        self.assertFalse(result["accepted"])
        self.assertGreater(result["self_violating_image_contacts"], 0)

    def test_flexible_periodic_translation_and_axis_reversal(self):
        rng = np.random.default_rng(893)
        q = rng.normal(size=(2, 4, 3))
        q[:, 0] = rng.uniform(0, 4, (2, 3))
        vertices, _, u, _ = decode(q, 6.0)
        reverse = np.concatenate(
            (vertices[:, -1:, :] + [8.0, -4.0, 12.0], -u[:, ::-1]), axis=1
        )
        kwargs = dict(
            overlap_tolerance=1 / 120, a2_tolerance=0.035, max_turn_degrees=30.0
        )
        a = chain_audit(q, 6.0, 1.0, 4.0, np.eye(3) / 3, **kwargs)
        b = chain_audit(reverse, 6.0, 1.0, 4.0, np.eye(3) / 3, **kwargs)
        self.assertEqual(a["violating_image_contacts"], b["violating_image_contacts"])
        self.assertAlmostEqual(
            a["maximum_relative_overlap"], b["maximum_relative_overlap"], places=11
        )

    def test_bootstrap_failure_has_no_checkpoint(self):
        s = FlexibleSettings(
            box=8.0,
            length=4.0,
            segments=2,
            target_count=2,
            initial_count=2,
            attempts=1,
            candidates=2,
            max_stages=1,
            relax_iterations=2,
        )
        saved = []

        def fail(q, *args):
            q = q.copy()
            q[-1] = q[0]
            return q, dict(accepted=True), dict(iterations=1, evaluations=1)

        with patch("cufhom.geometry.flexible.relax_chains", fail):
            q, result = generate_flexible(
                s, np.eye(3) / 3, checkpoint=lambda *x: saved.append(x)
            )
        self.assertEqual(len(q), 0)
        self.assertEqual(saved, [])
        self.assertEqual(result["status"], "bootstrap_failed")

    def test_flexible_later_failure_keeps_last_valid_state(self):
        s = FlexibleSettings(
            box=16.0,
            length=2.0,
            diameter=0.5,
            segments=2,
            target_count=5,
            initial_count=4,
            batch_size=1,
            attempts=2,
            candidates=1,
            max_stages=2,
            relax_iterations=2,
        )
        u = np.array(
            [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
        ) / np.sqrt(3.0)
        q0 = np.zeros((4, 3, 3))
        q0[:, 0] = [[2, 2, 2], [8, 2, 2], [2, 8, 2], [2, 2, 8]]
        q0[:, 1:] = u[:, None]
        saved = []

        def fake(q, *args):
            result = q0.copy() if len(q) == 4 else q.copy()
            if len(q) > 4:
                result[-1] = result[0]
            return result, dict(accepted=True), dict(iterations=1, evaluations=1)

        with patch("cufhom.geometry.flexible.relax_chains", fake):
            final, summary = generate_flexible(
                s, np.eye(3) / 3, checkpoint=lambda q, event: saved.append(q)
            )
        self.assertEqual(summary["status"], "stalled")
        self.assertEqual(len(saved), 1)
        np.testing.assert_array_equal(final, q0)
        np.testing.assert_array_equal(saved[0], q0)

    def test_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            generate_flexible(FlexibleSettings(length=1.0, segments=4), np.eye(3) / 3)


if __name__ == "__main__":
    unittest.main()
