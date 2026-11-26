"""Periodic capsule raster and public geometry export certification."""

import numpy as np
from cufhom.geometry import GeometryRequest, generate_geometry
from cufhom.geometry.voxel import rasterize_capsules

from tests.geometry.reference.raster_reference import _rasterize_fibers_python


def test_capsule_raster_matches_exhaustive_periodic_reference():
    rng = np.random.default_rng(653)
    c = rng.uniform(-5.0, 20.0, (8, 3))
    u = rng.normal(size=c.shape)
    u /= np.linalg.norm(u, axis=1)[:, None]
    u[:3] = np.eye(3)
    for length in (5.0, 23.0):
        actual = rasterize_capsules(c, u, length, 1.13, 16)
        expected = _rasterize_fibers_python(c, u, length, 1.13, 16, 16**3 + 1)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        translated = rasterize_capsules(c + [32.0, -16.0, 48.0], -u, length, 1.13, 16)
        np.testing.assert_array_equal(translated[0], actual[0])


def test_public_generator_exports_only_certified_fields(tmp_path):
    for name, grid, accepted in (("resolved", 36, True), ("unresolved", 3, False)):
        output = tmp_path / name
        result = generate_geometry(
            GeometryRequest(
                output=output,
                box=12.0,
                ar=2.0,
                diameter=1.0,
                vf=0.02,
                a2=(1.0, 0.0),
                seed=572,
                threads=1,
                grid=grid,
                shape="cylinder",
                solver="a4",
                batch_size=11,
                a2_tolerance=0.01,
                voxel_a2_tolerance=0.01,
                vf_tolerance=0.005,
                overlap_tolerance=1e-6,
                verbose=False,
            )
        )
        assert result["accepted"] is accepted
        assert (output / "geometry.npz").exists() is accepted
        if accepted:
            with np.load(output / "geometry.npz", allow_pickle=False) as geometry:
                assert geometry["phase"].shape == (grid,) * 3
                assert geometry["ori"].shape == (grid,) * 3 + (3,)
                assert set(np.unique(geometry["phase"])) == {0, 1}
                np.testing.assert_allclose(
                    np.abs(geometry["ori"][geometry["phase"] == 1]),
                    np.tile([1.0, 0.0, 0.0], (np.count_nonzero(geometry["phase"]), 1)),
                    atol=1e-6,
                )
            assert (
                result["geometric_certificate"]["audit"]["maximum_relative_overlap"]
                <= 1e-6
            )
            assert abs(result["vf"] - 0.02) <= 0.005
        else:
            assert not result["vf_ok"]


def test_bootstrap_failure_saves_diagnostics_without_export(tmp_path):
    output = tmp_path / "failed"
    summary = generate_geometry(
        GeometryRequest(
            output=output,
            box=12.0,
            ar=2.0,
            vf=0.02,
            grid=36,
            solver="a4",
            initial_count=6,
            bootstrap_max_count=6,
            max_stages=1,
            relax_iterations=1,
            max_relax_iterations=1,
            candidates=1,
            attempts=1,
            threads=1,
            seed=1,
            verbose=False,
        )
    )
    assert summary["status"] == "bootstrap_failed"
    assert summary["accepted"] is False
    assert summary["accepted_count"] == 0
    assert (output / "summary.json").exists()
    assert not (output / "geometry.npz").exists()
