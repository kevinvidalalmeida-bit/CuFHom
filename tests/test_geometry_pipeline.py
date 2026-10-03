"""Periodic capsule raster and campaign export certification."""
import numpy as np
from cufhom.pipeline.sam_shapes import rasterize_capsules
from cufhom.pipeline.rve_generator import generate_rve_main
from cufhom.pipeline.geometry import GeometryRequest, generate_geometry
from tests.helpers.raster_reference import _rasterize_fibers_python


def test_capsule_raster_matches_exhaustive_periodic_reference():
    rng = np.random.default_rng(653)
    c = rng.uniform(-5., 20., (8, 3))
    u = rng.normal(size=c.shape)
    u /= np.linalg.norm(u, axis=1)[:, None]
    u[:3] = np.eye(3)
    for length in (5., 23.):
        actual = rasterize_capsules(c, u, length, 1.13, 16)
        expected = _rasterize_fibers_python(c, u, length, 1.13, 16, 16**3+1)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        translated = rasterize_capsules(c+[32., -16., 48.], -u, length, 1.13, 16)
        np.testing.assert_array_equal(translated[0], actual[0])


def test_campaign_exports_only_certified_fields(tmp_path):
    parameters = dict(caja_um=12., L_um=2., d_um=1., Vf_target=.02,
                      a11=1., a22=0., seed=572, sam_num_threads=1)
    for name, grid, accepted in (('resolved', 36, True), ('unresolved', 3, False)):
        output = tmp_path/name
        result = generate_rve_main(dict(parameters, resol=grid/12., output_dir=output))
        assert result['accepted'] is accepted
        assert (output/'phase.npy').exists() is accepted
        assert (output/'ori.npy').exists() is accepted
        if accepted:
            with np.load(output/'geometry/geometry.npz', allow_pickle=False) as geometry:
                np.testing.assert_array_equal(np.load(output/'phase.npy'), geometry['phase'])
                np.testing.assert_array_equal(np.load(output/'ori.npy'), geometry['ori'])
            assert result['sam_final_overlap_relative'] <= 1e-6
            assert abs(result['sam_vf']-.02) <= result['sam_vf_tolerance']
        else:
            assert not result['sam_vf_ok']


def test_bootstrap_failure_saves_diagnostics_without_export(tmp_path):
    output = tmp_path/'failed'
    summary = generate_geometry(GeometryRequest(
        output=output, box=12., ar=2., vf=.02, grid=36, solver='a4',
        initial_count=6, bootstrap_max_count=6, max_stages=1,
        relax_iterations=1, max_relax_iterations=1, candidates=1, attempts=1,
        threads=1, seed=1, verbose=False))
    assert summary['status'] == 'bootstrap_failed'
    assert summary['accepted'] is False
    assert summary['accepted_count'] == 0
    assert (output/'summary.json').exists()
    assert not (output/'geometry.npz').exists()
