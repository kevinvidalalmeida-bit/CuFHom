"""Indexed GPU construction must preserve material IDs and constitutive tables."""
import numpy as np
import pytest

import cufhom.fft.linear.solver as solver


@pytest.mark.parametrize('fibers', ['none', 'mixed', 'all'])
@pytest.mark.parametrize('layout_mode', ['new', 'legacy', 'none'])
def test_flat_indexed_field_matches_dense(fibers, layout_mode):
    cp = solver.cp
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    phase = np.zeros((3, 4, 5), np.uint8)
    if fibers == 'mixed':
        phase[::2, 1:3, ::2] = 1
    elif fibers == 'all':
        phase[:] = 1
    ori = np.random.default_rng(5).normal(size=phase.shape+(3,)).astype('float32')
    phase.flags.writeable = ori.flags.writeable = False
    layout = solver.compile_cfield_geometry(phase, ori)
    np.testing.assert_array_equal(layout['fiber_flat_indices'], np.flatnonzero(phase == 1))
    assert not layout['fiber_flat_indices'].flags.writeable
    if layout_mode == 'legacy':
        layout.pop('fiber_flat_indices')
    elif layout_mode == 'none':
        layout = None
    cm = np.eye(6, dtype='float32')*3
    cf = np.diag(np.arange(1, 7, dtype='float32'))
    (index, table), groups, _ = solver._build_cfield_gpu(
        phase, ori, cm, cf, indexed=True, storage='sym21', geometry_layout=layout)
    dense, dense_groups, _ = solver._build_cfield_gpu(
        phase, ori, cm, cf, indexed=False, storage='sym21', geometry_layout=layout)
    assert groups == dense_groups
    cp.testing.assert_array_equal(table[index].transpose(3, 0, 1, 2), dense)


@pytest.mark.parametrize('backend,dtype', [
    ('cpu', np.float32), ('cpu', np.float64),
    ('gpu', np.float32), ('gpu', np.float64), ('gpu_indexed', np.float32),
])
@pytest.mark.parametrize('fibers', ['none', 'mixed', 'all'])
def test_affine_derivatives_match_perturbed_material_fields(backend, dtype, fibers):
    xp = np if backend == 'cpu' else solver.cp
    if backend != 'cpu':
        if xp is None:
            pytest.skip('CuPy unavailable')
        try:
            if not xp.cuda.runtime.getDeviceCount():
                pytest.skip('No CUDA device')
        except xp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
    phase = np.zeros((3, 4, 5), np.uint8)
    if fibers == 'mixed':
        phase[::2, 1:3, ::2] = 1
    elif fibers == 'all':
        phase[:] = 1
    ori = np.random.default_rng(572).normal(size=phase.shape+(3,))
    if backend == 'gpu_indexed':
        operators, _, _ = solver._build_affine_sensitivity_operator_specs_gpu(
            phase, ori, dtype, assign_chunk_voxels=7)
        derivatives = xp.stack([
            xp.moveaxis(op['table'][op['index']], -1, 0) for op in operators])
    else:
        derivatives = solver._build_affine_sensitivity_cfields(
            phase, ori, dtype, xp=xp, assign_chunk_voxels=7)
    if backend != 'cpu':
        derivatives = xp.asnumpy(derivatives)

    def stiffness(coefficients):
        lam, mu, c_tt, c_cross, c_lt, c_ll, g_lt = coefficients
        matrix = np.eye(6)*2*mu
        matrix[:3, :3] += lam
        fiber = np.diag([c_ll, c_tt, c_tt, c_tt-c_cross, 2*g_lt, 2*g_lt])
        fiber[0, 1] = fiber[1, 0] = fiber[0, 2] = fiber[2, 0] = c_lt
        fiber[1, 2] = fiber[2, 1] = c_cross
        return matrix, fiber

    # Differentiate the assembled physical stiffness, including axis rotations,
    # without using the derivative builder's affine bases as the reference.
    coefficients = np.array([2., 1.2, 14., 2., 3., 24., 5.])
    step = 1e-4
    for coefficient in range(7):
        plus, minus = coefficients.copy(), coefficients.copy()
        plus[coefficient] += step
        minus[coefficient] -= step
        positive = solver._build_cfield_cpu(phase, ori, *stiffness(plus), np.float64,
                                            storage='sym21')[0]
        negative = solver._build_cfield_cpu(phase, ori, *stiffness(minus), np.float64,
                                            storage='sym21')[0]
        finite_difference = (positive-negative)/(2*step)
        tolerance = 2e-6 if dtype == np.float32 else 2e-9
        np.testing.assert_allclose(derivatives[coefficient], finite_difference,
                                   rtol=tolerance, atol=tolerance)
