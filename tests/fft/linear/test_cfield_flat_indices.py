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
