"""Safety checks for diagnostic-only CUDA graph and fused-product experiments."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from benchmark_cuda_graph_operator import FrozenTangentGraph, cp, fft
from benchmark_fused_adjoint_dot import FusedAdjointDot
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    old = fft.get_fft_backend_status()
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    fft.set_cupy_rfft_batch_mode('component')
    yield
    cp.cuda.get_current_stream().synchronize()
    fft.set_fft_backend(old['requested'])
    fft.set_cupy_plan_mode(old['cupy_plan_mode'])
    fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode'])


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('amplitude', [1e-6, .01])
@pytest.mark.parametrize('mask', [[True,False,False,False,False,False], [True]*6])
def test_graph_new_input_and_immutable_state(dtype, amplitude, mask):
    shape = (6,8,10)
    rng = np.random.default_rng(410)
    law = J2Material()
    ids = cp.asarray(rng.integers(0,3,size=shape), dtype=cp.int32)
    table = rng.normal(size=(3,6,6)); table = table@table.transpose(0,2,1)+np.eye(6)
    material = CuPyMaterial(cp, law, ids, cp.asarray(table, dtype=dtype), dtype)
    state = initial_state(shape, cp, dtype)
    strain = cp.asarray(amplitude*rng.normal(size=(6,)+shape), dtype=dtype)
    response = material.update(strain, state)
    saved = [(value,value.copy()) for value in (*vars(state).values(), *vars(response).values())]
    Q = DisplacementCoordinates(MixedProjector(shape, (1.,2.,3.), mask, cp))
    z = Q.adjoint(strain)
    snapshot = z.copy()
    candidate = FrozenTangentGraph(Q, material, response, z)
    eps = np.finfo(dtype).eps
    expected = Q.adjoint(material.tangent(Q.strain(z), response))
    cp.testing.assert_allclose(candidate.run_raw(), expected, rtol=32*eps, atol=32*eps)
    for factor in (1., -1.3, .4, 1.):
        value = z*dtype(factor)
        expected = Q.adjoint(material.tangent(Q.strain(value), response))
        # A stream different from the capture stream must see fresh input.
        cp.testing.assert_allclose(candidate.run_graph_with_copy(value), expected,
                                   rtol=32*eps, atol=32*eps)
    cp.testing.assert_array_equal(z, snapshot)
    for current, original in saved:
        cp.testing.assert_array_equal(current, original)


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('shape', [(5,7,9), (6,8,10), (1,1,2)])
@pytest.mark.parametrize('mask', [[True,False,False,False,False,False], [False]*6, [True]*6])
def test_fused_map_dot_partial_blocks_nyquist_and_means(dtype, shape, mask):
    Q = DisplacementCoordinates(MixedProjector(shape, (1.,2.,3.), mask, cp))
    rng = np.random.default_rng(221)
    h = cp.asarray(rng.normal(size=(6,)+Q.hshape)+1j*rng.normal(size=(6,)+Q.hshape), dtype=dtype)
    p = cp.asarray(rng.normal(size=Q.size)+1j*rng.normal(size=Q.size), dtype=dtype)
    hs, ps = h.copy(), p.copy()
    expected = Q._map(h, True)
    # Independent FP64 product, not the candidate reduction tree.
    ph = cp.asnumpy(p).astype(np.complex128)
    yh = cp.asnumpy(expected).astype(np.complex128)
    reference = np.vdot(ph, yh).real
    candidate = FusedAdjointDot(Q, dtype)
    actual, scalar = candidate(h, p)
    eps = np.finfo(p.real.dtype).eps
    cp.testing.assert_allclose(actual, expected, rtol=16*eps, atol=16*eps)
    scale = max(1., np.linalg.norm(ph)*np.linalg.norm(yh))
    assert abs(float(scalar)-reference) <= 16*eps*scale
    cp.testing.assert_array_equal(h, hs); cp.testing.assert_array_equal(p, ps)
    # Near-zero products cannot be checked with an undefined relative error.
    _, zero = candidate(h, cp.zeros_like(p))
    assert float(zero) == 0
    with pytest.raises(ValueError):
        candidate(h, p[:-1])
