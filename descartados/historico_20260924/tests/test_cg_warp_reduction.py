"""Numerical and ownership checks of optional warp-based GPU CG reductions."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/ffthompy_core/ffthompy'))
cp = pytest.importorskip('cupy')
from ffthompy.tensors import Tensor
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('size', [1, 31, 257, 4099])
def test_dot_update_and_independent_scalars(dtype, size):
    rng = np.random.default_rng(193)
    vals = [rng.normal(size=(2, size)).astype(dtype) for _ in range(4)]
    tensors = [Tensor(val=cp.asarray(a), order=1, fft_form='r') for a in vals]
    x, p, r, ap = tensors
    reduction = CupyWarpCGReduction(x, max_blocks=3)
    rr = reduction.dot(r, r)
    saved = float(rr)
    dot = reduction.dot(p, ap)
    expected = np.sum(vals[1].astype(np.float64)*vals[3].astype(np.float64))/size
    tol = 4e-7 if dtype == np.float32 else 2e-14
    np.testing.assert_allclose(float(dot), expected, atol=tol, rtol=tol)
    alpha = cp.asarray(.125, dtype=dtype)
    next_rr = reduction.update_xr_rr(x, alpha, p, r, ap)
    np.testing.assert_allclose(cp.asnumpy(x.val), vals[0]+.125*vals[1], atol=tol, rtol=tol)
    np.testing.assert_allclose(cp.asnumpy(r.val), vals[2]-.125*vals[3], atol=tol, rtol=tol)
    expected_rr = float(cp.sum(r.val.astype(cp.float64)**2)/size)
    np.testing.assert_allclose(float(next_rr), expected_rr, rtol=tol)
    assert float(rr) == saved  # Later reductions must not overwrite CG's old rr.


def test_strided_fallback_and_configuration():
    a = Tensor(val=cp.ones((2, 8), dtype=cp.float32), order=1, fft_form='r')
    reduction = CupyWarpCGReduction(a)
    strided = a.copy(val=cp.ones((2, 16), dtype=cp.float32)[:, ::2])
    assert reduction.update_xr_rr(strided, cp.float32(.5), a, a, a) is None
    assert not reduction.supports(strided, {})
    assert not reduction.supports(a, {'scal': lambda x, y: 0})
    with pytest.raises(ValueError):
        CupyWarpCGReduction(a, max_blocks=0)
