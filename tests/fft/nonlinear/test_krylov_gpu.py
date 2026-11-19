"""Exercise wide-scalar GMRES CUDA kernels with native FP32 work fields."""

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear.gmres import euclidean_dot, gmres

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("amplitude", [2e38, 1e-39, 1e-44])
@pytest.mark.parametrize("kind", ["identity", "diagonal", "real_linear"])
def test_cuda_gmres_extreme_scales(dtype, amplitude, kind):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    host = np.zeros(32, dtype=dtype)
    host.real[::2] = amplitude
    if host.dtype.kind == "c":
        host.imag[::2] = host.real[::2] * np.float32(0.5)
    storage = cp.asarray(host)
    rhs = storage[::2]
    saved = storage.copy()
    diagonal = cp.linspace(0.7, 1.0, 16, dtype=rhs.real.dtype)
    backing = cp.empty_like(storage)

    def action(value):
        if kind == "identity":
            return value
        if kind == "diagonal":
            backing[::2] = diagonal * value
            return backing[::2]
        result = diagonal * value + np.float32(0.03) * value.conj()
        result[:-1] += np.float32(0.02) * value[1:]
        return result

    try:
        solution, count = gmres(action, rhs, euclidean_dot, 1e-5, 120, restart=4)
    except NonlinearConvergenceError as error:
        # A device that flushes stored FP32 subnormals must reject them,
        # instead of returning x=0 with an alleged exact-zero RHS.
        assert amplitude < np.finfo(np.float32).tiny
        assert "underflowed" in str(error)
        assert np.any(cp.asnumpy(rhs) != 0)
        cp.testing.assert_array_equal(storage, saved)
        return
    residual = rhs - action(solution)
    assert np.sqrt(euclidean_dot(residual, residual) / euclidean_dot(rhs, rhs)) <= 1e-5
    assert solution.dtype == rhs.dtype and bool(cp.isfinite(solution).all())
    assert 1 <= count <= 120
    cp.testing.assert_array_equal(storage, saved)
