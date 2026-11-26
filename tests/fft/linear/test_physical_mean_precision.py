"""Macroscopic stresses must not lose the small mean of cancelling FP32 data."""

import numpy as np
import pytest
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_physical_mean_retains_cancellation_remainder(backend, dtype):
    if backend == "cupy":
        if cp is None:
            pytest.skip("CuPy unavailable")
        try:
            if not cp.cuda.runtime.getDeviceCount():
                pytest.skip("CUDA unavailable")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA unavailable")
    xp = np if backend == "numpy" else cp
    values = np.array([1e8, 1.0, -1e8], dtype=dtype).reshape(1, 3, 1, 1)
    # Each voxel value is exactly representable in FP32. Its nonzero mean
    # must survive cancellation; casting an FP32 mean afterwards cannot help.
    field = Tensor(
        name="stress",
        val=xp.asarray(values),
        N=(3, 1, 1),
        order=1,
        Fourier=False,
        fft_form="r",
    )
    actual = field.mean()
    np.testing.assert_array_equal(actual, np.array([1 / 3]))
    assert field.val.dtype == dtype
    np.testing.assert_array_equal(
        xp.asnumpy(field.val) if backend == "cupy" else field.val, values
    )


def test_physical_mandel_mean_is_componentwise():
    values = np.array([1e8, 1.0, -1e8], np.float32)
    field = Tensor(
        name="stress",
        val=np.stack([values * i for i in range(1, 7)]).reshape(6, 3, 1, 1),
        N=(3, 1, 1),
        order=1,
        Fourier=False,
        fft_form="r",
    )
    np.testing.assert_array_equal(field.mean(), np.arange(1, 7, dtype=float) / 3)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cupy_mandel_mean_matches_fp64_host_mean(dtype):
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    values = np.random.default_rng(3).normal(size=(6, 24, 20, 18)).astype(
        dtype
    ) + np.arange(6).reshape(6, 1, 1, 1)
    field = Tensor(
        name="stress",
        val=cp.asarray(values),
        N=(24, 20, 18),
        order=1,
        Fourier=False,
        fft_form="r",
    )
    expected = values.reshape(6, -1).astype(np.float64).mean(axis=1)
    actual = field.mean()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
    # Fixed reduction tree: repeated means are bit-identical.
    np.testing.assert_array_equal(field.mean(), actual)
