"""Generated HEX8 avoids unused cuFFT resources, retaining its operator."""

import numpy as np
import pytest
from cufhom.fft.linear import hex8_fused
from cufhom.fft.linear.hex8_general import Hex8Operator, _CudaPCG, cp, homogenize_hex8

pytestmark = pytest.mark.gpu


@pytest.fixture
def cuda():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp


def _problem(shape):
    rng = np.random.default_rng(22026)
    a = rng.normal(size=(2, 6, 6))
    stiffness = a.transpose(0, 2, 1) @ a + 3 * np.eye(6)
    stiffness[1] *= 4
    x, y, z = np.indices(shape)
    labels = ((x >= shape[0] // 2) | ((y == 1) & (z == 2))).astype(np.int32)
    return labels, stiffness


@pytest.mark.parametrize("working", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(4, 6, 8), (8, 8, 8)])
def test_generated_preconditioner_never_constructs_cufft(
    cuda, monkeypatch, shape, working
):
    """Fail immediately if the generated path allocates any library FFT plan.

    Then compare its actual correction and Parseval energy with the complete
    cuFFT route. This exercises both fresh-residual preparation and direction
    initialization; no mocked numerical kernels are involved.
    """
    import cupyx.scipy.fft

    labels, stiffness = _problem(shape)
    op = Hex8Operator(labels, stiffness, lengths=(0.8, 1.3, 1.1), working=working)

    def forbidden(*args, **kwargs):
        pytest.fail("Generated HEX8 constructed an unused cuFFT plan")

    with monkeypatch.context() as patch:
        patch.setattr(cupyx.scipy.fft, "get_fft_plan", forbidden)
        generated = hex8_fused.FusedCudaPCG(op)
    assert generated.forward is generated.backward is generated.spectrum is None
    assert generated.rho is generated.rmax is None
    library = _CudaPCG(op)
    residual = np.random.default_rng(22027).normal(size=(3, *shape)).astype(working)
    residual -= residual.mean(axis=(1, 2, 3), keepdims=True)
    energies = []
    for pcg in (generated, library):
        pcg.reset_history(2)
        cp.copyto(pcg.r, cp.asarray(residual))
        energies.append(pcg.prepare(0, 0.0, 1))
        pcg.proceed(1)
    tolerance = 3e-6 if working == "float32" else 3e-13
    expected = cp.asnumpy(library.p)
    np.testing.assert_allclose(
        cp.asnumpy(generated.p),
        expected,
        atol=tolerance * np.abs(expected).max(),
        rtol=tolerance,
    )
    assert energies[0] == pytest.approx(energies[1], rel=tolerance)


@pytest.mark.parametrize(
    "dtype,working,rtol,agreement",
    [
        ("float32", "float32", 2e-5, 3e-6),
        ("float64", "float32", 1e-8, 3e-11),
        ("float64", "float64", 1e-9, 3e-12),
    ],
)
def test_six_loads_with_generated_and_library_preconditioners(
    cuda, monkeypatch, dtype, working, rtol, agreement
):
    """The allocation-only change preserves all six fresh physical audits."""
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, stiffness = _problem((6, 8, 10))
    options = dict(lengths=(0.8, 1.3, 1.1), dtype=dtype, working=working, rtol=rtol)
    generated = homogenize_hex8(labels, stiffness, **options)
    monkeypatch.setattr(hex8_fused, "fused_pcg", lambda op: None)
    library = homogenize_hex8(labels, stiffness, **options)
    assert len(generated["infos"]) == len(library["infos"]) == 6
    for result, fft in ((generated, "generated"), (library, "cufft")):
        assert all(
            info["accepted"]
            and info["residual_inf"] <= info["threshold"]
            and info["preconditioner_fft"] == fft
            for info in result["infos"]
        )
        np.testing.assert_allclose(result["C"], result["C"].T, rtol=0, atol=0)
    np.testing.assert_allclose(
        generated["C"],
        library["C"],
        rtol=agreement,
        atol=agreement * np.abs(library["C"]).max(),
    )
