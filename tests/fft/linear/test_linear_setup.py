"""Coordinate setup versus production maps and independent physical residuals."""

from types import SimpleNamespace

import numpy as np
import pytest
from cufhom.fft.linear.coordinates import (
    _build_coordinate_rhs,
    _spectral_distance,
    make_linear_solver,
)
from cufhom.fft.linear.fused import FusedPipeline
from cufhom_core import projections
from cufhom_core.general.solver import linear_solver
from cufhom_core.tensors import DFT, Operator, Tensor, fft

from tests.fft.linear.test_linear_fused_fft import _problem

cp = pytest.importorskip("cupy")


pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual", preserve_cache=True)


@pytest.mark.parametrize("shape", [(8, 8, 8), (12, 10, 14), (15, 9, 10), (13, 11, 26)])
def test_tiled_coordinate_maps_match_production(shape):
    # Even grids pack the k2 Nyquist plane into the tile-major k2 = 0 plane, so
    # the two layouts round differently there (bitwise on odd grids only).
    Q, A, rng = _problem(shape)
    pipeline = FusedPipeline(Q, A)
    tiled = Q.tiled(pipeline.config["ta"])
    field = cp.asarray(rng.normal(size=(6,) + shape).astype(np.float32))
    original, h = Q.adjoint_with_spectrum(field)
    actual, ht = tiled.adjoint_with_spectrum(field)

    def rel(a, b):
        return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))

    exact = shape[-1] % 2 == 1
    check = (
        (lambda a, b: cp.array_equal(a, b))
        if exact
        else (lambda a, b: rel(a, b) <= 1e-6)
    )
    assert cp.array_equal(h, ht)
    assert check(actual, pipeline.to_fused(original))
    assert check(tiled.strain(actual), Q.strain(original))
    assert check(pipeline.to_production(actual), original)


@pytest.mark.parametrize("shape", [(8, 10, 12), (7, 5, 9)])
@pytest.mark.parametrize("difference", [False, True])
def test_physical_spectral_norm_matches_real_parseval(shape, difference):
    rng = np.random.default_rng(831)
    fields = rng.normal(size=(2, 6) + shape)
    a, b = [
        cp.asarray(np.fft.rfftn(f, axes=(1, 2, 3)).astype(np.complex64)) for f in fields
    ]
    delta = cp.asnumpy(a - b) if difference else cp.asnumpy(a)
    # NumPy64 inverse is independent of the CUDA reduction and cuFFT.
    real = np.fft.irfftn(
        delta.astype(np.complex128), s=shape, axes=(1, 2, 3)
    ) * np.prod(shape)
    expected = np.sum(real * real) / np.prod(shape)
    actual = _spectral_distance(a, shape, b if difference else None)
    np.testing.assert_allclose(actual, expected, rtol=2e-14)


def test_rhs_omits_inverse_forward_round_trip(monkeypatch):
    Q, A, rng = _problem((8, 10, 12))
    pipeline = FusedPipeline(Q, A)
    template = Tensor(
        name="EN",
        val=cp.zeros((6,) + Q.shape, cp.float32),
        N=Q.shape,
        order=1,
        Fourier=False,
        fft_form="r",
    )
    template.add_mean(np.array([1.0, 0, 0, 0, 0, 0], np.float32))
    calls = []
    original = fft.rfftn_cached

    def forward(*args, **kwargs):
        calls.append("forward")
        return original(*args, **kwargs)

    def no_inverse(*args, **kwargs):
        raise AssertionError("Coordinate RHS made a round trip through real space")

    with monkeypatch.context() as patch:
        patch.setattr(fft, "rfftn_cached", forward)
        patch.setattr(fft, "irfftn_cached", no_inverse)
        patch.setattr(fft, "irfftn_unscaled_cached", no_inverse)
        prepared = _build_coordinate_rhs(template, A, Q, pipeline)
    assert calls == ["forward"]
    tiled, _, actual, projected, norm = prepared.cufhom_coordinate_rhs
    physical = fft.irfftn_unscaled_cached(projected, Q.shape, (1, 2, 3))
    old = pipeline.to_fused(Q.adjoint(physical) / np.float32(np.prod(Q.shape)))
    np.testing.assert_allclose(
        cp.asnumpy(actual), cp.asnumpy(old), rtol=3e-6, atol=3e-6
    )
    physical64 = cp.asnumpy(physical).astype(np.float64)
    np.testing.assert_allclose(
        norm, np.sqrt(np.sum(physical64**2) / np.prod(Q.shape)), rtol=2e-7
    )
    assert tiled.tile == pipeline.config["ta"]


@pytest.mark.parametrize("load", range(6))
@pytest.mark.parametrize("force_correction", [False, True])
def test_coordinate_rhs_passes_original_real_equation(
    load, force_correction, monkeypatch
):
    Q, A, _ = _problem((12, 10, 14))
    shape = Q.shape
    FN = DFT(N=shape, fft_form="r")
    FiN = DFT(N=shape, fft_form="r", inverse=True)
    hG = projections.elasticity_direct(
        shape, Q.P.lengths, fft_form="r", dtype=np.float32
    )
    GN = Operator(mat=[[FiN, hG, FN]])
    Afun = Operator(mat=[[GN, A]])
    par = dict(
        rtol=1e-5,
        atol=0.0,
        maxiter=1000,
        real_dtype="float32",
        keep_solution_on_device=True,
    )
    pb = SimpleNamespace(Y=Q.P.lengths, solver=par, solve={})
    adapter = make_linear_solver(
        A=A, GN=GN, pb=pb, fallback=linear_solver, bulk=2.5, shear=1.2
    )
    template = Tensor(
        name="EN",
        val=cp.zeros((6,) + shape, cp.float32),
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
    )
    macro = np.zeros(6, np.float32)
    macro[load] = 1
    template.add_mean(macro)
    prepared = adapter.build_rhs(template)
    assert prepared is not None
    if force_correction:
        import cufhom.fft.linear.coordinates as coordinates

        monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", "0")
        monkeypatch.setattr(
            coordinates,
            "fused_cg",
            lambda action, rhs, *a, **kw: (cp.zeros_like(rhs), 1),
        )
    if force_correction:
        # A failed audit fails the load; no strain-CG correction follows.
        from cufhom.fft.linear.common import LinearSolveError

        with pytest.raises(LinearSolveError) as caught:
            adapter(solver="CG", Afun=Afun, B=prepared, x0=None, par=par)
        assert caught.value.status == "fresh_residual_rejected"
        return
    result, info = adapter(solver="CG", Afun=Afun, B=prepared, x0=None, par=par)
    # Rebuild the old physical RHS and operator separately, including their
    # independent inverse FFT rounding; reported convergence cannot hide failure.
    rhs = cp.asnumpy(Afun(-template).val).astype(np.float64)
    projected = cp.asnumpy(Afun(result).val).astype(np.float64)
    true_relative = np.linalg.norm(rhs - projected) / np.linalg.norm(rhs)
    assert true_relative <= par["rtol"]
    assert info["linear_coordinate_rhs"] and info["linear_coordinate_layout"] == "tiled"
    assert info["true_residual_converged"]
    assert info["correction_iterations"] == 0
    np.testing.assert_allclose(
        info["rhs_norm"], np.linalg.norm(rhs) / np.sqrt(np.prod(shape)), rtol=2e-7
    )


def test_rhs_defers_to_existing_route_for_initial_fields():
    Q, A, _ = _problem((8, 8, 8))
    pb = SimpleNamespace(
        Y=Q.P.lengths, solver={}, solve={"initial_solution_fields": []}
    )
    adapter = make_linear_solver(
        A=A, GN=None, pb=pb, fallback=linear_solver, bulk=2.5, shear=1.2
    )
    assert adapter.build_rhs(None) is None
