"""Independent coordinate, CG restart and physical-solution checks."""

import numpy as np
import pytest
from cufhom.fft.krylov import ConvergenceError
from cufhom.fft.linear.krylov import SymmetricCoordinates, fused_cg
from cufhom.fft.operators import MixedProjector
from cufhom_core.tensors import fft
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    old_backend, old_plan = fft.FFT_BACKEND, fft.CUPY_PLAN_MODE
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual")
    yield
    fft.set_fft_backend(old_backend)
    fft.set_cupy_plan_mode(old_plan)


def real_dot(a, b):
    real = cp.float32 if a.dtype in (cp.float32, cp.complex64) else cp.float64
    return float(
        cp.sum(a.view(real).astype(cp.float64) * b.view(real).astype(cp.float64))
    )


def matrix_protocol(apply, observe=None):
    """fused_cg's step/update protocol for an explicit SPD action.

    Mirrors the production kernels: the deferred direction update precedes
    H, p.Hp is an FP64 device scalar and an invalid step poisons the norm.
    observe(new_norm, smoothing_active) may replace the returned norm.
    """

    def action_and_dot(p, *, direction=None):
        if direction is not None:
            r, beta = direction
            p[:] = r + cp.float32(beta) * p
        ap = apply(p)
        return ap, cp.asarray(real_dot(p, ap), cp.float64)

    def update(p, ap, rr, pap, x, r, smoothing=None):
        pap = float(pap)
        if not (np.isfinite(pap) and pap > 0):
            return np.nan
        alpha = np.float32(rr / pap)
        if smoothing is not None:
            y, weight = smoothing
            y += np.float32(weight) * (x - y)
        x += alpha * p
        r -= alpha * ap
        new = real_dot(r, r)
        return new if observe is None else observe(new, smoothing is not None)

    return dict(action_and_dot=action_and_dot, device_update_callback=update)


@pytest.mark.parametrize("shape", [(5, 7, 5), (6, 4, 8), (1, 3, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fused_maps_cpu_adjoint_and_owned_outputs(shape, dtype):
    lengths = (1.0, 1.7, 0.9)
    Q = SymmetricCoordinates(MixedProjector(shape, lengths, [True] * 6, cp), 4.3, 1.6)
    cpu = SymmetricCoordinates(MixedProjector(shape, lengths, [True] * 6, np), 4.3, 1.6)
    rng = np.random.default_rng(2909)
    field = rng.normal(size=(6,) + shape).astype(dtype)
    ctype = np.complex64 if dtype == np.float32 else np.complex128
    p = (rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size)).astype(ctype)
    r = (rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size)).astype(ctype)
    pg, rg, fg = cp.asarray(p), cp.asarray(r), cp.asarray(field)
    tol = 8e-7 if dtype == np.float32 else 2e-14
    result = Q.adjoint(fg)
    expected = cpu._map(np.fft.rfftn(field, axes=(1, 2, 3)), True)
    assert (
        np.linalg.norm(cp.asnumpy(result) - expected) / np.linalg.norm(expected) < tol
    )
    # Zero/mean/Nyquist and partial CUDA blocks are included in random input.
    beta = dtype(0.37)
    strain = Q.strain_updated_direction(pg, rg, beta)
    updated = r + beta * p
    h = cpu._map(updated, False)
    expected_strain = np.fft.irfftn(h, s=shape, axes=(1, 2, 3)) * np.prod(shape)
    assert (
        np.linalg.norm(cp.asnumpy(strain) - expected_strain)
        / np.linalg.norm(expected_strain)
        < tol
    )
    np.testing.assert_allclose(cp.asnumpy(pg), updated, rtol=tol, atol=tol)
    np.testing.assert_array_equal(cp.asnumpy(rg), r)
    np.testing.assert_array_equal(cp.asnumpy(fg), field)
    # Subsequent calls cannot overwrite an earlier result.
    saved_strain = cp.asnumpy(strain)
    Q.strain_updated_direction(pg, rg, beta)
    np.testing.assert_array_equal(cp.asnumpy(strain), saved_strain)


@pytest.mark.parametrize("smoothing", [False, True])
def test_cg_reliable_restart_with_deferred_direction(smoothing):
    diagonal = cp.linspace(1.0, 10.0, 129, dtype=cp.float32)
    rhs = cp.ones(129, cp.complex64)

    def action(x):
        return diagonal * x

    directions, count = [], [0]

    def false_recursive_once(new, active):
        count[0] += 1
        return 0.0 if count[0] == 1 else new

    protocol = matrix_protocol(action, false_recursive_once)
    inner = protocol["action_and_dot"]

    def action_and_dot(p, *, direction=None):
        directions.append(direction is not None)
        return inner(p, direction=direction)

    protocol["action_and_dot"] = action_and_dot
    x, iterations = fused_cg(
        action, rhs, real_dot, 1e-5, 100, residual_smoothing=smoothing, **protocol
    )
    assert iterations > 2
    assert directions[:2] == [False, False]  # first action and reliable restart
    assert any(directions[2:])
    residual = rhs - action(x)
    assert np.sqrt(real_dot(residual, residual) / real_dot(rhs, rhs)) <= 1e-5
    np.testing.assert_allclose(cp.asnumpy(x), 1 / cp.asnumpy(diagonal), rtol=5e-5)
    np.testing.assert_array_equal(cp.asnumpy(rhs), np.ones(129))


@pytest.mark.parametrize("false_estimate", [False, True])
def test_late_smoothing_dense_spd_and_replacement(false_estimate):
    rng = np.random.default_rng(3009)
    q, _ = np.linalg.qr(rng.normal(size=(129, 129)))
    matrix = (q * np.geomspace(1.0, 80.0, 129)) @ q.T
    b = rng.normal(size=129).astype(np.float32)
    a, rhs = cp.asarray(matrix, dtype=cp.float32), cp.asarray(b)
    calls, corrupted = [], [False]

    def observe(new, active):
        calls.append(active)
        # Corrupt the estimate AFTER smoothing starts, so the audit must
        # reset the delayed blend as well as the CG search direction.
        if active and false_estimate and not corrupted[0]:
            corrupted[0] = True
            return 0.0
        return new

    x, _ = fused_cg(
        lambda v: a @ v,
        rhs,
        real_dot,
        1e-5,
        250,
        residual_smoothing=True,
        **matrix_protocol(lambda v: a @ v, observe),
    )
    result = cp.asnumpy(x).astype(np.float64)
    assert not calls[0] and any(calls)
    if false_estimate:
        assert corrupted[0] and not calls[calls.index(True) + 1]
    # Independent CPU64 equation and solution, rather than the recurrence.
    assert np.linalg.norm(b - matrix @ result) / np.linalg.norm(b) < 1.1e-5
    ref = np.linalg.solve(matrix, b)
    assert np.linalg.norm(result - ref) / np.linalg.norm(ref) < 8e-5
    np.testing.assert_array_equal(cp.asnumpy(rhs), b)


def test_smoothing_zero_rhs_and_budget():
    rhs = cp.zeros(16, cp.complex64)

    def identity(v):
        return v

    x, iterations = fused_cg(
        identity,
        rhs,
        real_dot,
        1e-5,
        10,
        residual_smoothing=True,
        **matrix_protocol(identity),
    )
    assert iterations == 0 and not bool(cp.any(x))
    rhs[:] = 1

    def ramp(v):
        return cp.arange(1, 17, dtype=cp.float32) * v

    with pytest.raises(ConvergenceError, match="true residual"):
        fused_cg(
            ramp,
            rhs,
            real_dot,
            1e-5,
            1,
            residual_smoothing=True,
            **matrix_protocol(ramp),
        )


def test_fused_cg_rejects_nonpositive_energy():
    rhs = cp.ones(8, cp.complex64)

    def negative(v):
        return -v

    with pytest.raises(ConvergenceError, match="Nonfinite"):
        fused_cg(negative, rhs, real_dot, 1e-5, 10, **matrix_protocol(negative))
    with pytest.raises(ValueError, match="device denominator"):
        fused_cg(
            negative,
            rhs,
            real_dot,
            1e-5,
            10,
            action_and_dot=lambda p, **kw: (p.copy(), -1.0),
            device_update_callback=lambda *a: 0.0,
        )
