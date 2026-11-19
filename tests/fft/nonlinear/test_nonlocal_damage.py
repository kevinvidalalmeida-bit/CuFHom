"""GPU-only analytic and variational checks of periodic damage regularization.

The independent operator oracle uses full complex FFTs, whereas production
uses real half spectra. CPU operations are limited to scalar assertions and
configuration; no CPU FFT, PDE solve or reference linear solve is used.
"""

import math
from contextlib import contextmanager

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear.nonlocal_damage import (
    NonlocalConvergenceError,
    NonlocalHelmholtz,
)
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")


def tolerance(dtype):
    return 3e-6 if dtype == "float32" else 1e-11


def dot(x, y):
    return float(
        cp.sum(x.astype(cp.float64) * y.astype(cp.float64), dtype=cp.float64).item()
    )


def coordinates(shape, dtype):
    return [
        cp.arange(n, dtype=dtype).reshape(
            tuple(n if j == axis else 1 for j in range(3))
        )
        / n
        for axis, n in enumerate(shape)
    ]


def full_gradient(field, lengths):
    """Independent full-complex Fourier derivative, including Nyquist planes."""
    shape = field.shape
    valid = cp.ones(shape, dtype=cp.bool_)
    for axis, n in enumerate(shape):
        if n % 2 == 0:
            plane = [slice(None)] * 3
            plane[axis] = n // 2
            valid[tuple(plane)] = False
    spectrum = cp.fft.fftn(field)
    waves = [
        (2 * math.pi * cp.fft.fftfreq(n, d=length / n))
        .astype(field.dtype)
        .reshape(tuple(n if j == axis else 1 for j in range(3)))
        for axis, (n, length) in enumerate(zip(shape, lengths))
    ]
    return [cp.fft.ifftn(1j * k * valid * spectrum).real for k in waves], waves, valid


def reference_action(field, lengths, ell, mask):
    gradients, waves, valid = full_gradient(field, lengths)
    result = field.copy()
    for gradient, k in zip(gradients, waves):
        flux = field.dtype.type(ell * ell) * mask * gradient
        result -= cp.fft.ifftn(1j * k * valid * cp.fft.fftn(flux)).real
    return result


def assert_audit(op, source, field, diagnostics, rtol):
    residual = source - op.apply(field)
    ratio = math.sqrt(dot(residual, residual) / dot(source, source))
    assert diagnostics["converged"]
    assert diagnostics["relative_residual"] <= rtol
    assert diagnostics["residual"] == diagnostics["relative_residual"]
    assert ratio <= rtol
    assert diagnostics["relative_residual"] == pytest.approx(
        ratio, rel=1e-10, abs=1e-16
    )
    mean_source = float(cp.mean(source, dtype=cp.float64).item())
    mean_field = float(cp.mean(field, dtype=cp.float64).item())
    assert mean_field == pytest.approx(
        mean_source,
        rel=0,
        abs=4 * np.finfo(field.dtype).eps * max(1.0, abs(mean_source)),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_constant_source_preserved(dtype):
    shape = (10, 8, 12)
    source = cp.full(shape, 0.375, dtype=dtype)
    op = NonlocalHelmholtz(
        shape, (0.8, 1.2, 0.9), 0.07, cp.ones(shape, cp.bool_), dtype=dtype
    )
    field, diagnostics = op.solve(source)
    cp.testing.assert_array_equal(field, source)
    assert diagnostics["method"] == "fft" and diagnostics["iterations"] == 0
    assert_audit(op, source, field, diagnostics, 1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(9, 7, 11), (10, 8, 12)])
def test_analytic_sinusoid_attenuation(dtype, shape):
    lengths, ell = (0.8, 1.2, 0.9), 0.025
    x, y, z = coordinates(shape, dtype)
    mode1 = cp.cos(2 * math.pi * (x + 2 * y - z))
    mode2 = cp.sin(2 * math.pi * (2 * x - y)) + cp.zeros(shape, dtype=dtype)
    source = 0.4 + 0.2 * mode1 + 0.1 * mode2
    k1_squared = (2 * math.pi) ** 2 * sum(
        (m / length) ** 2 for m, length in zip((1, 2, -1), lengths)
    )
    k2_squared = (2 * math.pi) ** 2 * ((2 / lengths[0]) ** 2 + (1 / lengths[1]) ** 2)
    exact = 0.4 + 0.2 * mode1 / (1 + ell**2 * k1_squared)
    exact += 0.1 * mode2 / (1 + ell**2 * k2_squared)
    op = NonlocalHelmholtz(shape, lengths, ell, cp.ones(shape, cp.bool_), dtype=dtype)
    field, diagnostics = op.solve(source, rtol=tolerance(dtype))
    cp.testing.assert_allclose(
        field, exact, rtol=3 * tolerance(dtype), atol=tolerance(dtype)
    )
    assert diagnostics["method"] == "fft" and diagnostics["preconditioner_calls"] == 1
    assert_audit(op, source, field, diagnostics, tolerance(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_nyquist_planes_keep_mass_and_have_zero_gradient(dtype, axis):
    shape = (10, 8, 12)
    indices = cp.arange(shape[axis]).reshape(
        tuple(shape[axis] if j == axis else 1 for j in range(3))
    )
    x = coordinates(shape, dtype)[(axis + 1) % 3]
    # A cross mode on a Nyquist plane must lose ALL gradient components.
    source = (
        cp.where(indices % 2, -1, 1) * cp.cos(2 * math.pi * x)
        + cp.zeros(shape, dtype=dtype)
    ).astype(dtype)
    op = NonlocalHelmholtz(
        shape, (1.0, 1.0, 1.0), 0.035, cp.ones(shape, cp.bool_), dtype=dtype
    )
    cp.testing.assert_allclose(op.apply(source), source, rtol=0, atol=tolerance(dtype))
    field, diagnostics = op.solve(source, rtol=tolerance(dtype))
    cp.testing.assert_allclose(field, source, rtol=0, atol=tolerance(dtype))
    assert_audit(op, source, field, diagnostics, tolerance(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(9, 7, 11), (10, 8, 12)])
def test_variable_coefficient_symmetry_energy_and_independent_action(dtype, shape):
    lengths, ell = (0.8, 1.2, 0.9), 0.045
    mask = cp.ones(shape, cp.bool_)
    mask[2:5, 1:4, :] = False
    rng = cp.random.RandomState(16)
    u = rng.standard_normal(shape).astype(dtype)
    v = rng.standard_normal(shape).astype(dtype)
    op = NonlocalHelmholtz(shape, lengths, ell, mask, dtype=dtype)
    au, av = op.apply(u), op.apply(v)
    expected = reference_action(u, lengths, ell, mask)
    cp.testing.assert_allclose(
        au, expected, rtol=3 * tolerance(dtype), atol=3 * tolerance(dtype)
    )
    assert dot(u, av) == pytest.approx(
        dot(au, v),
        rel=5 * tolerance(dtype),
        abs=5 * tolerance(dtype) * math.sqrt(dot(u, u) * dot(v, v)),
    )
    gradients, _, _ = full_gradient(u, lengths)
    energy = dot(u, u) + float(np.dtype(dtype).type(ell**2)) * sum(
        dot(mask * gradient, gradient) for gradient in gradients
    )
    assert dot(u, au) == pytest.approx(energy, rel=5 * tolerance(dtype))
    assert dot(u, au) >= dot(u, u) * (1 - 5 * tolerance(dtype))
    assert dot(u, au) > 0
    # Coefficients are copied; subsequent caller mask mutation cannot alter A.
    mask.fill(False)
    cp.testing.assert_array_equal(op.apply(u), au)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("shape", [(9, 7, 11), (10, 8, 12)])
def test_composite_true_residual_and_global_mean(dtype, shape):
    lengths, ell = (0.8, 1.2, 0.9), 0.035
    mask = cp.ones(shape, cp.bool_)
    mask[2:5, 1:4, :] = False
    x, y, z = coordinates(shape, dtype)
    source = (
        mask * (1 + 0.2 * cp.cos(2 * math.pi * (x + y)) + 0.1 * cp.sin(2 * math.pi * z))
    ).astype(dtype)
    original = source.copy()
    op = NonlocalHelmholtz(shape, lengths, ell, mask, dtype=dtype)
    field, diagnostics = op.solve(source, rtol=tolerance(dtype))
    assert isinstance(field, cp.ndarray) and field.dtype == np.dtype(dtype)
    assert diagnostics["method"] == "pcg" and 0 < diagnostics["iterations"] <= 500
    assert_audit(op, source, field, diagnostics, tolerance(dtype))
    residual = source - reference_action(field, lengths, ell, mask)
    assert math.sqrt(dot(residual, residual) / dot(source, source)) <= 3 * tolerance(
        dtype
    )
    coefficient = float(cp.mean(mask, dtype=cp.float64).item()) * float(
        np.dtype(dtype).type(ell**2)
    )
    assert diagnostics["preconditioner_coefficient"] == pytest.approx(
        coefficient, rel=2e-15
    )
    cp.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_identity_zero_source_and_warm_initial_reuse(dtype):
    shape = (9, 7, 11)
    mask = cp.ones(shape, cp.bool_)
    mask[2:5, :, :] = False
    source = (mask * cp.random.RandomState(28).uniform(0.2, 1.0, size=shape)).astype(
        dtype
    )
    identity = NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.0, mask, dtype=dtype)
    field, diagnostics = identity.solve(source)
    cp.testing.assert_array_equal(field, source)
    assert diagnostics["method"] == "identity" and diagnostics["residual"] == 0
    op = NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.035, mask, dtype=dtype)
    field, first = op.solve(source, rtol=tolerance(dtype))
    initial, saved = field.copy(), field.copy()
    warm, second = op.solve(source, rtol=tolerance(dtype), initial=initial)
    assert second["iterations"] <= first["iterations"]
    assert_audit(op, source, warm, second, tolerance(dtype))
    cp.testing.assert_array_equal(initial, saved)
    # A later solve cannot overwrite returned storage or the initial guess.
    op.solve(source * 0.75, rtol=tolerance(dtype))
    cp.testing.assert_array_equal(field, saved)
    zero, diagnostics = op.solve(cp.zeros_like(source), initial=initial)
    cp.testing.assert_array_equal(zero, cp.zeros_like(source))
    assert diagnostics["method"] == "zero" and diagnostics["iterations"] == 0
    cp.testing.assert_array_equal(initial, saved)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_all_fiber_mask_has_mass_only_and_zero_supported_source(dtype):
    shape = (5, 7, 9)
    op = NonlocalHelmholtz(
        shape, (1.0, 1.0, 1.0), 0.08, cp.zeros(shape, cp.bool_), dtype=dtype
    )
    initial = cp.random.RandomState(30).standard_normal(shape).astype(dtype)
    cp.testing.assert_array_equal(op.apply(initial), initial)
    field, diagnostics = op.solve(cp.zeros_like(initial), initial=initial)
    cp.testing.assert_array_equal(field, cp.zeros_like(initial))
    assert diagnostics["residual"] == 0 and diagnostics["operator_calls"] == 0
    with pytest.raises(ValueError, match="zero in fibers"):
        op.solve(cp.ones_like(initial))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_spectral_positivity_is_not_a_pointwise_clamp(dtype):
    # An impulse with an unsmoothed Nyquist component has negative lobes.
    shape = (8, 1, 1)
    source = cp.zeros(shape, dtype=dtype)
    source[0, 0, 0] = 1
    op = NonlocalHelmholtz(
        shape, (1.0, 1.0, 1.0), 0.3, cp.ones(shape, cp.bool_), dtype=dtype
    )
    rtol = 1e-5 if dtype == "float32" else 1e-11
    field, diagnostics = op.solve(source, rtol=rtol)
    assert float(field[3, 0, 0].item()) < -0.01
    assert_audit(op, source, field, diagnostics, rtol)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_iteration_exhaustion_raises_with_true_residual_diagnostics(dtype):
    shape = (9, 7, 11)
    mask = cp.ones(shape, cp.bool_)
    mask[2:6, 1:5, :] = False
    source = (mask * cp.random.RandomState(17).uniform(0.1, 1.0, size=shape)).astype(
        dtype
    )
    op = NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.12, mask, dtype=dtype)
    with pytest.raises(NonlocalConvergenceError, match="true residual") as captured:
        op.solve(source, rtol=tolerance(dtype), maxiter=1)
    assert isinstance(captured.value, NonlinearConvergenceError)
    diagnostics = captured.value.diagnostics
    assert not diagnostics["converged"] and diagnostics["iterations"] == 1
    assert diagnostics["residual"] > tolerance(dtype)
    # The same instance remains usable after rejection.
    field, diagnostics = op.solve(source, rtol=tolerance(dtype))
    assert_audit(op, source, field, diagnostics, tolerance(dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rejects_nonfinite_inputs_and_nonzero_fiber_source(dtype):
    shape = (5, 5, 5)
    mask = cp.ones(shape, cp.bool_)
    mask[0, 0, 0] = False
    op = NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.03, mask, dtype=dtype)
    source = cp.ones(shape, dtype=dtype)
    with pytest.raises(ValueError, match="zero in fibers"):
        op.solve(source)
    source[0, 0, 0] = 0
    source[1, 0, 0] = cp.nan
    with pytest.raises(ValueError, match="source must be finite"):
        op.solve(source)
    source[1, 0, 0] = 1
    initial = cp.zeros_like(source)
    initial[0, 0, 0] = cp.inf
    with pytest.raises(ValueError, match="initial must be finite"):
        op.solve(source, initial=initial)
    with pytest.raises(TypeError, match="CuPy array"):
        op.solve(np.zeros(shape, dtype=dtype))
    with pytest.raises(ValueError, match="shape, dtype and device"):
        op.solve(cp.zeros((4, 5, 5), dtype=dtype))
    with pytest.raises(ValueError, match="shape, dtype and device"):
        op.solve(cp.zeros(shape, dtype=cp.complex64))
    for rtol in (0, -1, 1, math.nan, math.inf):
        with pytest.raises(ValueError, match="rtol"):
            op.solve(source, rtol=rtol)
    with pytest.raises(ValueError, match="maxiter"):
        op.solve(source, maxiter=0)


@pytest.mark.parametrize("bad_mask", [-1.0, 0.5, math.nan, math.inf])
def test_rejects_nonbinary_coefficient(bad_mask):
    mask = cp.ones((5, 5, 5), dtype=cp.float32)
    mask[0, 0, 0] = bad_mask
    with pytest.raises(ValueError, match="only zero and one"):
        NonlocalHelmholtz(mask.shape, (1.0, 1.0, 1.0), 0.03, mask)


def test_configuration_and_optional_timers():
    shape = (5, 5, 5)
    mask = cp.ones(shape, cp.bool_)
    with pytest.raises(ValueError, match="GPU-only"):
        NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.03, mask, xp=np)
    for ell in (-0.1, math.inf, math.nan, 1e200):
        with pytest.raises(ValueError, match="length_scale"):
            NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), ell, mask)
    with pytest.raises(ValueError, match="three positive integer"):
        NonlocalHelmholtz((5, 5), (1.0, 1.0, 1.0), 0.03, mask)
    with pytest.raises(ValueError, match="positive values"):
        NonlocalHelmholtz(shape, (1.0, 0.0, 1.0), 0.03, mask)
    with pytest.raises(ValueError, match="float32 or float64"):
        NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.03, mask, dtype="float16")

    class Timers:
        def __init__(self):
            self.calls = []

        @contextmanager
        def measure(self, name):
            self.calls.append(name)
            yield

    timers = Timers()
    op = NonlocalHelmholtz(shape, (1.0, 1.0, 1.0), 0.03, mask, timers=timers)
    op.solve(cp.ones(shape, cp.float32))
    assert {
        "nonlocal_solve",
        "nonlocal_preconditioner",
        "nonlocal_operator",
        "nonlocal_reductions",
    } <= set(timers.calls)
