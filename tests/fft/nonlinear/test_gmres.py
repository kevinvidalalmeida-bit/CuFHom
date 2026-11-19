"""Focused GPU validation for real-inner, right-preconditioned restarted GMRES."""

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear.gmres import euclidean_dot, gmres, real_inner_products
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


def dot(a, b):
    """Independent FP64 reference; deliberately has no batched callback."""
    real = cp.sum(a.real.astype(cp.float64) * b.real, dtype=cp.float64)
    if a.dtype.kind == "c":
        real += cp.sum(a.imag.astype(cp.float64) * b.imag, dtype=cp.float64)
    return float(real)


def assert_residual(action, rhs, solution, tolerance):
    residual = rhs - action(solution)
    assert np.sqrt(dot(residual, residual) / dot(rhs, rhs)) <= tolerance


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_batched_real_inner_matches_independent_dot(dtype):
    rng = np.random.default_rng(43)
    host = rng.normal(size=(5, 3, 7))
    value = rng.normal(size=(3, 7))
    if "complex" in dtype:
        host = host + 1j * rng.normal(size=host.shape)
        value = value + 1j * rng.normal(size=value.shape)
    basis, value = cp.asarray(host, dtype=dtype), cp.asarray(value, dtype=dtype)
    expected = np.array([dot(v, value) for v in basis])
    np.testing.assert_allclose(
        cp.asnumpy(real_inner_products(basis, value)), expected, rtol=5e-14, atol=1e-13
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_exact_right_preconditioner_diagonal(dtype):
    rhs = cp.asarray(np.random.default_rng(7).normal(size=(3, 4, 5)), dtype=dtype)
    diagonal = cp.asarray(
        np.linspace(1.0, 30.0, rhs.size).reshape(rhs.shape), dtype=dtype
    )
    tolerance = 1e-5 if dtype == "float32" else 1e-12
    original = rhs.copy()

    def action(x):
        return diagonal * x

    solution, count = gmres(
        action, rhs, euclidean_dot, tolerance, 20, precondition=lambda v: v / diagonal
    )
    assert count == 1
    assert_residual(action, rhs, solution, tolerance)
    cp.testing.assert_array_equal(rhs, original)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_nonsymmetric_complex_real_linear_operator(dtype):
    # A acts on real and imaginary parts with different coefficients plus an
    # upper neighbor coupling. This is real-linear, not complex-linear.
    rng = np.random.default_rng(11)
    exact = cp.asarray(rng.normal(size=31) + 1j * rng.normal(size=31), dtype=dtype)
    real_dtype = cp.float32 if dtype == "complex64" else cp.float64
    diagonal = cp.asarray(np.linspace(1.0, 1.8, exact.size), dtype=real_dtype)

    def action(x):
        result = diagonal * x + real_dtype(0.12) * x.conj()
        result[:-1] += real_dtype(0.07) * x[1:]
        return result

    rhs = action(exact)
    tolerance = 1e-5 if dtype == "complex64" else 1e-12
    solution, count = gmres(
        action,
        rhs,
        euclidean_dot,
        tolerance,
        100,
        precondition=lambda v: v / diagonal,
        restart=6,
    )
    assert 1 < count <= 100
    assert_residual(action, rhs, solution, tolerance)
    cp.testing.assert_allclose(solution, exact, rtol=3 * tolerance, atol=3 * tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_restart_budget_and_true_residual(dtype):
    rhs = cp.asarray(np.random.default_rng(19).normal(size=40), dtype=dtype)
    diagonal = cp.asarray(np.linspace(1.0, 2.0, rhs.size), dtype=dtype)
    tolerance = 1e-5 if dtype == "float32" else 1e-12

    def action(v):
        return diagonal * v

    solution, count = gmres(action, rhs, euclidean_dot, tolerance, 120, restart=2)
    assert count > 2
    assert_residual(action, rhs, solution, tolerance)
    with pytest.raises(NonlinearConvergenceError, match="true residual"):
        gmres(action, rhs, euclidean_dot, tolerance, 1, restart=12)


def test_supplied_weighted_dot_is_not_bypassed():
    rhs = cp.asarray([1.0, -0.7, 0.3, 1.1], dtype=cp.float64)
    weight = cp.asarray([1.0, 3.0, 2.0, 5.0], dtype=cp.float64)
    matrix = cp.asarray(
        [
            [2.0, 0.2, 0.0, 0.0],
            [0.0, 1.7, 0.3, 0.0],
            [0.1, 0.0, 1.4, 0.2],
            [0.0, 0.1, 0.0, 2.1],
        ]
    )
    calls = []

    def weighted(a, b):
        calls.append(1)
        return float(cp.sum(weight * a * b, dtype=cp.float64))

    # Dense matvec is confined to this tiny test; GMRES uses no dense GPU solve.
    def action(x):
        return matrix @ x

    solution, count = gmres(action, rhs, weighted, 1e-12, 10, restart=4)
    residual = rhs - action(solution)
    assert np.sqrt(weighted(residual, residual) / weighted(rhs, rhs)) <= 1e-12
    assert count <= 4 and len(calls) > count


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_zero_rhs_happy_and_unhappy_breakdown(dtype):
    zero = cp.zeros(9, dtype=dtype)

    def forbidden(_):
        raise AssertionError(
            "Exact zero rhs must not invoke operator or preconditioner"
        )

    solution, count = gmres(
        forbidden, zero, euclidean_dot, 1e-5, 10, precondition=forbidden
    )
    assert count == 0
    cp.testing.assert_array_equal(solution, zero)
    rhs = cp.ones(9, dtype=dtype)
    solution, count = gmres(lambda x: x, rhs, euclidean_dot, 1e-5, 10)
    assert count == 1
    assert_residual(lambda x: x, rhs, solution, 1e-5)
    with pytest.raises(NonlinearConvergenceError, match="breakdown"):
        gmres(lambda x: cp.zeros_like(x), rhs, euclidean_dot, 1e-5, 10)


def test_happy_breakdown_must_pass_true_residual():
    rhs = cp.asarray([1.0, 0.0, 0.0], dtype=cp.float64)
    calls = []

    def inconsistent_action(v):
        # Simulates an incorrect/stale operator at the audit: the Arnoldi
        # equation appears exact, but rhs-Ax remains nonzero. It must fail.
        calls.append(1)
        return v if len(calls) == 1 else 2 * v

    with pytest.raises(
        NonlinearConvergenceError, match="breakdown failed true residual"
    ):
        gmres(inconsistent_action, rhs, euclidean_dot, 1e-12, 10)
    assert len(calls) == 2


def test_optimistic_estimate_restarts_from_true_residual():
    import importlib

    module = importlib.import_module("cufhom.fft.nonlinear.gmres")
    original = module._back_substitute
    rhs = cp.asarray(np.random.default_rng(99).normal(size=20), dtype=cp.float64)
    diagonal = cp.asarray(np.linspace(1.0, 1.6, rhs.size))

    def action(v):
        return diagonal * v

    calls = []

    def wrong_first_update(upper, right):
        y = original(upper, right)
        calls.append(1)
        return y * 0.8 if len(calls) == 1 else y

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "_back_substitute", wrong_first_update)
        solution, count = gmres(action, rhs, euclidean_dot, 1e-11, 120, restart=8)
    assert len(calls) > 1 and count > 8
    assert_residual(action, rhs, solution, 1e-11)


@pytest.mark.parametrize("where", ["rhs", "operator"])
def test_nonfinite_norm_fails(where):
    rhs = cp.ones(5, dtype=cp.float64)
    if where == "rhs":
        rhs[0] = cp.nan
    action = (lambda v: v * cp.nan) if where == "operator" else (lambda v: v)
    with pytest.raises(NonlinearConvergenceError, match="Nonfinite"):
        gmres(action, rhs, euclidean_dot, 1e-5, 10)
