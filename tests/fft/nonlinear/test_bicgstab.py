"""BiCGStab(l) with the GMRES contract, and the hybrid GMRES completion."""

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear import gmres as module
from cufhom.fft.nonlinear.gmres import bicgstab, euclidean_dot, nonsymmetric_solve
from cufhom_core.tensors.fft import cp


def backends():
    out = [np]
    try:
        if cp is not None and cp.cuda.runtime.getDeviceCount():
            out.append(pytest.param(cp, marks=pytest.mark.gpu))
    except Exception:
        pass
    return out


def dot(a, b):
    """Independent FP64 reference without the Euclidean batch protocol."""
    a, b = (np.asarray(v.get() if hasattr(v, "get") else v) for v in (a, b))
    return float(
        np.sum(a.real.astype(np.float64) * b.real)
        + np.sum(a.imag.astype(np.float64) * b.imag)
    )


def operator(xp, dtype, n=40, seed=3):
    """Nonsymmetric, real-linear (conjugating for complex) diagonally dominant map."""
    rng = np.random.default_rng(seed)
    A = np.eye(n) * 4 + rng.normal(scale=0.4, size=(n, n))
    B = rng.normal(scale=0.2, size=(n, n))
    A, B = (
        xp.asarray(
            A,
            dtype=np.dtype(dtype).char.lower()
            if "complex" not in dtype
            else ("float32" if dtype == "complex64" else "float64"),
        ),
        xp.asarray(B, dtype=A.dtype),
    )
    if "complex" in dtype:
        return lambda x: (A @ x + B @ x.conj()).astype(dtype)
    return lambda x: A @ x


@pytest.mark.parametrize("xp", backends())
@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("metric", ["euclidean", "custom"])
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_bicgstab_meets_true_residual(xp, dtype, metric, degree):
    action = operator(xp, dtype)
    rng = np.random.default_rng(5)
    rhs = rng.normal(size=40) + (1j * rng.normal(size=40) if "complex" in dtype else 0)
    rhs = xp.asarray(rhs, dtype=dtype)
    original = rhs.copy()
    tolerance = 1e-5 if dtype in ("float32", "complex64") else 1e-11
    x, count = bicgstab(
        action,
        rhs,
        euclidean_dot if metric == "euclidean" else dot,
        tolerance,
        200,
        degree=degree,
    )
    residual = rhs - action(x)
    assert np.sqrt(dot(residual, residual) / dot(rhs, rhs)) <= tolerance
    assert 0 < count <= 200 and count % 2 == 0  # may converge within a cycle
    assert x.dtype == rhs.dtype and type(x) is type(rhs)
    assert bool((rhs == original).all())


@pytest.mark.parametrize("xp", backends())
def test_right_preconditioner_and_zero_rhs(xp):
    diagonal = xp.asarray(np.linspace(1.0, 50.0, 60))

    def action(x):
        return diagonal * x

    rhs = xp.asarray(np.random.default_rng(2).normal(size=60))
    x, count = bicgstab(
        action, rhs, euclidean_dot, 1e-12, 10, precondition=lambda v: v / diagonal
    )
    assert (
        count == 2
    )  # exact residual after the first BiCG step: rho = 0 is audited, not a breakdown
    np.testing.assert_allclose(
        np.asarray(x.get() if xp is cp else x),
        np.asarray((rhs / diagonal).get() if xp is cp else rhs / diagonal),
        rtol=1e-12,
    )
    zero, count = bicgstab(action, xp.zeros(60), euclidean_dot, 1e-6, 10)
    assert count == 0 and not bool(zero.any())


@pytest.mark.parametrize("xp", backends())
def test_breakdown_is_completed_by_gmres(xp):
    # A rotation: (rhat, A rhat) = 0 breaks BiCGStab at its first step.
    def action(x):
        return xp.stack([-x[1], x[0]])

    rhs = xp.asarray([1.0, 0.0])
    with pytest.raises(NonlinearConvergenceError, match="breakdown") as error:
        bicgstab(action, rhs, euclidean_dot, 1e-10, 20)
    assert error.value.iterations == 1 and error.value.best is None
    x, count, fallback = nonsymmetric_solve(action, rhs, euclidean_dot, 1e-10, 20)
    assert fallback and count == 1 + 2
    residual = rhs - action(x)
    assert dot(residual, residual) <= 1e-20


@pytest.mark.parametrize("xp", backends())
def test_budget_and_invalid_controls(xp):
    action = operator(xp, "float64")
    rhs = xp.asarray(np.random.default_rng(1).normal(size=40))
    with pytest.raises(NonlinearConvergenceError, match="after 4 iterations") as error:
        bicgstab(action, rhs, euclidean_dot, 1e-14, 5)
    assert error.value.iterations == 4 and error.value.best is not None
    # The hybrid shares one budget: GMRES gets the single remaining application.
    with pytest.raises(NonlinearConvergenceError, match="GMRES failed"):
        nonsymmetric_solve(action, rhs, euclidean_dot, 1e-14, 5)
    with pytest.raises(NonlinearConvergenceError, match="Hybrid"):
        nonsymmetric_solve(action, rhs, euclidean_dot, 1e-14, 4)
    for tolerance, maxiter in ((0.0, 5), (1.0, 5), (float("nan"), 5), (0.1, 0)):
        with pytest.raises(ValueError):
            bicgstab(action, rhs, euclidean_dot, tolerance, maxiter)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", ["float32", "complex64", "float64", "complex128"])
@pytest.mark.gpu
def test_fused_gram_matches_reference_on_gpu(k, dtype):
    if cp not in backends():
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(9)
    fields = [
        cp.asarray(
            rng.normal(size=(3, 50))
            + (1j * rng.normal(size=(3, 50)) if "complex" in dtype else 0),
            dtype=dtype,
        )
        for _ in range(k)
    ]
    got = module._gram(euclidean_dot, fields)
    expected = [[dot(a, b) for b in fields] for a in fields]
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    np.testing.assert_allclose(module._gram(dot, fields), expected, rtol=1e-12)


@pytest.mark.parametrize("xp", backends())
def test_stagnation_is_completed_by_gmres_from_the_last_iterate(xp, monkeypatch):
    action = operator(xp, "float64", n=60, seed=8)
    rhs = xp.asarray(np.random.default_rng(4).normal(size=60))
    reference, plain = bicgstab(action, rhs, euclidean_dot, 1e-10, 400)
    with pytest.raises(NonlinearConvergenceError, match="stagnated") as error:
        bicgstab(
            action, rhs, euclidean_dot, 1e-10, 400, stagnation=1, degree=1
        ) if False else module.bicgstab(
            action, rhs, euclidean_dot, 1e-10, 400, stagnation=1
        )
    assert error.value.best is not None
    calls = []
    original = module.gmres

    def spy(op, residual, *args, **kwargs):
        calls.append(np.sqrt(dot(residual, residual) / dot(rhs, rhs)))
        return original(op, residual, *args, **kwargs)

    monkeypatch.setattr(module, "gmres", spy)
    real = module.bicgstab
    monkeypatch.setattr(
        module, "bicgstab", lambda *a, **k: real(*a, **dict(k, stagnation=1))
    )
    x, count, fallback = nonsymmetric_solve(action, rhs, euclidean_dot, 1e-10, 400)
    assert (
        fallback and calls and calls[0] < 1
    )  # warm start from progress, not from zero
    residual = rhs - action(x)
    assert np.sqrt(dot(residual, residual) / dot(rhs, rhs)) <= 1e-10
    assert count <= 400 and plain > 0
