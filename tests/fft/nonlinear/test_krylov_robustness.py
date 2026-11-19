"""CPU algorithm/control checks; never construct or invoke a CUDA kernel.

GPU-only engines are exercised through NumPy doubles for their controller,
not as a new CPU backend. Kernel arithmetic/equivalence needs GPU validation.
"""

import importlib
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError

gm = importlib.import_module("cufhom.fft.nonlinear.gmres")
kr = importlib.import_module("cufhom.fft.nonlinear.krylov")
nl = importlib.import_module("cufhom.fft.nonlinear.nonlocal_damage")


@pytest.fixture
def pcg(monkeypatch):
    monkeypatch.setattr(kr, "cp", np)
    engine = kr.FusedCoordinatePCG.__new__(kr.FusedCoordinatePCG)
    events = []

    def precondition(value, reference):
        events.append("precondition")
        return value.copy()

    engine.Q = SimpleNamespace(size=4, precondition=precondition)

    def update(p, ap, alpha, x, r, reference):
        events.append("update")
        x += np.float32(alpha) * p
        r -= np.float32(alpha) * ap
        rr = gm.euclidean_dot(r, r)
        return rr, rr

    def direction(r, beta, p, reference):
        events.append("direction")
        p *= np.float32(beta)
        p += r

    engine.update, engine.direction = update, direction
    engine.events = events
    return engine


def pcg_solve(
    engine, action, rhs=None, tolerance=1e-5, maxiter=20, dot=gm.euclidean_dot
):
    if rhs is None:
        rhs = np.ones(4, dtype=np.complex64)
    return engine.solve(action, rhs, dot, tolerance, maxiter, None)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 1.0, np.inf, np.nan])
def test_pcg_rejects_invalid_tolerance_before_work(pcg, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        pcg_solve(pcg, lambda v: v, tolerance=tolerance)
    assert pcg.events == []


@pytest.mark.parametrize(
    "budget, error", [(0, ValueError), (-1, ValueError), (1.5, TypeError)]
)
def test_pcg_rejects_invalid_budget_before_work(pcg, budget, error):
    with pytest.raises(error):
        pcg_solve(pcg, lambda v: v, maxiter=budget)
    assert pcg.events == []


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan])
def test_pcg_rhs_norm_is_validated(pcg, value):
    with pytest.raises(NonlinearConvergenceError):
        pcg_solve(pcg, lambda v: v, dot=lambda a, b: value)
    assert pcg.events == []


def test_pcg_exact_zero_avoids_operator_and_preconditioner(pcg):
    def forbidden(_):
        pytest.fail("Zero RHS must not apply the operator")

    x, count = pcg_solve(pcg, forbidden, np.zeros(4, np.complex64))
    assert count == 0 and not np.any(x)
    assert pcg.events == []


@pytest.mark.parametrize("sign", [-1.0, np.nan])
def test_pcg_rejects_lost_curvature_before_update(pcg, sign):
    with pytest.raises(NonlinearConvergenceError, match="positivity"):
        pcg_solve(pcg, lambda v: sign * v)
    assert pcg.events == ["precondition"]


@pytest.mark.parametrize("pap", [1e-100, 1e100])
def test_pcg_rejects_unrepresentable_fp32_step_before_update(pcg, pap):
    values = iter([1.0, 1.0, pap])
    with pytest.raises(NonlinearConvergenceError, match="Unsafe.*step"):
        pcg_solve(pcg, lambda v: v, dot=lambda a, b: next(values))
    assert pcg.events == ["precondition"]


def test_pcg_final_audit_can_accept_pessimistic_recurrence(pcg):
    original = pcg.update
    calls = []

    def pessimistic(*args):
        original(*args)
        return 1.0, 1.0

    def action(value):
        calls.append(1)
        return value.copy()

    pcg.update = pessimistic
    x, count = pcg_solve(pcg, action, maxiter=1)
    np.testing.assert_array_equal(x, np.ones(4, np.complex64))
    assert count == 1 and len(calls) == 2  # one Krylov step and one fresh audit
    assert pcg.events == ["precondition", "update"]


def test_pcg_releases_obsolete_fields_before_fft_audit(pcg):
    original = pcg.update
    obsolete = []
    calls = []

    def update(p, ap, alpha, x, r, reference):
        obsolete.extend(weakref.ref(v) for v in (p, ap, r))
        return original(p, ap, alpha, x, r, reference)

    def action(value):
        if calls:
            assert all(ref() is None for ref in obsolete)
        calls.append(1)
        return value.copy()

    pcg.update = update
    _, count = pcg_solve(pcg, action)
    assert count == 1 and len(calls) == 2


@pytest.mark.parametrize("estimate", [0.0, 1.0])
def test_pcg_final_rejection_avoids_unused_restart_and_direction(pcg, estimate):
    original = pcg.update
    calls = []

    def estimate_override(*args):
        _, rz = original(*args)
        return estimate, rz

    def action(value):
        calls.append(1)
        return np.array([1.0, 2.0, 3.0, 4.0], np.float32) * value

    pcg.update = estimate_override
    with pytest.raises(NonlinearConvergenceError, match="true residual after 1"):
        pcg_solve(pcg, action, maxiter=1)
    assert len(calls) == 2
    assert pcg.events == ["precondition", "update"]


def test_pcg_optimistic_estimate_restarts_with_fresh_residual(pcg):
    original = pcg.update

    def optimistic(*args):
        _, rz = original(*args)
        return 0.0, rz

    pcg.update = optimistic
    diagonal = np.array([1.0, 1.2, 1.4, 1.6], np.float32)
    rhs = np.array([1.0, -0.2, 0.4, 1j], np.complex64)
    x, count = pcg_solve(pcg, lambda v: diagonal * v, rhs, maxiter=40)
    assert 1 < count <= 40
    assert "direction" not in pcg.events
    residual = rhs - diagonal * x
    assert gm.euclidean_dot(residual, residual) <= 1e-10 * gm.euclidean_dot(rhs, rhs)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_gmres_nonsymmetric_real_linear_restart_and_fresh_residual(dtype):
    rng = np.random.default_rng(139)
    exact = rng.normal(size=17)
    if "complex" in dtype:
        exact = exact + 1j * rng.normal(size=17)
    exact = exact.astype(dtype)
    diagonal = np.linspace(1.0, 1.6, 17).astype(exact.real.dtype)

    def action(value):
        result = diagonal * value + np.float32(0.13) * value.conj()
        result[:-1] += np.float32(0.07) * value[1:]
        return result

    rhs = action(exact)
    original = rhs.copy()
    tolerance = 1e-5 if exact.real.dtype == np.float32 else 1e-12
    solution, count = gm.gmres(
        action,
        rhs,
        gm.euclidean_dot,
        tolerance,
        90,
        precondition=lambda v: v / diagonal,
        restart=2,
    )
    residual = rhs - action(solution)
    assert 2 < count <= 90
    assert (
        np.sqrt(gm.euclidean_dot(residual, residual) / gm.euclidean_dot(rhs, rhs))
        <= tolerance
    )
    np.testing.assert_array_equal(rhs, original)
    np.testing.assert_allclose(solution, exact, rtol=3 * tolerance, atol=3 * tolerance)


def test_gmres_weighted_dot_keeps_its_metric():
    rhs = np.array([1.0, -0.4, 0.6, 2.0])
    matrix = np.array(
        [
            [2.0, 0.3, 0.0, 0.0],
            [0.0, 1.5, 0.2, 0.0],
            [0.1, 0.0, 1.8, 0.3],
            [0.0, 0.1, 0.0, 2.1],
        ]
    )
    weights = np.array([1.0, 4.0, 2.0, 3.0])
    calls = []

    def weighted(a, b):
        calls.append(1)
        return float(np.sum(weights * a * b))

    solution, count = gm.gmres(lambda v: matrix @ v, rhs, weighted, 1e-12, 4, restart=4)
    residual = rhs - matrix @ solution
    assert weighted(residual, residual) <= 1e-24 * weighted(rhs, rhs)
    assert count == 4 and len(calls) > count


def test_gmres_underflow_is_not_an_exact_zero_rhs():
    rhs = np.full(4, 1e-200)
    with pytest.raises(NonlinearConvergenceError, match="underflowed"):
        gm.gmres(lambda v: v, rhs, gm.euclidean_dot, 1e-5, 10)


@pytest.mark.parametrize(
    "dtype,amplitude",
    [
        ("float32", 2e38),
        ("complex64", 2e38),
        ("float32", 1e-39),
        ("complex64", 1e-39),
        ("float32", 1e-44),
        ("complex64", 1e-44),
        ("float64", 1e140),
        ("complex128", 1e140),
        ("float64", 1e-140),
        ("complex128", 1e-140),
    ],
)
@pytest.mark.parametrize("kind", ["identity_alias", "diagonal_strided", "real_linear"])
def test_gmres_extreme_rhs_fresh_accuracy_alias_and_strides(dtype, amplitude, kind):
    # Strided caller storage is immutable. Both components matter, including
    # for conjugation, where A is linear over the reals but not the complexes.
    storage = np.zeros(32, dtype=dtype)
    rhs = storage[::2]
    rhs.real = amplitude
    if rhs.dtype.kind == "c":
        rhs.imag = rhs.real * np.float32(0.5)
    original = storage.copy()
    backing = np.empty(32, dtype=dtype)
    diagonal = np.linspace(0.7, 1.0, 16).astype(rhs.real.dtype)
    action_inputs = []

    def action(value):
        assert value.dtype == rhs.dtype and np.isfinite(value).all()
        action_inputs.append(1)
        if kind == "identity_alias":
            return value
        if kind == "diagonal_strided":
            backing[::2] = diagonal * value
            return backing[::2]
        result = diagonal * value + np.float32(0.03) * value.conj()
        result[:-1] += np.float32(0.02) * value[1:]
        return result

    tolerance = 1e-5 if rhs.real.dtype == np.float32 else 1e-12
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        solution, count = gm.gmres(
            action, rhs, gm.euclidean_dot, tolerance, 120, restart=4
        )
    residual = rhs - action(solution)
    relative = np.sqrt(
        gm.euclidean_dot(residual, residual) / gm.euclidean_dot(rhs, rhs)
    )
    assert relative <= tolerance
    assert solution.dtype == rhs.dtype and np.isfinite(solution).all()
    if kind == "identity_alias":
        assert (
            count == 1 and len(action_inputs) == 3
        )  # Arnoldi, audit, independent audit
    else:
        assert 1 < count <= 120
    np.testing.assert_array_equal(storage, original)


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("amplitude", [2e38, 1e-39])
def test_gmres_extreme_rhs_right_preconditioner(dtype, amplitude):
    rhs = np.full(16, amplitude, dtype=dtype)
    if rhs.dtype.kind == "c":
        rhs.imag = rhs.real * np.float32(0.5)
    diagonal = np.linspace(1.0, 1.6, 16).astype(rhs.real.dtype)

    def action(value):
        return diagonal * value

    x, count = gm.gmres(
        action,
        rhs,
        gm.euclidean_dot,
        1e-5,
        4,
        precondition=lambda value: value / diagonal,
    )
    assert count == 1
    residual = rhs - action(x)
    assert (
        np.sqrt(gm.euclidean_dot(residual, residual) / gm.euclidean_dot(rhs, rhs))
        <= 1e-5
    )


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("scale", [2e38, 1e-40])
def test_normalization_extremes_use_fp64_denominator_without_fp64_field(
    monkeypatch, dtype, scale
):
    value = np.full(16, scale, dtype=dtype)
    if value.dtype.kind == "c":
        value.imag = value.real * np.float32(0.5)
    norm = np.sqrt(gm.euclidean_dot(value, value))
    expected = (
        value.astype("complex128" if value.dtype.kind == "c" else "float64") / norm
    )
    divide = np.divide
    calls = []

    def widened_divide(source, denominator, *, out, dtype):
        assert source.dtype == out.dtype == np.float32
        assert isinstance(denominator, np.float64) and denominator == norm
        assert dtype is np.float64
        calls.append(1)
        return divide(source, denominator, out=out, dtype=dtype)

    monkeypatch.setattr(np, "divide", widened_divide)
    # In-place normalization is safe; complex parts share the same real norm.
    gm._normalize(value, norm, value)
    assert len(calls) == 1
    np.testing.assert_allclose(value, expected, rtol=8e-8, atol=0.0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
def test_normalization_ordinary_path_is_one_native_multiply(monkeypatch, dtype):
    value = np.array([1.0, -0.3, 0.7], dtype=dtype)
    if value.dtype.kind == "c":
        value.imag = [0.1, -0.6, 0.2]
    norm = np.sqrt(gm.euclidean_dot(value, value))
    multiply = np.multiply
    calls = []

    def native_multiply(source, inverse, *, out):
        assert source.dtype == out.dtype == value.dtype
        assert np.asarray(inverse).dtype == value.real.dtype
        calls.append(1)
        return multiply(source, inverse, out=out)

    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary normalization must not use widened division")

    monkeypatch.setattr(np, "multiply", native_multiply)
    monkeypatch.setattr(np, "divide", forbidden)
    out = np.empty_like(value)
    gm._normalize(value, norm, out)
    assert len(calls) == 1
    expected = (
        value.astype("complex128" if value.dtype.kind == "c" else "float64") / norm
    )
    np.testing.assert_allclose(
        out, expected, rtol=2 * np.finfo(value.real.dtype).eps, atol=0.0
    )


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
@pytest.mark.parametrize("coefficient", [8e38, 1.23456789e-39])
@pytest.mark.parametrize("subtract", [False, True])
def test_basis_update_retains_extreme_coefficients_and_field_storage(
    monkeypatch, dtype, coefficient, subtract
):
    basis = np.full((1, 16), 0.25, dtype=dtype)
    if basis.dtype.kind == "c":
        basis.imag = -0.125
    target = np.zeros(16, dtype=dtype)
    multiply = np.multiply
    calls = []

    def product_into_field(source, value, *, dtype, out):
        assert dtype is np.float64
        assert source.dtype == out.dtype == np.float32
        assert value == coefficient
        calls.append(1)
        return multiply(source, value, dtype=dtype, out=out)

    monkeypatch.setattr(np, "multiply", product_into_field)
    gm._basis_update(basis, np.array([coefficient]), target, subtract)
    expected = (
        coefficient
        * basis[0].astype("complex128" if basis.dtype.kind == "c" else "float64")
    ).astype(target.dtype)
    if subtract:
        expected = -expected
    np.testing.assert_array_equal(target, expected)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_extreme_identity_still_rejects_stale_audit(dtype):
    rhs = np.full(16, 2e38, dtype=dtype)
    calls = []

    def stale(value):
        calls.append(1)
        return value if len(calls) == 1 else np.float32(0.5) * value

    with pytest.raises(
        NonlinearConvergenceError, match="breakdown failed true residual"
    ):
        gm.gmres(stale, rhs, gm.euclidean_dot, 1e-5, 4)
    assert len(calls) == 2


def test_fresh_residual_underflow_cannot_accept_a_nonzero_field():
    rhs = np.full(4, 1e-150)
    original = rhs.copy()
    calls = []

    def stale(value):
        calls.append(1)
        return value if len(calls) == 1 else value * (1.0 - 1e-15)

    # The residual is nonzero but its square underflows FP64. Reject instead
    # of accepting zero against an even smaller requested target.
    with pytest.raises(NonlinearConvergenceError, match="underflowed"):
        gm.gmres(stale, rhs, gm.euclidean_dot, 1e-16, 4)
    assert len(calls) == 2
    np.testing.assert_array_equal(rhs, original)


def test_gmres_exact_zero_and_operator_alias():
    def forbidden(_):
        pytest.fail("Zero RHS must not invoke A or P")

    zero = np.zeros(4)
    x, count = gm.gmres(
        forbidden, zero, gm.euclidean_dot, 1e-10, 4, precondition=forbidden
    )
    assert count == 0 and not np.any(x)
    rhs = np.ones(4)
    x, count = gm.gmres(lambda v: v, rhs, gm.euclidean_dot, 1e-10, 4)
    assert count == 1
    np.testing.assert_allclose(x, rhs, rtol=1e-14)
    np.testing.assert_array_equal(rhs, np.ones(4))


def test_gmres_breakdown_must_pass_fresh_audit():
    calls = []

    def stale(value):
        calls.append(1)
        return value if len(calls) == 1 else 2 * value

    with pytest.raises(
        NonlinearConvergenceError, match="breakdown failed true residual"
    ):
        gm.gmres(stale, np.ones(4), gm.euclidean_dot, 1e-10, 4)
    assert len(calls) == 2


@pytest.mark.parametrize("restart, budget", [(1, 3), (2, 5), (12, 3)])
def test_gmres_skips_terminal_basis_write_and_respects_global_budget(
    monkeypatch, restart, budget
):
    normalizations = []
    actions = []
    normalize = gm._normalize

    def counted_normalize(*args, **kwargs):
        normalizations.append(1)
        return normalize(*args, **kwargs)

    def action(value):
        actions.append(1)
        return np.linspace(1.0, 4.0, 15) * value

    monkeypatch.setattr(gm, "_normalize", counted_normalize)
    with pytest.raises(NonlinearConvergenceError, match=f"after {budget} iterations"):
        gm.gmres(action, np.ones(15), gm.euclidean_dot, 1e-14, budget, restart=restart)
    cycles = (budget + restart - 1) // restart
    assert len(actions) == budget + cycles
    # Each cycle normalizes its initial residual, then only consumed vectors.
    # The previous implementation used budget+cycles full-field writes.
    assert len(normalizations) == budget


def test_gmres_strided_operator_output_has_one_owned_copy(monkeypatch):
    copied = []
    array = np.array
    backing = np.empty(8)

    def action(value):
        backing[::2] = value
        return backing[::2]

    def counted_array(value, *args, **kwargs):
        result = array(value, *args, **kwargs)
        if (
            value is not None
            and isinstance(value, np.ndarray)
            and np.shares_memory(value, backing)
        ):
            copied.append(result)
        return result

    monkeypatch.setattr(np, "array", counted_array)
    x, count = gm.gmres(action, np.ones(4), gm.euclidean_dot, 1e-12, 4)
    assert count == 1 and len(copied) == 1
    assert copied[0].flags.c_contiguous and not np.shares_memory(copied[0], backing)
    np.testing.assert_allclose(x, np.ones(4))


@pytest.mark.parametrize(
    "route,fails",
    [
        ("host", False),
        ("host", True),
        ("graph", False),
        ("graph", True),
        ("gmres", False),
        ("gmres", True),
        ("bind", True),
    ],
)
def test_tangent_pipeline_releases_on_every_exit(monkeypatch, route, fails):
    monkeypatch.setattr(kr, "cp", np)
    monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", "1")
    events = []
    pipeline = SimpleNamespace(
        config={"graph_iterations": int(route == "graph")},
        pending=None,
        complex_dtype=np.dtype(np.float64),
    )

    def bind(*args):
        events.append("bind")
        if route == "bind":
            raise ValueError("bad binding")

    def solve(*args, **kwargs):
        events.append("solve")
        if fails:
            pipeline.pending = (np.ones(4),) * 4
            raise NonlinearConvergenceError("inner failure")
        return np.ones(4), 2

    def release():
        assert pipeline.pending is None
        events.append("release")

    def production(value):
        assert events[-1] == "release"
        return value

    pipeline.bind, pipeline.release = bind, release
    pipeline.graph_cg, pipeline.matvec = solve, lambda v: v
    pipeline.to_fused = lambda value: value.copy()
    pipeline.to_production, pipeline.settle = production, lambda: None
    engine = kr.TangentPipeline.__new__(kr.TangentPipeline)
    engine.pipeline = pipeline
    engine.Q = SimpleNamespace(whiten=lambda v: v, strain=lambda v: v)
    response = SimpleNamespace(
        a=None,
        b=None,
        direction=None,
        bulk=None,
        coupling_to_direction=None,
        coupling_to_pressure=None,
    )
    monkeypatch.setattr(kr, "fused_cg", solve)
    monkeypatch.setattr(kr, "FusedStep", lambda p: None)
    monkeypatch.setattr(gm, "gmres", solve)
    method = "gmres" if route == "gmres" else "cg"
    if fails:
        with pytest.raises((ValueError, NonlinearConvergenceError)):
            engine.solve(
                response, np.ones(4), 1e-5, 10, gm.euclidean_dot, method=method
            )
    else:
        _, count = engine.solve(
            response, np.ones(4), 1e-5, 10, gm.euclidean_dot, method=method
        )
        assert count == 2
    assert events.count("release") == 1
    assert events[-1] == "release" and pipeline.pending is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(method="typo"),
        dict(method="gmres", restart=0),
        dict(tolerance=1.0),
        dict(maxiter=0),
    ],
)
def test_tangent_rejects_bad_controls_before_binding(kwargs):
    engine = kr.TangentPipeline.__new__(kr.TangentPipeline)
    controls = dict(tolerance=1e-5, maxiter=10, method="cg") | kwargs
    # No pipeline exists: validation must happen before touching GPU state.
    with pytest.raises(ValueError):
        engine.solve(None, None, dot=gm.euclidean_dot, **controls)


@pytest.mark.parametrize("kwargs", [dict(rtol=0.0), dict(rtol=np.nan), dict(maxiter=0)])
def test_nonlocal_rejects_bad_controls_without_cuda(kwargs):
    engine = nl.NonlocalHelmholtz.__new__(nl.NonlocalHelmholtz)
    with pytest.raises(ValueError):
        engine.solve(None, **kwargs)


@pytest.fixture
def nonlocal_controller(monkeypatch):
    """A tiny SPD constant-preserving action, exercising only PCG control."""
    engine = nl.NonlocalHelmholtz.__new__(nl.NonlocalHelmholtz)
    engine.xp, engine.timers = np, None
    engine.dtype = np.dtype("float64")
    engine._constant = engine._identity = False
    engine._mean_coefficient = 1.0
    engine._mask = np.ones(4, dtype=bool)
    engine._r, engine._z, engine._p, engine._ap = [np.empty(4) for _ in range(4)]
    engine._field = lambda value, name: value
    engine._dot = gm.euclidean_dot
    difference = np.eye(4) - np.roll(np.eye(4), 1, axis=1)
    matrix = np.eye(4) + difference.T @ np.diag([0.2, 1.0, 0.5, 2.0]) @ difference

    def apply(value, out):
        engine._operator_calls += 1
        out[:] = matrix @ value

    def precondition(value, out):
        engine._preconditioner_calls += 1
        out[:] = value

    def update(p, ap, alpha, x, r):
        x += alpha * p
        r -= alpha * ap

    def direction(z, beta, p):
        p *= beta
        p += z

    engine._apply_into, engine._precondition_into = apply, precondition
    kernels = SimpleNamespace(
        source_flags=lambda *args: np.int32(0), update=update, direction=direction
    )
    monkeypatch.setattr(nl, "_kernels", lambda: kernels)
    return engine, matrix


def test_nonlocal_controller_fresh_acceptance_and_mean(nonlocal_controller):
    engine, matrix = nonlocal_controller
    source = np.array([1.0, 0.3, -0.4, 2.0])
    original = source.copy()
    x, diagnostics = engine._solve(source, 1e-12, 10, None)
    assert diagnostics["converged"]
    residual = source - matrix @ x
    assert np.linalg.norm(residual) <= 1e-12 * np.linalg.norm(source)
    assert abs(x.mean() - source.mean()) <= 1e-15
    assert diagnostics["relative_residual"] <= 1e-12
    np.testing.assert_array_equal(source, original)


def test_nonlocal_final_audit_and_no_unused_preconditioner(nonlocal_controller):
    engine, matrix = nonlocal_controller
    source = np.array([1.0, 0.3, -0.4, 2.0])
    with pytest.raises(nl.NonlocalConvergenceError) as error:
        engine._solve(source, 1e-12, 1, None)
    diagnostics = error.value.diagnostics
    assert not diagnostics["converged"] and diagnostics["iterations"] == 1
    assert diagnostics["operator_calls"] == 3  # initial, step, final audit
    assert diagnostics["preconditioner_calls"] == 1


def test_nonlocal_optimistic_recurrence_never_accepts_stale_field(
    nonlocal_controller, monkeypatch
):
    engine, matrix = nonlocal_controller
    original = engine._dot
    calls = []

    def optimistic(a, b):
        calls.append(1)
        # source, initial true residual, rz, pAp, then recursive rr
        return 0.0 if len(calls) == 5 else original(a, b)

    engine._dot = optimistic
    source = np.array([1.0, 0.3, -0.4, 2.0])
    with pytest.raises(nl.NonlocalConvergenceError) as error:
        engine._solve(source, 1e-12, 1, None)
    assert error.value.diagnostics["relative_residual"] > 1e-12
