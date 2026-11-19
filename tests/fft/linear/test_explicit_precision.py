"""Explicit precision selection, fail-closed loads and independent references."""

import importlib.util
import inspect

import cufhom.fft as fft
import numpy as np
import pytest
from cufhom.fft import homogenize
from cufhom.fft.linear import coordinates, framework, hex8_general
from cufhom.fft.linear.common import LinearSolveError, WorkingPrecisionError

from tests.fft.linear.test_discrete_reference_verification import (
    LENGTHS,
    audit_case,
    explicit_problem,
    gpu,
)


def test_recovery_api_and_failure_capture_are_removed():
    assert not hasattr(fft, "RecoveryOptions")
    assert "RecoveryOptions" not in fft.__all__
    assert "recovery" not in inspect.signature(homogenize).parameters
    assert importlib.util.find_spec("cufhom.fft.linear.recovery") is None
    assert (
        "capture_recovery"
        not in inspect.signature(coordinates.homogenize_gani_mixed).parameters
    )
    assert (
        "capture_recovery"
        not in inspect.signature(hex8_general.homogenize_hex8).parameters
    )
    with pytest.raises(TypeError, match="recovery"):
        homogenize(np.zeros((2, 2, 2), int), np.eye(6)[None], recovery="adaptive")


@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_failed_load_does_not_escalate_and_user_can_run_fp64(
    kind, backend, monkeypatch
):
    if backend == "cupy":
        gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case((3, 3, 3))
    called = []
    for name in ("_gani", "_gani_mixed", "_hex8"):
        original = getattr(framework, name)

        def record(*args, _original=original, _name=name, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(framework, name, record)
    with pytest.raises(RuntimeError) as caught:
        homogenize(
            labels,
            table,
            discretization=kind,
            backend=backend,
            reference=c0,
            lengths=LENGTHS,
            maxiter=1,
            rtol=1e-10,
        )
    assert len(called) == 1
    assert not hasattr(caught.value, "solution") and not hasattr(
        caught.value, "completed_infos"
    )
    if isinstance(caught.value, LinearSolveError):
        assert caught.value.status == "maxiter"
    called.clear()
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend=backend,
        reference=c0,
        lengths=LENGTHS,
        working="float64",
        rtol=1e-9,
        verify_tensor=True,
    )
    assert len(called) == 1
    np.testing.assert_allclose(
        result["C"], explicit_problem(kind, labels, table)[2], atol=3e-11, rtol=3e-11
    )
    assert result["working"] == "float64" and result["backend"] == backend


@pytest.mark.parametrize("energy", [None, 1e-6])
def test_missing_mixed_capacity_never_selects_fp64_implicitly(energy, monkeypatch):
    monkeypatch.setattr(
        framework,
        "_gani_residency",
        lambda material, configuration="mixed", energy=False, library=False: None,
    )

    def forbidden(*a, **kw):
        pytest.fail("Capacity failure must not change the selected working precision")

    monkeypatch.setattr(framework, "_gani", forbidden)
    with pytest.raises(MemoryError, match="residency budget"):
        homogenize(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            working="float32",
            energy_rtol=energy,
        )


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_unsafe_fp32_is_rejected_and_explicit_fp64_keeps_the_material(kind):
    q = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0]
    table = (q @ np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1e-10]) @ q.T)[None]
    labels = np.zeros((3, 3, 3), int)
    with pytest.raises(WorkingPrecisionError) as caught:
        homogenize(
            labels, table, discretization=kind, reference=np.eye(6), working="float32"
        )
    assert caught.value.status == "invalid_working_operator"
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        reference=np.eye(6),
        working="float64",
    )
    np.testing.assert_allclose(result["C"], table[0], atol=3e-14, rtol=3e-14)


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_allocation_failure_is_propagated_once(kind, monkeypatch):
    calls = []

    def failed(*a, **kw):
        calls.append(1)
        raise MemoryError("induced allocation failure")

    monkeypatch.setattr(framework, "_gani" if kind == "gani" else "_hex8", failed)
    with pytest.raises(MemoryError, match="induced allocation"):
        homogenize(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            discretization=kind,
            backend="numpy",
        )
    assert len(calls) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize("verify", [False, True])
def test_fields_are_retained_only_for_requested_tensor_verification(
    kind, verify, monkeypatch
):
    gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case()
    function = "_gani_mixed" if kind == "gani" else "_hex8"
    original = getattr(framework, function)
    observed = []

    def record(*args, **kwargs):
        observed.append(args[-1])  # fields, not a failure checkpoint
        return original(*args, **kwargs)

    monkeypatch.setattr(framework, function, record)
    result = homogenize(
        labels,
        table,
        discretization=kind,
        reference=c0,
        lengths=LENGTHS,
        rtol=1e-6,
        verify_tensor=verify,
    )
    assert len(observed) == 1
    assert (observed[0] is not None) == verify
    if verify:
        assert len(observed[0]) == 6 and result["tensor_verification"]["accepted"]
    assert (
        not {"recovery", "recovery_verification", "requested_backend"} & result.keys()
    )
    assert all("recoveries" not in audit for audit in result["audits"])


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_energy_budget_failure_is_not_a_success_or_an_automatic_retry(
    kind, monkeypatch
):
    gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case()
    with pytest.raises(LinearSolveError) as caught:
        homogenize(
            labels,
            table,
            discretization=kind,
            reference=c0,
            lengths=LENGTHS,
            working="float32",
            maxiter=1,
            energy_rtol=1e-10,
        )
    assert caught.value.status == "maxiter" and not hasattr(caught.value, "solution")


@pytest.mark.gpu
@pytest.mark.parametrize("storage", ["device", "host"])
def test_later_cuda_failure_stops_before_remaining_loads(storage, monkeypatch):
    gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case()
    original = hex8_general._solve_cuda
    calls = []

    def fail_third(op, macro, rtol, atol, maxiter, energy=None, residency="device"):
        load = int(np.argmax(macro))
        calls.append(load)
        return original(
            op, macro, rtol, atol, 1 if load == 2 else maxiter, energy, residency
        )

    monkeypatch.setattr(hex8_general, "_solve_cuda", fail_third)
    with pytest.raises(LinearSolveError) as caught:
        homogenize(
            labels,
            table,
            discretization="hex8",
            reference=c0,
            lengths=LENGTHS,
            storage=storage,
            rtol=1e-8,
        )
    assert calls == [0, 1, 2]
    assert caught.value.status == "maxiter" and caught.value.diagnostics["load_id"] == 2
    assert not hasattr(caught.value, "solution") and not hasattr(
        caught.value, "completed_infos"
    )


@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize(
    "contrast,bulk,shape,reference_scale",
    [
        (1.0, 3.0, (3, 3, 3), 1.0),
        (1e2, 1e4, (3, 4, 3), 0.01),
        (1e5, 3.0, (3, 3, 4), 100.0),
    ],
)
def test_adversarial_materials_with_explicit_ieee_fp64(
    kind, contrast, bulk, shape, reference_scale
):
    xyz = np.indices(shape)
    labels = ((xyz[0] + 2 * xyz[1] + xyz[2]) % 3 == 0).astype(int)
    base = 2 * np.eye(6)
    base[:3, :3] += bulk - 2 / 3
    a = np.random.default_rng(1903).normal(size=(6, 6))
    base += 0.03 * a.T @ a
    table = np.array([base, contrast * base])
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        reference=reference_scale * np.eye(6),
        working="float64",
        lengths=(0.125, 3.0, 0.75),
        rtol=1e-9,
        maxiter=1500,
        verify_tensor=True,
    )
    exact = explicit_problem(kind, labels, table, lengths=(0.125, 3.0, 0.75))[2]
    np.testing.assert_allclose(result["C"], exact, atol=1e-8, rtol=2e-8)
    assert result["tensor_verification"]["native_residual_passed"]
    assert not result["tensor_verification"]["certified"]


def test_df64_is_not_an_iterative_working_dtype():
    with pytest.raises(TypeError):
        homogenize(
            np.zeros((2, 2, 2), int), np.eye(6)[None], backend="numpy", working="df64"
        )


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_explicit_fp64_against_independent_80_digit_direct_solution(kind):
    # Same independent high-precision reference retained from P1. The DFT/Q1
    # assembly remains FP64; only direct elimination/Schur use 80 digits.
    from decimal import Decimal, localcontext

    labels, table, c0 = audit_case((3, 3, 3))
    table[1] *= 100
    k, f, _, _ = explicit_problem(kind, labels, table)
    if kind == "hex8":
        keep = np.array(
            [i for i in range(len(k)) if i not in (0, labels.size, 2 * labels.size)]
        )
        k, f = k[np.ix_(keep, keep)], f[keep]
    d = table[labels].mean(axis=(0, 1, 2))
    with localcontext() as context:
        context.prec = 80
        n = len(k)
        a = [
            [Decimal.from_float(float(v)) for v in row]
            for row in np.column_stack((k, f))
        ]
        original_f = [[Decimal.from_float(float(v)) for v in row] for row in f]
        for j in range(n):
            pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
            a[j], a[pivot] = a[pivot], a[j]
            for i in range(j + 1, n):
                factor = a[i][j] / a[j][j]
                for column in range(j + 1, n + 6):
                    a[i][column] -= factor * a[j][column]
        u = [[Decimal(0) for _ in range(6)] for _ in range(n)]
        for i in range(n - 1, -1, -1):
            for column in range(6):
                u[i][column] = (
                    a[i][n + column]
                    - sum(a[i][j] * u[j][column] for j in range(i + 1, n))
                ) / a[i][i]
        exact = np.array(
            [
                [
                    float(
                        Decimal.from_float(float(d[i, j]))
                        - sum(original_f[t][i] * u[t][j] for t in range(n))
                    )
                    for j in range(6)
                ]
                for i in range(6)
            ]
        )
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        reference=c0,
        lengths=LENGTHS,
        working="float64",
        rtol=1e-10,
    )
    np.testing.assert_allclose(result["C"], exact, atol=3e-10, rtol=3e-12)
