"""Regressions against full DFT and separately differentiated HEX8 elements."""

import itertools
import json

import numpy as np
import pytest
from cufhom.fft.linear.common import (
    FreshProgress,
    IndexedMaterial,
    LinearSolveError,
    working_material_check,
)
from cufhom.fft.linear.framework import homogenize
from cufhom.fft.linear.hex8_general import Hex8Operator, _check_working_elements
from cufhom.fft.linear.verification import (
    TensorVerificationError,
    verify_effective_tensor,
)

LENGTHS = (0.8, 1.3, 1.1)


def audit_case(shape=(4, 4, 4)):
    rng = np.random.default_rng(82026)
    a = rng.normal(size=(2, 6, 6))
    table = a.transpose(0, 2, 1) @ a + 2 * np.eye(6)
    table[1] *= 4
    labels = (rng.random(shape) < 0.4).astype(np.int32)
    c0 = 4 * np.eye(6)
    c0[:3, :3] += 5 - 4 / 3
    return labels, table, c0


def explicit_problem(kind, labels, table, lengths=LENGTHS):
    """No production Q, B, element table, FFT symbol or solver is reused."""
    shape, n = labels.shape, labels.size
    d = table[labels].mean(axis=(0, 1, 2))
    if kind == "gani":
        grid = np.array(list(np.ndindex(shape)))
        ks = (
            np.array(
                np.meshgrid(*[np.fft.fftfreq(s) * s for s in shape], indexing="ij")
            )
            .reshape(3, -1)
            .T
        )
        transform = np.exp(-2j * np.pi * ((ks / shape) @ grid.T))
        symbols = np.zeros((6, 6, n))
        for t, k in enumerate(ks):
            # Even-grid Nyquist indices: zero wavenumber on that axis (derivative of
            # the real trigonometric interpolant); a vanishing wavevector is excluded.
            q = (
                np.array(
                    [
                        0.0 if s % 2 == 0 and abs(k[i]) == s // 2 else k[i]
                        for i, s in enumerate(shape)
                    ]
                )
                / lengths
            )
            if np.all(q == 0):
                continue
            x, y, z = q / np.linalg.norm(q)
            s = np.array(
                [[x, 0, 0], [0, y, 0], [0, 0, z], [0, z, y], [z, 0, x], [y, x, 0]]
            )
            s[3:] /= np.sqrt(2)
            symbols[:, :, t] = s @ np.linalg.solve(s.T @ s, s.T)
        projector = (
            (
                np.einsum(
                    "ki,cdk,kj->cidj",
                    transform.conj(),
                    symbols,
                    transform,
                    optimize=True,
                )
                / n
            )
            .reshape(6 * n, 6 * n)
            .real
        )
        eig, basis = np.linalg.eigh((projector + projector.T) / 2)
        b = basis[:, eig > 0.5]
        c = np.block(
            [[np.diag(table[labels.ravel(), i, j]) for j in range(6)] for i in range(6)]
        )
        a = np.kron(np.eye(6), np.ones((n, 1)))
        k, f = b.T @ c @ b / n, -b.T @ c @ a / n
        exact = d - f.T @ np.linalg.solve(k, f)
        return k, f, exact, b
    nodes = np.array(list(itertools.product((0, 1), repeat=3)))
    signs = 2 * nodes - 1
    h = np.array(lengths) / shape
    k, f = np.zeros((3 * n, 3 * n)), np.zeros((3 * n, 6))
    for cell in np.ndindex(shape):
        dofs = np.array(
            [
                component * n
                + np.ravel_multi_index(tuple((cell + node) % shape), shape)
                for node in nodes
                for component in range(3)
            ]
        )
        for xi in itertools.product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3):
            grad = [
                [
                    signs[a, i]
                    * np.prod([1 + signs[a, j] * xi[j] for j in range(3) if j != i])
                    / (4 * h[i])
                    for i in range(3)
                ]
                for a in range(8)
            ]
            b = np.zeros((6, 24))
            for a, (x, y, z) in enumerate(grad):
                b[:, 3 * a : 3 * a + 3] = np.array(
                    [[x, 0, 0], [0, y, 0], [0, 0, z], [0, z, y], [z, 0, x], [y, x, 0]]
                )
            b[3:] /= np.sqrt(2)
            c = table[labels[cell]]
            k[np.ix_(dofs, dofs)] += b.T @ c @ b / (8 * n)
            f[dofs] -= b.T @ c / (8 * n)
    return k, f, d - f.T @ np.linalg.pinv(k, rcond=1e-12, hermitian=True) @ f, None


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 3, 3), (3, 4, 5), (4, 4, 4)])
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_six_load_public_tensor_against_independent_direct_system(shape, kind):
    labels, table, c0 = audit_case(shape)
    _, _, exact, _ = explicit_problem(kind, labels, table)
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        lengths=LENGTHS,
        reference=c0,
        rtol=1e-10,
        verify_tensor=True,
        verification_rtol=1e-8,
    )
    np.testing.assert_allclose(result["C"], exact, atol=1e-11, rtol=2e-12)
    v = result["tensor_verification"]
    assert v["all_cross_products"] and v["six_loads"] and v["accepted"]
    assert v["tensor_target_passed"] and v["arithmetic"] == "float64"
    assert (
        not v["certified"]
        and not result["energy_error_bound"]["returned_matrix_certified"]
    )
    np.testing.assert_array_equal(result["C"], result["C"].T)


def gpu():
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    return cp


@pytest.mark.gpu
def test_gani_fp32_bracket_holds_and_verified_cross_products(monkeypatch):
    gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case()
    rounded = table.astype(np.float32).astype(float)
    _, _, exact_rounded, _ = explicit_problem("gani", labels, rounded)
    _, _, exact_original, _ = explicit_problem("gani", labels, table)
    production = homogenize(
        labels, table, dtype="float32", reference=c0, lengths=LENGTHS, rtol=1e-5
    )
    err = np.diag(production["C"] - exact_rounded)
    bound = production["energy_error_bound"]
    # The FP32 configuration evaluates the rounded coefficients in df64 and
    # assembles by the residual-corrected identity (as HEX8), so the former
    # counterexample of the FP32 strain-field assembly no longer occurs: the
    # error of the rounded problem lies in its algebraic bracket.
    assert production["precision_contract"]["evaluation"] == ["df64"]
    assert np.all(err <= np.array(bound["upper"]) * (1 + 1e-3))
    assert np.all(err >= np.array(bound["lower"]) * (1 - 1e-3))
    assert bound["scope"] == "algebraic_error_of_exact_field_energy"
    assert not bound["assembly_rounding_included"]
    verified = homogenize(
        labels,
        table,
        dtype="float32",
        reference=c0,
        lengths=LENGTHS,
        rtol=1e-5,
        verify_tensor=True,
        verification_rtol=1e-8,
    )
    np.testing.assert_allclose(
        verified["C_production"], production["C"], atol=2e-13, rtol=0
    )
    assert np.linalg.norm(verified["C"] - exact_original) < 0.01 * np.linalg.norm(
        production["C"] - exact_original
    )
    v = verified["tensor_verification"]
    assert v["coefficient_source"] == "original_input_float64"
    error = verified["C"] - exact_original
    pad = 1e-12
    assert np.all(np.diag(error) <= np.array(v["energy_error_bound"]["upper"]) + pad)
    assert np.linalg.eigvalsh(error)[0] >= -pad
    assert (
        np.linalg.eigvalsh(error - v["G0"] / v["spectral_bounds"]["gamma_plus"])[0]
        >= -pad
    )
    assert (
        np.linalg.eigvalsh(v["G0"] / v["spectral_bounds"]["gamma_minus"] - error)[0]
        >= -pad
    )


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize("evaluation", ["df64", "float64"])
def test_mixed_verification_and_actual_evaluation_metadata(
    kind, evaluation, monkeypatch
):
    gpu()
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    labels, table, c0 = audit_case()
    result = homogenize(
        labels,
        table,
        discretization=kind,
        reference=c0,
        lengths=LENGTHS,
        rtol=1e-6,
        evaluation=evaluation,
        verify_tensor=True,
    )
    assert result["tensor_verification"]["native_residual_passed"]
    for a in result["audits"]:
        assert a["status"] == "accepted" and not a["recursive_residual_is_acceptance"]
        assert (
            a["problem_precision"] == "float64" and a["working_precision"] == "float32"
        )
        assert a["restarts"] is not None and a["replacements"] is not None
    if kind == "gani":
        assert result["precision_contract"]["evaluation"] == ["float64_library"]


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_energy_zero_load_and_reference_execution_are_honest(kind):
    gpu()
    result = homogenize(
        np.zeros((4, 4, 4), int),
        np.eye(6)[None],
        discretization=kind,
        energy_rtol=1e-8,
        verify_tensor=True,
    )
    assert result["tensor_verification"]["tensor_target_passed"]
    np.testing.assert_allclose(result["C"], np.eye(6), atol=1e-14)
    if kind == "hex8":
        labels, table, c0 = audit_case()
        staged = homogenize(
            labels, table, discretization=kind, execution="reference", reference=c0
        )
        assert staged["precision_contract"]["evaluation"] == ["float64"]


def test_prepared_fp32_descriptor_does_not_claim_to_recover_original_coefficients():
    labels, table, c0 = audit_case((3, 3, 3))
    result = homogenize(
        IndexedMaterial(labels, table, dtype="float32"),
        discretization="hex8",
        backend="numpy",
        reference=c0,
        rtol=2e-5,
        verify_tensor=True,
    )
    assert (
        result["tensor_verification"]["coefficient_source"]
        == "descriptor_coefficients_after_rounding"
    )


def test_verification_rejects_original_residual_even_if_production_was_accepted():
    labels, table, c0 = audit_case((3, 3, 3))
    fields = np.zeros((6, 3, *labels.shape))
    with pytest.raises(TensorVerificationError) as failure:
        verify_effective_tensor(
            IndexedMaterial(labels, table),
            fields,
            kind="hex8",
            lengths=LENGTHS,
            reference=c0,
            production=np.eye(6),
            rtol=1e-6,
            atol=0,
        )
    assert failure.value.status == "fresh_residual_rejected"
    assert not failure.value.diagnostics["accepted"]


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_missing_indicators_are_explicit_and_reports_are_json_finite(kind):
    labels, table, c0 = audit_case((3, 3, 3))
    result = homogenize(
        labels, table, discretization=kind, backend="numpy", reference=c0
    )
    bracket = result["energy_error_bound"]
    json.dumps(bracket, allow_nan=False)
    assert bracket["available"] and np.isfinite(bracket["upper"]).all()
    if kind == "gani":
        assert bracket["G0"] == [None] * 6
        assert bracket["method"] == ["projected_stress_upper"] * 6
        assert bracket["lower"] == [0.0] * 6


@pytest.mark.parametrize("bad", [-1, 0, 1, np.inf, np.nan, True, [1e-6], 1j])
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_invalid_energy_target_is_rejected_before_dispatch(bad, kind):
    with pytest.raises(ValueError, match="energy tolerance"):
        homogenize(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            discretization=kind,
            backend="numpy",
            energy_rtol=bad,
        )


def test_mixed_energy_capacity_is_checked_before_allocating(monkeypatch):
    from cufhom.fft.linear import framework

    monkeypatch.setattr(
        framework,
        "_gani_residency",
        lambda material, configuration="mixed", energy=False, library=False: None,
    )
    with pytest.raises(MemoryError, match="residency budget"):
        homogenize(np.zeros((2, 2, 2), int), np.eye(6)[None], energy_rtol=1e-6)


def test_extreme_spd_material_requires_explicit_higher_working_precision():
    q = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0]
    c = q @ np.diag([1, 1, 1, 1, 1, 1e-10]) @ q.T
    assert np.linalg.eigvalsh(c)[0] > 0
    assert np.linalg.eigvalsh(c.astype(np.float32).astype(float))[0] < 0
    with pytest.raises(ValueError, match="working=float64"):
        # The screen precedes device allocation, even on a CPU-only host.
        Hex8Operator(
            np.zeros((4, 4, 4), int), c[None], reference=np.eye(6), working="float32"
        )
    assert working_material_check(c[None], "float64")["material_rounding_checked"]


def test_element_rounding_has_a_separate_finiteness_screen():
    with pytest.raises(ValueError, match="Working element table is nonfinite"):
        Hex8Operator(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            backend="numpy",
            dtype="float32",
            lengths=(1e-25, 1e-25, 1e-25),
        )


def test_actual_rounded_element_diagnostics_do_not_claim_global_spd():
    labels, table, _ = audit_case((3, 3, 3))
    op = Hex8Operator(labels, table, backend="numpy")
    diagnostic = _check_working_elements(op.ke, np.ones(3) / 3, np.float32)
    assert diagnostic["minimum_strain_eigenvalue"] > 0
    assert not diagnostic["global_operator_certified"]


def test_fresh_stagnation_is_bounded_and_progress_resets_it():
    progress = FreshProgress()
    for i in range(5):
        progress.check(1.0, i)
    progress.check(0.1, 5)
    for i in range(4):
        progress.check(0.1, i + 6)
    with pytest.raises(LinearSolveError) as failure:
        progress.check(0.1, 10)
    assert failure.value.status == "precision_limit" and failure.value.iterations == 10
    with pytest.raises(LinearSolveError) as failure:
        FreshProgress().check(np.nan, 1)
    assert failure.value.status == "nonfinite"


@pytest.mark.gpu
def test_underflowed_gani_step_cannot_be_a_zero_update_success():
    cp = gpu()
    from cufhom.fft.linear.energy import _STEP

    kernel = cp.RawKernel(_STEP, "energy_step")
    step = cp.empty(1, cp.float32)
    kernel((1,), (1,), (np.float64(1e-300), cp.asarray([1e300]), step))
    assert not np.isfinite(float(step[0]))


@pytest.mark.gpu
@pytest.mark.parametrize(
    "denominator,numerator",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (np.nan, 1.0),
        (np.inf, 1.0),
        (1e-300, 1e300),
        (1e300, 1e-300),
        (1.0, 1e-100),
        (1e-100, 1.0),
    ],
)
def test_hex8_unsafe_curvature_or_step_is_an_explicit_device_failure(
    denominator, numerator
):
    cp = gpu()
    from cufhom.fft.linear.hex8_general import _reduction_kernel

    state, scalar = cp.zeros(2, cp.int32), cp.zeros(8, cp.float64)
    scalar[0] = numerator
    kernel = _reduction_kernel("hex8_pcg_alpha", np.float32)
    kernel(
        (1,),
        (256,),
        (
            cp.asarray([denominator]),
            np.int32(1),
            scalar,
            state,
            cp.empty(12),
            np.int32(1),
        ),
    )
    assert int(state[0]) == 2


@pytest.mark.gpu
def test_gani_rejected_false_candidates_stop_at_precision_limit():
    cp = gpu()
    from cufhom.fft.linear.krylov import fused_cg

    inspections = []

    def replace(x, r):
        inspections.append(1)
        x.fill(0)
        r.fill(1)
        return 1.0

    with pytest.raises(LinearSolveError) as failure:
        fused_cg(
            None,
            cp.ones(1, cp.complex64),
            lambda a, b: 1.0,
            0.0,
            100,
            action_and_dot=lambda *a, **kw: (None, cp.asarray(1.0, cp.float64)),
            device_update_callback=lambda *a: 0.0,
            replace=replace,
            energy=lambda *a: True,
            accept=lambda *a: 1.0,
            fresh_metric=lambda: 1.0,
        )
    assert failure.value.status == "precision_limit"
    assert len(inspections) == 6 and failure.value.iterations == 6


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_verification_reports_an_unattainable_tensor_target(kind):
    labels, table, c0 = audit_case((3, 3, 3))
    with pytest.raises(TensorVerificationError) as failure:
        homogenize(
            labels,
            table,
            discretization=kind,
            backend="numpy",
            reference=c0,
            verify_tensor=True,
            rtol=1e-3,
            verification_rtol=1e-30,
        )
    assert failure.value.status == "energy_target_rejected"


@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize(
    "geometry,contrast", [("laminate", 10.0), ("inclusion", 1e3), ("interfaces", 1e4)]
)
def test_rotated_anisotropy_and_high_contrast_against_direct_reference(
    kind, geometry, contrast
):
    shape = (3, 3, 3)
    x, y, z = np.indices(shape)
    labels = (
        (x == 0)
        if geometry == "laminate"
        else (x == 1) & (y == 1) & (z == 1)
        if geometry == "inclusion"
        else (x + y + z) % 2
    ).astype(int)
    rotation = np.linalg.qr(np.random.default_rng(37).normal(size=(3, 3)))[0]
    mandel = np.zeros((6, 6))
    for a in range(6):
        tensor = np.zeros((3, 3))
        if a < 3:
            tensor[a, a] = 1
        else:
            i, j = [(1, 2), (0, 2), (0, 1)][a - 3]
            tensor[i, j] = tensor[j, i] = 1 / np.sqrt(2)
        transformed = rotation @ tensor @ rotation.T
        mandel[:, a] = [
            *np.diag(transformed),
            np.sqrt(2) * transformed[1, 2],
            np.sqrt(2) * transformed[0, 2],
            np.sqrt(2) * transformed[0, 1],
        ]
    base = mandel @ np.diag([3.0, 4.0, 7.0, 2.0, 5.0, 6.0]) @ mandel.T
    table = np.array([base, contrast * base])
    _, _, exact, _ = explicit_problem(kind, labels, table)
    result = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        lengths=LENGTHS,
        rtol=1e-8,
        verify_tensor=True,
    )
    np.testing.assert_allclose(result["C"], exact, rtol=3e-9, atol=3e-9)


def test_near_incompressibility_has_a_working_precision_diagnostic():
    c = 2 * np.eye(6)
    c[:3, :3] += 1e4
    assert working_material_check(c[None], "float32")["maximum_condition"] > 1e4
    c[:3, :3] += 1e8
    with pytest.raises(ValueError, match="working=float64"):
        working_material_check(c[None], "float32")
    assert working_material_check(c[None], "float64")["material_rounding_checked"]


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_budget_exhaustion_has_load_and_precision_context(backend):
    if backend == "cupy":
        gpu()
    labels, table, c0 = audit_case((3, 3, 3))
    with pytest.raises(LinearSolveError) as failure:
        homogenize(
            labels,
            table,
            discretization="hex8",
            backend=backend,
            reference=c0,
            rtol=1e-12,
            maxiter=1,
        )
    assert failure.value.status == "maxiter"
    assert failure.value.diagnostics["load_id"] == 0
    assert failure.value.diagnostics["accepted"] is False


@pytest.mark.gpu
def test_failed_cuda_solve_does_not_leave_an_operator_pcg_cycle(monkeypatch):
    gpu()
    from cufhom.fft.linear import hex8_general

    operators = []

    class Observed(hex8_general.Hex8Operator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            operators.append(self)

    labels, table, c0 = audit_case()
    with pytest.raises(LinearSolveError):
        hex8_general.homogenize_hex8(
            labels, table, operator_type=Observed, reference=c0, maxiter=1, rtol=1e-12
        )
    assert operators[0]._pcg is None


@pytest.mark.parametrize("shape", [(3, 3, 3), (4, 4, 4), (3, 4, 5)])
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_full_residual_gram_and_energy_identity_at_unconverged_fields(kind, shape):
    labels, table, c0 = audit_case(shape)
    k, f, exact, basis = explicit_problem(kind, labels, table)
    u = 0.01 * np.random.default_rng(719).normal(size=f.shape)
    residual = f - k @ u
    d = table[labels].mean(axis=(0, 1, 2))
    expected_energy = d - f.T @ u - u.T @ f + u.T @ k @ u
    if kind == "gani":
        k0 = basis.T @ np.kron(c0, np.eye(labels.size)) @ basis / labels.size
        gram = residual.T @ np.linalg.solve(k0, residual)
        fields = (basis @ u).T.reshape(6, 6, *shape)
    else:
        k0 = explicit_problem(kind, np.zeros_like(labels), c0[None])[0]
        gram = residual.T @ np.linalg.pinv(k0, rcond=1e-12, hermitian=True) @ residual
        fields = u.T.reshape(6, 3, *shape)
    with pytest.raises(TensorVerificationError) as failure:
        verify_effective_tensor(
            IndexedMaterial(labels, table),
            fields,
            kind=kind,
            lengths=LENGTHS,
            reference=c0,
            production=expected_energy,
            rtol=1e-12,
            atol=0.0,
        )
    report = failure.value.diagnostics
    np.testing.assert_allclose(report["C"], expected_energy, atol=2e-12, rtol=2e-13)
    np.testing.assert_allclose(report["G0"], gram, atol=2e-11, rtol=3e-12)
    actual_error = report["C"] - exact
    energy_error = (
        residual.T @ np.linalg.pinv(k, rcond=1e-12, hermitian=True) @ residual
    )
    np.testing.assert_allclose(actual_error, energy_error, atol=4e-11, rtol=1e-11)
    bounds = report["spectral_bounds"]
    assert np.linalg.eigvalsh(actual_error - gram / bounds["gamma_plus"])[0] >= -1e-11
    assert np.linalg.eigvalsh(gram / bounds["gamma_minus"] - actual_error)[0] >= -1e-11


@pytest.mark.parametrize("first,second", [(None, None), (0, None), (None, 0), (0, 0)])
def test_unrecorded_legacy_counters_are_not_reported_as_zero(first, second):
    from types import SimpleNamespace

    from cufhom_core.applications import _projected_strain_solve

    records = iter((first, second))
    calls = []

    def legacy_solve(**kwargs):
        count = next(records)
        calls.append(1)
        info = dict(
            kit=1 if len(calls) == 1 else 0,
            norm_res_log=[0.0],
            norm_res_rel=0.0,
            converged=True,
        )
        if count is not None:
            info.update(restarts=count, replacements=count)
        return kwargs["B"], info

    field = SimpleNamespace(val=np.zeros((1,)), N=(1,))
    _, info = _projected_strain_solve(
        legacy_solve,
        None,
        field,
        None,
        SimpleNamespace(solver=dict(kind="CG", maxiter=10)),
        lambda x: x,
    )
    assert info["true_residual_converged"]
    expected = 0 if first is not None and second is not None else None
    assert info["restarts"] == expected and info["replacements"] == expected
