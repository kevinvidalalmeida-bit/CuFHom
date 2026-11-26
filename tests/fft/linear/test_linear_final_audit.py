"""Reuse must preserve the complete fields and original physical audit."""

import json

import numpy as np
import pytest

from tests.fft.linear.test_linear_coordinates import case, cp, solve_homogenization

# Reexport the autouse fixture so pytest discovers it in this module.
from tests.fft.linear.test_linear_coordinates import gpu as gpu

pytestmark = pytest.mark.gpu


def library_route(monkeypatch):
    """These checks exercise the library FFT route; odd grids are fused now."""
    from cufhom.fft.linear import coordinates

    monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)


def run(parameters, path):
    options = dict(
        parameters,
        solution_field_return_in_memory=True,
        solver_timing_path=str(path),
        check_true_residual=True,
    )
    C = solve_homogenization(options)
    infos = json.loads(path.read_text())["cufhom_core_application_timing"][
        "primaldual"
    ]["primal"]["load_solver_infos"]
    assert len(infos) == 6 and all(
        i["true_residual_converged"]
        and i["true_norm_res_rel"] <= parameters["solver_rtol"]
        for i in infos
    )
    return C, options["_solution_fields_result"], infos


@pytest.mark.parametrize("shape", [(5, 7, 5), (6, 4, 8)])
def test_default_reuse_preserves_complete_fields_against_cpu(shape, tmp_path):
    parameters = case(shape)
    C, fields, infos = run(parameters, tmp_path / "gpu.json")
    reference, reference_fields, _ = run(
        dict(
            parameters,
            fft_backend="scipy",
            solver_real_dtype="float64",
            solver_rtol=1e-10,
        ),
        tmp_path / "cpu.json",
    )
    assert all(
        i["linear_final_strain_reuse"]
        and i["linear_final_audit_reuse"]
        and i["linear_final_spectrum_reuse"]
        for i in infos
    )
    np.testing.assert_allclose(C, reference, atol=3e-6, rtol=2e-6)
    for a, b in zip(fields, reference_fields):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 3e-5


def test_changed_material_warm_start_disables_retention(tmp_path):
    parameters = case((5, 7, 5))
    _, fields, _ = run(parameters, tmp_path / "first.json")
    target = dict(parameters, Em=4.1, Ef_L=82.0, initial_solution_fields=fields)
    C, new_fields, infos = run(target, tmp_path / "warm.json")
    baseline, baseline_fields, _ = run(
        dict(
            target, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10
        ),
        tmp_path / "base.json",
    )
    assert all(i["warm_start_accepted"] for i in infos)
    assert not any(
        i["linear_final_strain_reuse"] or i["linear_final_audit_reuse"] for i in infos
    )
    np.testing.assert_allclose(C, baseline, atol=3e-6, rtol=2e-6)
    for a, b in zip(new_fields, baseline_fields):
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 3e-5


def test_false_inner_exit_fails_without_switching_solver(tmp_path, monkeypatch):
    """An iterate that fails its audit fails the load: no strain-CG correction."""
    library_route(monkeypatch)
    import cufhom.fft.linear.coordinates as lc
    from cufhom.fft.linear.common import LinearSolveError

    parameters = case((5, 7, 5))

    def false_exit(action, rhs, *args, **kwargs):
        kwargs["residual_action"](rhs.copy())
        return cp.zeros_like(rhs), 1

    def forbidden(*args, **kwargs):
        pytest.fail("the core strain CG must not continue a failed coordinate solve")

    monkeypatch.setattr(lc, "fused_cg", false_exit)
    original = lc.make_linear_solver
    monkeypatch.setattr(
        lc, "make_linear_solver", lambda **kw: original(**dict(kw, fallback=forbidden))
    )
    with pytest.raises(RuntimeError) as caught:  # the core wraps the load failure
        run(parameters, tmp_path / "false.json")
    cause = caught.value.__cause__ or caught.value.__context__
    assert (
        isinstance(cause, LinearSolveError)
        and cause.status == "fresh_residual_rejected"
    )


def test_mismatched_physical_operator_keeps_full_audit(monkeypatch, tmp_path):
    library_route(monkeypatch)
    from cufhom_core.tensors.operators import Operator

    original = Operator.apply_material_result
    calls = []

    def mismatch(self, stress, material, projection, **kwargs):
        calls.append(1)
        return original(self, stress, object(), projection, **kwargs)

    monkeypatch.setattr(Operator, "apply_material_result", mismatch)
    _, _, infos = run(case((5, 7, 5)), tmp_path / "mismatch.json")
    assert len(calls) == 6
    assert all(
        i["linear_final_strain_reuse"] and not i["linear_final_audit_reuse"]
        for i in infos
    )


@pytest.mark.parametrize("invalid", ["shape", "dtype"])
def test_incompatible_spectrum_retains_full_audit(invalid, tmp_path, monkeypatch):
    library_route(monkeypatch)
    from cufhom_core.tensors import operators

    original = operators.Operator.apply_material_result
    reference, fields, _ = run(case((5, 7, 5)), tmp_path / "base.json")
    calls = []

    def incompatible(self, stress, material, projection, **kwargs):
        spectrum = kwargs["spectrum"]
        calls.append(1)
        if invalid == "shape":
            kwargs["spectrum"] = spectrum[:1]
        else:
            kwargs["spectrum"] = spectrum.astype(cp.complex128)
        return original(self, stress, material, projection, **kwargs)

    monkeypatch.setattr(operators.Operator, "apply_material_result", incompatible)
    C, new_fields, infos = run(case((5, 7, 5)), tmp_path / "new.json")
    assert len(calls) == 6 and not any(
        i["linear_final_spectrum_reuse"] or i["linear_final_audit_reuse"] for i in infos
    )
    np.testing.assert_array_equal(C, reference)
    for a, b in zip(fields, new_fields):
        np.testing.assert_array_equal(a, b)


def test_profile_skips_only_material_and_forward(tmp_path, monkeypatch):
    library_route(monkeypatch)
    from cufhom_core.tensors.operators import Operator

    original = Operator.apply_material_result
    profiles = []

    def profiled(self, *args, **kwargs):
        stats = {name: dict(seconds=0.0, calls=0) for name in ["A", "FN", "hG", "FiN"]}
        self._component_profile = stats
        try:
            return original(self, *args, **kwargs)
        finally:
            profiles.append(stats)
            del self._component_profile

    monkeypatch.setattr(Operator, "apply_material_result", profiled)
    _, _, infos = run(case((5, 7, 5)), tmp_path / "new.json")
    assert len(profiles) == 6 and all(i["linear_final_spectrum_reuse"] for i in infos)
    for p in profiles:
        assert p["A"]["calls"] == p["FN"]["calls"] == 0
        assert p["hG"]["calls"] == p["FiN"]["calls"] == 1


def test_strain_route_returns_projected_fields(monkeypatch, tmp_path):
    # Without the final projection the strain-CG iterate of this case is
    # incompatible to ~5e-6; projected, only the rounding of one transform pair remains.
    import cufhom.fft.linear.coordinates as lc
    from cufhom.fft.operators import MixedProjector

    shape = (16, 18, 20)
    parameters = dict(
        case(shape),
        phase_array=(np.random.default_rng(3).random(shape) < 0.3).astype(np.uint8),
    )
    monkeypatch.setattr(lc, "make_linear_solver", lambda **kw: kw["fallback"])
    _, fields, infos = run(parameters, tmp_path / "strain.json")
    P = MixedProjector(shape, (1.0, 1.0, 1.0), np.ones(6, bool), cp)
    for e in map(cp.asarray, fields):
        assert float(cp.linalg.norm(P(e) - e) / cp.linalg.norm(e)) < 1e-6
    assert all(i["true_residual_converged"] for i in infos)
