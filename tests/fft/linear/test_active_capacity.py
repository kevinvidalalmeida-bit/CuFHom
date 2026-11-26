"""Force the FP64 active-state fallback on small, independently solvable cells."""

import json

import numpy as np
import pytest

pytest.importorskip("cupy")
from cufhom.fft.linear import active_capacity as cap
from cufhom.fft.linear import coordinates
from cufhom.fft.linear.fused import grid_supported as fused_grid

from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("actor", ["C", "Clib", "B"])
@pytest.mark.parametrize("shape", [(12, 10, 8), (9, 11, 13)])
def test_active_capacity_audits_and_independent_reference(
    monkeypatch, tmp_path, actor, shape
):
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    p = case(shape)
    p.update(
        solver_real_dtype="float64",
        solver_fp64_coordinates=True,
        solver_rtol=1e-9,
        solver_maxiter=1000,
    )
    reference = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-12))
    monkeypatch.setattr(cap, "required", lambda *args: True)
    monkeypatch.setattr(cap, "CHUNK", 37)
    if actor == "B":
        monkeypatch.setattr(
            coordinates, "make_linear_solver", lambda **kw: kw["fallback"]
        )
    if actor == "Clib":
        monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)
    path = tmp_path / "active.json"
    actual = solve_homogenization(dict(p, solver_timing_path=str(path)))
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    infos = json.loads(path.read_text())["cufhom_core_application_timing"][
        "primaldual"
    ]["primal"]["load_solver_infos"]
    assert len(infos) == 6
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-9 for i in infos
    )
    assert all(
        i["active_state_strategy"]
        == (
            "pinned_x_r_gpu_direction"
            if actor == "B"
            else "pinned_x_gpu_residual_direction"
        )
        for i in infos
    )
    assert all(
        i["linear_fused_fft"] == (actor == "C" and fused_grid(shape)) for i in infos
    )


@pytest.mark.parametrize("zero", [True, False])
def test_zero_rhs_and_budget_are_not_false_acceptance(monkeypatch, zero):
    monkeypatch.setattr(cap, "required", lambda *args: True)
    p = case((8, 8, 8))
    p.update(
        solver_real_dtype="float64",
        solver_fp64_coordinates=True,
        solver_rtol=1e-9,
        solver_maxiter=1,
    )
    if zero:
        p["phase_array"][:] = 0
        actual = solve_homogenization(p)
        reference = solve_homogenization(
            dict(p, fft_backend="scipy", solver_rtol=1e-12)
        )
        np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
    else:
        with pytest.raises(RuntimeError, match="physical audit"):
            solve_homogenization(p)


@pytest.mark.parametrize("actor", ["C", "Clib", "B"])
def test_active_capacity_restart_after_failed_physical_audit(
    monkeypatch, tmp_path, actor
):
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    monkeypatch.setattr(cap, "required", lambda *args: True)
    if actor == "B":
        monkeypatch.setattr(
            coordinates, "make_linear_solver", lambda **kw: kw["fallback"]
        )
    elif actor == "Clib":
        monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)
    original = cap.ActiveCapacity.physical

    def inspection(self, strain, mean, **kw):
        result = original(self, strain, mean, **kw)
        if kw.get("rhs"):
            self.test_rhs_norm = result[0]
        elif not hasattr(self, "test_failed_once"):
            self.test_failed_once = True
            return max(result, 1.1e-9 * self.test_rhs_norm)
        return result

    monkeypatch.setattr(cap.ActiveCapacity, "physical", inspection)
    p = case((12, 10, 8))
    p.update(
        solver_real_dtype="float64",
        solver_fp64_coordinates=True,
        solver_rtol=1e-9,
        solver_maxiter=1000,
    )
    path = tmp_path / "restart.json"
    actual = solve_homogenization(dict(p, solver_timing_path=str(path)))
    reference = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-12))
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    infos = json.loads(path.read_text())["cufhom_core_application_timing"][
        "primaldual"
    ]["primal"]["load_solver_infos"]
    assert infos[0]["physical_audit_tightenings"] >= 1
    assert all(i["true_norm_res_rel"] <= 1e-9 for i in infos)


@pytest.mark.parametrize("actor", ["C", "Clib", "B"])
def test_nondefault_stream_preserves_inputs(monkeypatch, actor):
    from cufhom_core.tensors.fft import cp

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    monkeypatch.setattr(cap, "required", lambda *args: True)
    if actor == "B":
        monkeypatch.setattr(
            coordinates, "make_linear_solver", lambda **kw: kw["fallback"]
        )
    elif actor == "Clib":
        monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)
    p = case((12, 10, 8))
    saved = [p[k].copy() for k in ["phase_array", "ori_array"]]
    p.update(
        solver_real_dtype="float64", solver_fp64_coordinates=True, solver_rtol=1e-9
    )
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        actual = solve_homogenization(p)
    stream.synchronize()
    reference = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-12))
    np.testing.assert_allclose(actual, reference, atol=2e-10, rtol=2e-10)
    for key, value in zip(["phase_array", "ori_array"], saved):
        np.testing.assert_array_equal(p[key], value)


def test_selector_does_not_change_fp32_or_384(monkeypatch):
    monkeypatch.setattr(cap, "_device_capacity", lambda *args: 16 * 2**30)
    assert not cap.required((512,) * 3, np.float32)
    assert not cap.required((384,) * 3, np.float64)
    assert cap.required((512,) * 3, np.float64)
