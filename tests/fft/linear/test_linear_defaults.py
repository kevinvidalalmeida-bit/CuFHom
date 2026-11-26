"""Automatic fast solves must match CPU64 and retain physical input support."""

import json

import numpy as np
import pytest
from cufhom.fft.linear.solver import cp, solve_homogenization


@pytest.mark.parametrize(
    "old_option", [dict(cupy_matrix_reference=True), dict(cupy_reference_eta=2.0)]
)
def test_retired_metric_is_explicit(old_option):
    with pytest.raises(ValueError, match="retired"):
        solve_homogenization(old_option)


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize("geometry", ["block", "laminate", "inclusions"])
def test_minimal_gpu_six_loads_match_cpu64(shape, geometry, tmp_path):
    if cp is None:
        pytest.skip("CuPy unavailable")
    phase = np.zeros(shape, np.uint8)
    if geometry == "block":
        phase[1:3, 1:3, :] = 1
    elif geometry == "laminate":
        phase[:2] = 1
    else:
        # Disconnected compact inclusions; no fibre axis or aspect ratio.
        coordinates = np.indices(shape)
        for center in ((1, 1, 1), tuple(n - 2 for n in shape)):
            phase[
                np.sum(
                    (coordinates - np.array(center)[:, None, None, None]) ** 2, axis=0
                )
                <= 1
            ] = 1
    ori = np.empty(shape + (3,), np.float32)
    ori[:] = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    p = dict(
        input_dir=".",
        phase_array=phase,
        ori_array=ori,
        Em=3.0,
        nu_m=0.3,
        Ef_L=70.0,
        Ef_T=8.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
    )
    ref = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-10))
    timing = tmp_path / "gpu.json"
    result = solve_homogenization(
        dict(p, fft_backend="cupy", solver_rtol=1e-5, solver_timing_path=str(timing))
    )
    assert np.linalg.norm(result - ref) / np.linalg.norm(ref) < 2e-6
    detail = json.loads(timing.read_text())
    assert detail["linear_formulation"] == "displacement"
    assert detail["cfield_indexed"] and detail["projection_storage"] == "direct"
    infos = detail["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    assert len(infos) == 6
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )


@pytest.mark.gpu
def test_general_anisotropic_cfield_retains_compatible_route(tmp_path):
    if cp is None:
        pytest.skip("CuPy unavailable")
    shape = (5, 4, 7)
    rng = np.random.default_rng(309)
    base = rng.normal(size=(6, 6))
    base = base.T @ base + 3 * np.eye(6)
    # Smooth arbitrary SPD tensor field, independent of phase/orientation labels.
    scale = 1 + 0.3 * np.sin(2 * np.pi * np.indices(shape)[0] / shape[0])
    cfield = base[:, :, None, None, None] * scale
    saved = cfield.copy()
    p = dict(
        input_dir=".",
        phase_array=np.zeros(shape, np.uint8),
        ori_array=np.zeros(shape + (3,), np.float32),
        Cfield_array=cfield,
        Em=3.0,
        nu_m=0.3,
        Ef_L=70.0,
        Ef_T=8.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
    )
    ref = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-10))
    timing = tmp_path / "general.json"
    result = solve_homogenization(
        dict(p, fft_backend="cupy", solver_rtol=1e-5, solver_timing_path=str(timing))
    )
    assert np.linalg.norm(result - ref) / np.linalg.norm(ref) < 2e-6
    np.testing.assert_array_equal(cfield, saved)
    detail = json.loads(timing.read_text())
    assert not detail["cfield_indexed"]
    assert detail["linear_formulation"] == "strain"
    assert detail["load_solver_summary"]["all_converged"]


@pytest.mark.parametrize(
    "option,value",
    [
        ("linear_formulation", "strain"),
        ("linear_preconditioning", "matrix"),
        ("linear_kernel_fusion", False),
        ("linear_energy_update", False),
        ("linear_residual_smoothing", False),
        ("linear_fft_autotune", False),
        ("linear_final_audit_reuse", False),
        ("linear_final_spectrum_reuse", False),
        ("solver_profile", "timing"),
    ],
)
def test_retired_solver_selection_fails_before_simulation(option, value):
    with pytest.raises(ValueError, match="remove these parameters: " + option):
        solve_homogenization({option: value})


@pytest.mark.gpu
def test_missing_backend_uses_cuda_automatically(tmp_path):
    if cp is None:
        pytest.skip("CuPy unavailable")
    shape = (5, 7, 5)
    phase = np.zeros(shape, np.uint8)
    phase[1:3, 1:3, :] = 1
    ori = np.zeros(shape + (3,), np.float32)
    ori[..., 0] = 1
    p = dict(
        input_dir=".",
        phase_array=phase,
        ori_array=ori,
        Em=3.0,
        nu_m=0.3,
        Ef_L=70.0,
        Ef_T=8.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
    )
    ref = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-10))
    path = tmp_path / "automatic.json"
    actual = solve_homogenization(dict(p, solver_timing_path=str(path)))
    detail = json.loads(path.read_text())
    assert detail["fft_backend"]["requested"] == "cupy"
    assert detail["linear_formulation"] == "displacement"
    assert all(
        i["true_residual_converged"]
        for i in detail["cufhom_core_application_timing"]["primaldual"]["primal"][
            "load_solver_infos"
        ]
    )
    np.testing.assert_allclose(actual, ref, atol=3e-6, rtol=2e-6)


def test_default_without_cuda_preserves_analytic_cpu_solution(monkeypatch, tmp_path):
    from cufhom.fft.linear import solver

    monkeypatch.setattr(solver, "_CUPY_AVAILABLE", False)
    shape = (4, 4, 4)
    p = dict(
        input_dir=".",
        phase_array=np.zeros(shape, np.uint8),
        ori_array=np.zeros(shape + (3,), np.float32),
        Em=3.0,
        nu_m=0.3,
        Ef_L=70.0,
        Ef_T=8.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
        solver_timing_path=str(tmp_path / "cpu.json"),
    )
    actual = solve_homogenization(p)
    lam = 3.0 * 0.3 / ((1 + 0.3) * (1 - 2 * 0.3))
    mu = 3.0 / (2 * (1 + 0.3))
    expected = np.eye(6) * 2 * mu
    expected[:3, :3] += lam
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    detail = json.loads((tmp_path / "cpu.json").read_text())
    assert detail["fft_backend"]["requested"] == "scipy"
    assert detail["solver_real_dtype"] == "float64"
    assert "fft_backend" not in p
