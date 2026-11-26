"""Linear displacement PCG: reference accuracy, physical audits and failures."""

import json

import numpy as np
import pytest
from cufhom.fft.linear.solver import cp, solve_homogenization

pytestmark = pytest.mark.gpu


def case(shape):
    phase = np.zeros(shape, np.uint8)
    phase[1:3, 1:3, :] = 1
    ori = np.zeros(shape + (3,), np.float32)
    ori[:] = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    return dict(
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
        fft_backend="cupy",
        solver_real_dtype="float32",
        solver_rtol=1e-5,
    )


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")


@pytest.mark.parametrize("shape", [(5, 7, 5), (6, 4, 8)])
def test_six_load_cpu_reference_and_true_residual(shape, tmp_path):
    p = case(shape)
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    timing = tmp_path / "timing.json"
    candidate = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    assert np.linalg.norm(candidate - reference) / np.linalg.norm(reference) < 2e-6
    report = json.loads(timing.read_text())
    assert report["linear_formulation"] == "displacement"
    infos = report["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    assert len(infos) == 6
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )


def test_nonconvergence_is_not_accepted():
    with pytest.raises(RuntimeError):
        solve_homogenization(dict(case((6, 4, 8)), solver_maxiter=1))


def test_homogeneous_zero_rhs():
    p = case((4, 4, 4))
    p["phase_array"][:] = 0
    candidate = solve_homogenization(p)
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    np.testing.assert_allclose(candidate, reference, atol=1e-6)


@pytest.mark.parametrize("backend", ["cupy", "scipy", "cupy64"])
def test_backend_automatically_uses_compatible_route(backend, tmp_path):
    p = case((4, 4, 4))
    if backend == "scipy":
        p.update(fft_backend="scipy", solver_real_dtype="float64")
    elif backend == "cupy64":
        p["solver_real_dtype"] = "float64"
    timing = tmp_path / "selection.json"
    candidate = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    metadata = json.loads(timing.read_text())
    assert metadata["linear_formulation"] == (
        "displacement" if backend == "cupy" else "strain"
    )
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    np.testing.assert_allclose(candidate, reference, atol=3e-6, rtol=2e-6)


def test_audit_corrects_inaccurate_coordinate_iterate(monkeypatch):
    import cufhom.fft.linear.coordinates as linear_coordinates

    p = case((5, 7, 5))
    reference = solve_homogenization(p)
    # Simulate a false inner convergence; acceptance must use the physical
    # residual and recover through the original equations.
    monkeypatch.setattr(
        linear_coordinates,
        "fused_cg",
        lambda action, rhs, *a, **kw: (cp.zeros_like(rhs), 1),
    )
    corrected = solve_homogenization(p)
    np.testing.assert_allclose(corrected, reference, atol=3e-6, rtol=2e-6)


def test_default_smoothing_and_audits(tmp_path):
    p = case((5, 7, 5))
    timing = tmp_path / "smoothing.json"
    actual = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=2e-6)
    report = json.loads(timing.read_text())
    assert report["linear_formulation"] == "displacement"
    infos = report["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )
