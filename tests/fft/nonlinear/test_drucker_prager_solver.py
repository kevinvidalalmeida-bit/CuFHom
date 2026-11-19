"""Pressure-dependent equilibrium, compact fused maps and restart on GPU."""

import numpy as np
import pytest
from cufhom.fft.nonlinear import solver
from cufhom.fft.nonlinear.krylov import TangentPipeline
from cufhom.materials.drucker_prager import CuPyDruckerPrager, DruckerPragerMaterial
from cufhom.materials.plasticity import initial_state

cp = solver.cp
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(cp is None, reason="CuPy unavailable"),
]


@pytest.fixture(autouse=True)
def gpu_fft_backend():
    old = solver.core_fft.get_fft_backend_status()
    solver.core_fft.set_fft_backend("cupy")
    yield
    solver.core_fft.set_fft_backend(old["requested"])


def problem(**kwargs):
    shape = (8, 6, 10)
    phase = np.zeros(shape, np.uint8)
    phase[2:4, 2:4, :] = 1
    ori = np.zeros(shape + (3,))
    ori[..., 2] = 1
    return dict(
        phase_array=phase,
        ori_array=ori,
        backend="cupy",
        model="drucker_prager",
        dtype="float32",
        path=[0.0, 0.006, 0.002],
        rtol=1e-5,
        return_fields=True,
        **kwargs,
    )


def physical(result):
    return [
        {k: v for k, v in row.items() if k != "seconds"} for row in result["history"]
    ]


@pytest.mark.parametrize("dilatancy", [None, 0.1, 0.0])
@pytest.mark.parametrize("axis", [0, 3])
def test_gpu_load_unload_audits(dilatancy, axis):
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy), axis=axis)
    result = solver.solve_nonlinear_homogenization(p)
    assert result["integrator"] == "backward_euler"
    assert result["linear_solver"] == ("cg" if dilatancy is None else "bicgstab")
    assert result["coordinate_pcg"] == "pipeline"
    assert not result["rejected_steps"]
    for r in result["history"]:
        assert r["residual"] <= result["rtol"] and r["macro_residual"] <= result["rtol"]
        assert (
            r["yield_violation"] <= result["rtol"]
            and r["hill_mandel_error"] <= 5 * result["rtol"]
        )
        assert r["plastic_dissipation"] >= 0
    for a in (
        result["fields"]["strain"],
        result["fields"]["stress"],
        result["state"].plastic_strain,
        result["state"].equivalent,
    ):
        assert np.isfinite(a).all()


@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_restart_and_fingerprint(tmp_path, dilatancy):
    # Predictor history is a solver acceleration, not part of the physical
    # checkpoint. Disable it in both paths to demand bitwise restart identity.
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy), predictor=False)
    path = tmp_path / "state.npz"
    first = solver.solve_nonlinear_homogenization(
        dict(p, path=[0.0, 0.006], checkpoint_path=path)
    )
    resumed = solver.solve_nonlinear_homogenization(
        dict(p, path=[0.006, 0.002], restart_from=path)
    )
    direct = solver.solve_nonlinear_homogenization(p)
    assert physical(resumed)[: len(first["history"])] == physical(first)
    for key in ("strain", "stress"):
        np.testing.assert_allclose(
            [r[key] for r in resumed["history"]],
            [r[key] for r in direct["history"]],
            atol=1e-7,
            rtol=1e-5,
        )
    for key in ("strain", "stress"):
        np.testing.assert_allclose(
            resumed["fields"][key], direct["fields"][key], atol=1e-7, rtol=1e-5
        )
    np.testing.assert_allclose(
        resumed["state"].plastic_strain,
        direct["state"].plastic_strain,
        atol=1e-7,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        resumed["state"].equivalent, direct["state"].equivalent, atol=1e-7, rtol=1e-5
    )
    with pytest.raises(ValueError, match="mismatch"):
        solver.solve_nonlinear_homogenization(
            dict(p, model="j2", matrix={}, path=[0.006, 0.002], restart_from=path)
        )
    with pytest.raises(ValueError, match="mismatch"):
        solver.solve_nonlinear_homogenization(
            dict(
                p,
                matrix=dict(friction=0.4, dilatancy=dilatancy),
                path=[0.006, 0.002],
                restart_from=path,
            )
        )


@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_fused_tangent_matches_production_fft(dilatancy):
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy))
    m = DruckerPragerMaterial(**p["matrix"])
    ids, table = solver._table(
        p["phase_array"],
        p["ori_array"],
        m,
        dict(E_L=70.0, E_T=8.0, nu_LT=0.25, nu_TT=0.3, G_LT=5.0),
    )
    ids = cp.asarray(ids)
    table = cp.asarray(table, dtype="float32")
    projector = solver.MixedProjector(
        ids.shape, (1.0, 1.0, 1.0), [True, False, False, False, False, False], cp
    )
    engine = TangentPipeline(projector, m, ids, table)
    gpu = CuPyDruckerPrager(cp, m, ids, table, "float32")
    rng = np.random.default_rng(734)
    strain = cp.asarray(rng.normal(0, 0.01, (6,) + ids.shape), dtype="float32")
    response = gpu.update(strain, initial_state(ids.shape, cp, cp.float32))
    pipeline = engine.pipeline
    pipeline.bind(
        response.a,
        response.b,
        response.direction,
        response.bulk,
        response.coupling_to_direction,
        response.coupling_to_pressure,
    )
    z = engine.Q.adjoint(cp.asarray(rng.normal(size=(6,) + ids.shape), dtype="float32"))
    expected = engine.Q.adjoint(gpu.tangent(engine.Q.strain(z), response))
    actual = pipeline.to_production(pipeline.matvec(pipeline.to_fused(z)))
    cp.testing.assert_allclose(actual, expected, rtol=4e-5, atol=3e-5)
    assert bool(cp.isfinite(pipeline.matvec(cp.zeros_like(pipeline.to_fused(z)))).all())


def test_configuration_rejects_wrong_integrator():
    with pytest.raises(ValueError, match="backward_euler"):
        solver.solve_nonlinear_homogenization(problem(integrator="midpoint"))
    with pytest.raises(ValueError, match="model must"):
        solver.solve_nonlinear_homogenization(problem() | dict(model="unknown"))


def test_j2_default_remains_midpoint():
    result = solver.solve_nonlinear_homogenization(
        problem() | dict(model="j2", matrix={}, path=[0.0, 0.001])
    )
    assert (
        result["model"] == "j2"
        and result["integrator"] == "midpoint"
        and result["linear_solver"] == "cg"
    )


@pytest.mark.parametrize("dilatancy", [None, 0.1])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ptc_shift_and_classic_fallback(dtype, dilatancy):
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy), globalization="ptc")
    p.update(dtype=dtype, path=[0.0, 0.003], coordinate_pcg="classic")
    result = solver.solve_nonlinear_homogenization(p)
    assert not result["rejected_steps"]
    assert max(r["residual"] for r in result["history"]) <= 1e-5
    assert any(r["ptc_iterations"] > 0 for r in result["history"])


@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_rejected_attempt_does_not_commit(tmp_path, monkeypatch, dilatancy):
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy))
    p.update(
        path=[0.0, 0.002],
        initial_increment=0.002,
        min_increment=0.002,
        max_cg=1,
        globalization="newton",
        checkpoint_path=tmp_path / "confirmed.npz",
    )
    with pytest.raises(solver.NonlinearConvergenceError):
        solver.solve_nonlinear_homogenization(p)
    with np.load(p["checkpoint_path"]) as ck:
        for key in ("strain", "plastic_strain", "equivalent"):
            assert not np.any(ck[key])


@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_cutback_preserves_confirmed_state(monkeypatch, dilatancy):
    original = TangentPipeline.solve
    calls = [0]

    def fail_second(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise solver.NonlinearConvergenceError("injected DP tangent failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TangentPipeline, "solve", fail_second)
    result = solver.solve_nonlinear_homogenization(
        problem(matrix=dict(friction=0.3, dilatancy=dilatancy), globalization="newton")
    )
    assert len(result["rejected_steps"]) == 1
    assert max(r["residual"] for r in result["history"]) <= 1e-5
    assert result["history"][-1]["target_strain"] == 0.002


@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_unsupported_fft_grid_uses_gpu_coordinate_fallback(dilatancy):
    p = problem(matrix=dict(friction=0.3, dilatancy=dilatancy))
    # A prime factor above 13 on the last axis is outside the fused pipeline.
    p["phase_array"] = np.concatenate([p["phase_array"]] * 2, axis=2)[:5, :, :17].copy()
    p["ori_array"] = np.concatenate([p["ori_array"]] * 2, axis=2)[:5, :, :17].copy()
    p["path"] = [0.0, 0.002]
    result = solver.solve_nonlinear_homogenization(p)
    assert result["backend"] == "cupy" and result["coordinate_pcg"] == "fused"
    assert not result["rejected_steps"]
    assert max(r["residual"] for r in result["history"]) <= 1e-5


def test_associated_graphs_and_host_loop_preserve_curve(monkeypatch):
    p = problem()
    p["path"] = [0.0, 0.003]
    graphs = solver.solve_nonlinear_homogenization(p)
    monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", "0")
    host = solver.solve_nonlinear_homogenization(p)
    for key in ("strain", "stress"):
        np.testing.assert_allclose(
            host["fields"][key], graphs["fields"][key], atol=1e-7, rtol=1e-5
        )
    assert max(r["residual"] for r in host["history"]) <= 1e-5
