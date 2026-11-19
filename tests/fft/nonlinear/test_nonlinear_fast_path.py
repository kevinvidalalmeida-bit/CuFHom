"""The retained nonlinear FFT fast path preserves the constrained problem."""

import numpy as np
import pytest
from cufhom.fft.nonlinear.solver import (
    MixedProjector,
    cp,
    solve_nonlinear_homogenization,
)
from cufhom.fft.operators import DisplacementCoordinates, IsotropicReference
from cufhom_core.tensors import fft as core_fft

# Outer acceptance bounds the residual (rtol*yield stress), not the state: soft
# plastic modes leave ~1e-7 differences in the equivalent plastic strain between
# equally converged Newton trajectories of this 6^3 case.
STATE_ATOL = 5e-7


def _case():
    phase = np.zeros((6, 6, 6), dtype=np.uint8)
    phase[2:4, 2:4, :] = 1
    orientation = np.zeros(phase.shape + (3,))
    orientation[..., 2] = 1
    return dict(
        phase_array=phase,
        ori_array=orientation,
        path=[0.0, 0.006, 0.004],
        initial_increment=0.001,
        return_fields=True,
    )


@pytest.mark.parametrize("shape", [(5, 7, 5), (6, 4, 8)])
def test_displacement_coordinates_are_an_isometry(shape):
    core_fft.set_fft_backend("scipy")
    control = np.array([True, False, False, False, False, False])
    projector = MixedProjector(shape, (1.0, 2.0, 3.0), control)
    coordinates = DisplacementCoordinates(projector)
    a, b = np.random.default_rng(4).normal(size=(2, 6) + shape)
    z, w = coordinates.adjoint(a), coordinates.adjoint(b)
    np.testing.assert_allclose(coordinates.strain(z), projector(a), atol=2e-12)
    np.testing.assert_allclose(
        coordinates.adjoint(coordinates.strain(z)), z, atol=2e-12
    )
    np.testing.assert_allclose(
        np.vdot(z, w).real, np.sum(projector(a) * projector(b)), atol=2e-12
    )


@pytest.mark.parametrize(
    "backend", ["scipy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_symmetric_coordinates_with_free_means_are_an_adjoint_pair(backend):
    """Q R and R Q* with the Cholesky root on the free means (mixed loading)."""
    from cufhom.fft.linear.krylov import SymmetricCoordinates

    if backend == "cupy" and cp is None:
        pytest.skip("CuPy unavailable")
    xp = np if backend == "scipy" else cp
    core_fft.set_fft_backend(backend)
    shape = (6, 5, 8)
    projector = MixedProjector(
        shape, (1.0, 2.0, 3.0), [False, True, False, False, False, False], xp
    )
    Q = SymmetricCoordinates(projector, 2.5, 1.2)
    rng = np.random.default_rng(6)
    field = xp.asarray(rng.normal(size=(6,) + shape))
    z = Q.adjoint(xp.asarray(rng.normal(size=(6,) + shape)))
    lhs = float(xp.sum(Q.strain(z) * field))
    rhs = float(xp.vdot(z, Q.adjoint(field)).real)
    assert abs(lhs - rhs) <= 1e-10 * abs(lhs)
    # The mean strain of Q R z is L m on the free components, zero on the prescribed one.
    means = xp.mean(Q.strain(z), axis=(1, 2, 3))
    expected = np.zeros(6)
    expected[projector.free] = Q.mean_root[projector.free, :5] @ np.asarray(
        z[3 * Q.count :].real.tolist()
    )
    np.testing.assert_allclose(
        np.asarray(means.tolist()), expected * Q.norm, atol=1e-12
    )


@pytest.mark.parametrize("shift", [0.0, 3.0])
def test_matrix_reference_is_exact_for_its_homogeneous_operator(shift):
    core_fft.set_fft_backend("scipy")
    projector = MixedProjector(
        (5, 7, 5), (1.0, 2.0, 3.0), [True, False, False, False, False, False]
    )
    reference = IsotropicReference(projector, 2.5, 1.2)
    reference.set_shift(shift, np.dtype("float64"))
    stiffness = np.eye(6) * 2 * 1.2
    stiffness[:3, :3] += 2.5 - 2 * 1.2 / 3
    stiffness *= 1 + shift  # PTC: P C P + shift K0
    field = projector(np.random.default_rng(5).normal(size=(6, 5, 7, 5)))
    rhs = projector(np.einsum("ij,j...->i...", stiffness, field))
    np.testing.assert_allclose(reference(rhs), field, atol=1e-12)


def test_default_fast_path_matches_strain_reference():
    fast = solve_nonlinear_homogenization(_case())
    reference = solve_nonlinear_homogenization(
        dict(_case(), formulation="strain", preconditioner="none")
    )
    assert fast["formulation"] == "displacement"
    assert fast["preconditioner"] == "matrix"
    np.testing.assert_allclose(
        [row["stress"] for row in fast["history"]],
        [row["stress"] for row in reference["history"]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        fast["fields"]["strain"], reference["fields"]["strain"], atol=1e-9
    )
    np.testing.assert_allclose(
        fast["state"].equivalent, reference["state"].equivalent, atol=1e-9
    )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_gpu_fast_path_matches_cpu():
    cpu = solve_nonlinear_homogenization(_case())
    gpu = solve_nonlinear_homogenization(dict(_case(), backend="cupy"))
    assert gpu["coordinate_pcg"] == "pipeline" and gpu["reuse_trial_response"]
    assert gpu["inner_forcing"] == "ew_target"
    assert cpu["coordinate_pcg"] == "classic" and cpu["inner_forcing"] == "ew"
    np.testing.assert_allclose(
        [row["stress"] for row in gpu["history"]],
        [row["stress"] for row in cpu["history"]],
        atol=2e-6,
    )
    assert max(row["residual"] for row in gpu["history"]) <= gpu["rtol"]


@pytest.mark.parametrize("globalization", ["newton", "ptc"])
def test_reuse_accepted_equilibrium_is_exact_and_saves_evaluations(
    globalization, monkeypatch
):
    import cufhom.fft.nonlinear.solver as solver

    original = solver.update_cpu
    calls = []

    def counted(e, state, *args):
        before = state.plastic_strain.copy(), state.equivalent.copy()
        answer = original(e, state, *args)
        np.testing.assert_array_equal(state.plastic_strain, before[0])
        np.testing.assert_array_equal(state.equivalent, before[1])
        calls.append(1)
        return answer

    monkeypatch.setattr(solver, "update_cpu", counted)
    p = dict(_case(), globalization=globalization)
    old = solve_nonlinear_homogenization(dict(p, reuse_trial_response=False))
    old_calls = len(calls)
    calls.clear()
    new = solve_nonlinear_homogenization(dict(p, reuse_trial_response=True))
    assert old_calls - len(calls) == sum(r["newton_iterations"] for r in new["history"])
    for name in ("stress", "strain", "residual", "cg_iterations"):
        np.testing.assert_array_equal(
            [r[name] for r in new["history"]], [r[name] for r in old["history"]]
        )
    np.testing.assert_array_equal(
        new["state"].plastic_strain, old["state"].plastic_strain
    )
    np.testing.assert_array_equal(new["state"].equivalent, old["state"].equivalent)


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
@pytest.mark.parametrize("engine", ["fused", "pipeline"])
@pytest.mark.parametrize("globalization", ["newton", "ptc"])
def test_fused_coordinate_curve_and_restart(globalization, engine, tmp_path):
    p = dict(_case(), backend="cupy", globalization=globalization)
    old = solve_nonlinear_homogenization(
        dict(p, coordinate_pcg="classic", reuse_trial_response=False)
    )
    new = solve_nonlinear_homogenization(
        dict(p, coordinate_pcg=engine, reuse_trial_response=True)
    )
    assert new["coordinate_pcg"] == engine and not new["rejected_steps"]
    for name in ("stress", "strain"):
        np.testing.assert_allclose(
            [r[name] for r in new["history"]],
            [r[name] for r in old["history"]],
            atol=1e-7,
            rtol=1e-5,
        )
    np.testing.assert_allclose(
        new["state"].equivalent, old["state"].equivalent, atol=STATE_ATOL
    )
    checkpoint = tmp_path / "confirmed.npz"
    options = dict(p, coordinate_pcg=engine, reuse_trial_response=True)
    solve_nonlinear_homogenization(
        dict(options, path=[0.0, 0.006], checkpoint_path=checkpoint)
    )
    restarted = solve_nonlinear_homogenization(
        dict(options, path=[0.006, 0.004], restart_from=checkpoint)
    )
    np.testing.assert_allclose(
        restarted["state"].equivalent, new["state"].equivalent, atol=STATE_ATOL
    )
    assert max(r["residual"] for r in new["history"]) <= new["rtol"]


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
@pytest.mark.parametrize("shape", [(8, 6, 10), (5, 4, 6)])
def test_tangent_pipeline_matches_production_operator(shape):
    """R Q* C_alg Q R of the fused kernels, free mean strains and fibers included."""
    from cufhom.fft.nonlinear.krylov import TangentPipeline
    from cufhom.fft.nonlinear.solver import _table
    from cufhom.materials.plasticity import CuPyMaterial, J2Material, initial_state

    core_fft.set_fft_backend("cupy")
    rng = np.random.default_rng(7)
    phase = (rng.random(shape) < 0.3).astype(np.int32)
    ori = rng.normal(size=shape + (3,))
    material = J2Material()
    ids, table = _table(
        phase, ori, material, dict(E_L=70.0, E_T=8.0, nu_LT=0.25, nu_TT=0.3, G_LT=5.0)
    )
    ids, table = cp.asarray(ids), cp.asarray(table, dtype=cp.float32)
    projector = MixedProjector(
        shape, (1.0, 2.0, 3.0), [True, False, False, False, False, False], cp
    )
    solver = TangentPipeline(projector, material, ids, table)
    Q, pipeline = solver.Q, solver.pipeline
    gpu = CuPyMaterial(cp, material, ids, table, np.float32)
    strain = cp.asarray(rng.normal(size=(6,) + shape) * 0.004, dtype=cp.float32)
    response = gpu.update(strain, initial_state(shape, cp, np.float32))
    assert 0 < float(cp.mean(response.b != 0)) < 1  # elastic and plastic matrix voxels
    pipeline.bind(response.a, response.b, response.direction)
    z = Q.adjoint(cp.asarray(rng.normal(size=(6,) + shape), dtype=cp.float32))
    assert float(cp.abs(z[3 * Q.count :]).min()) > 0  # free means are exercised
    fused = pipeline.to_fused(z)
    # Even grids pack the k2 Nyquist plane into k2 = 0: exact up to rounding.
    cp.testing.assert_allclose(
        pipeline.to_production(fused), z, rtol=0, atol=1e-6 * float(cp.abs(z).max())
    )
    expected = Q.adjoint(gpu.tangent(Q.strain(z), response))
    actual = pipeline.to_production(pipeline.matvec(fused))
    assert float(cp.linalg.norm(actual - expected) / cp.linalg.norm(expected)) < 2e-6
    energy = float(pipeline.apply(fused))
    assert abs(energy - float(cp.vdot(z, expected).real)) <= 2e-6 * abs(energy)


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_tangent_pipeline_host_loop_matches_graphs(monkeypatch):
    p = dict(_case(), backend="cupy", coordinate_pcg="pipeline")
    graphs = solve_nonlinear_homogenization(p)
    monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", "0")
    host = solve_nonlinear_homogenization(p)
    np.testing.assert_allclose(
        [r["stress"] for r in host["history"]],
        [r["stress"] for r in graphs["history"]],
        atol=1e-7,
    )
    assert max(r["residual"] for r in host["history"]) <= host["rtol"]


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_tangent_pipeline_policy():
    odd = dict(_case(), backend="cupy")
    odd["phase_array"] = odd["phase_array"][:, :, :5]
    odd["ori_array"] = odd["ori_array"][:, :, :5]
    assert solve_nonlinear_homogenization(odd)["coordinate_pcg"] == "pipeline"
    prime = dict(_case(), backend="cupy")
    prime["phase_array"] = np.concatenate([prime["phase_array"]] * 3, axis=2)[:, :, :17]
    prime["ori_array"] = np.concatenate([prime["ori_array"]] * 3, axis=2)[:, :, :17]
    assert solve_nonlinear_homogenization(prime)["coordinate_pcg"] == "fused"
    with pytest.raises(ValueError, match="primes <= 13"):
        solve_nonlinear_homogenization(dict(prime, coordinate_pcg="pipeline"))


def test_fused_coordinate_policy_cpu_and_invalid_options():
    result = solve_nonlinear_homogenization(dict(_case(), coordinate_pcg="auto"))
    assert result["coordinate_pcg"] == "classic"
    with pytest.raises(ValueError, match="requires CuPy"):
        solve_nonlinear_homogenization(dict(_case(), coordinate_pcg="fused"))
    with pytest.raises(ValueError, match="coordinate_pcg must"):
        solve_nonlinear_homogenization(dict(_case(), coordinate_pcg="typo"))


def test_cached_response_does_not_survive_rejected_attempt(monkeypatch):
    import cufhom.fft.nonlinear.solver as solver

    original = solver._cg
    counter = [0]

    def fail_second(*args, **kwargs):
        counter[0] += 1
        if counter[0] == 2:
            raise solver.NonlinearConvergenceError(
                "injected after an accepted Newton update"
            )
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_cg", fail_second)
    p = dict(_case(), globalization="newton")
    old = solve_nonlinear_homogenization(dict(p, reuse_trial_response=False))
    counter[0] = 0
    new = solve_nonlinear_homogenization(dict(p, reuse_trial_response=True))
    assert len(new["rejected_steps"]) == len(old["rejected_steps"]) == 1
    np.testing.assert_array_equal(new["fields"]["strain"], old["fields"]["strain"])
    np.testing.assert_array_equal(new["state"].equivalent, old["state"].equivalent)


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_fused_failed_increment_retains_confirmed_checkpoint(tmp_path):
    from cufhom.fft.nonlinear.solver import NonlinearConvergenceError

    checkpoint = tmp_path / "failed.npz"
    with pytest.raises(NonlinearConvergenceError):
        solve_nonlinear_homogenization(
            dict(
                _case(),
                backend="cupy",
                coordinate_pcg="fused",
                reuse_trial_response=True,
                max_cg=1,
                globalization="newton",
                min_increment=0.001,
                checkpoint_path=checkpoint,
            )
        )
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert not np.any(saved["strain"]) and not np.any(saved["equivalent"])
        assert not np.any(saved["plastic_strain"])


@pytest.mark.parametrize(
    "backend", ["scipy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
@pytest.mark.parametrize("globalization", ["newton", "ptc"])
def test_target_forcing_preserves_outer_acceptance_and_state(backend, globalization):
    if backend == "cupy" and cp is None:
        pytest.skip("CuPy unavailable")
    p = dict(
        _case(),
        backend=backend,
        globalization=globalization,
        coordinate_pcg="auto",
        reuse_trial_response=True,
    )
    old = solve_nonlinear_homogenization(dict(p, inner_forcing="ew"))
    new = solve_nonlinear_homogenization(dict(p, inner_forcing="ew_target"))
    assert not new["rejected_steps"] and len(new["history"]) == len(old["history"])
    error = np.max(
        np.abs(
            np.array([r["stress"] for r in new["history"]])
            - np.array([r["stress"] for r in old["history"]])
        )
    )
    assert error / 0.015 <= 0.001
    assert max(r["residual"] for r in new["history"]) <= new["rtol"]
    np.testing.assert_allclose(
        new["state"].equivalent, old["state"].equivalent, atol=STATE_ATOL
    )
    if globalization == "ptc":
        np.testing.assert_array_equal(new["fields"]["strain"], old["fields"]["strain"])


def test_target_forcing_does_not_relax_final_acceptance():
    from cufhom.fft.nonlinear.solver import _ew_tolerance

    # Near the goal, avoid solving to 0.008 when 0.3 is already sufficient.
    assert _ew_tolerance(1.5e-5, 3.5e-4, 0.04, 1e-5, target=1e-5) == 0.3
    assert _ew_tolerance(0.1, None, None, 1e-5, target=1e-5) == 0.1
    for error in np.geomspace(1.001e-5, 1.0, 100):
        eta = _ew_tolerance(error, 2 * error, 0.1, 1e-5, target=1e-5)
        assert 1e-5 <= eta <= 0.3
