"""Fused trial diagnostics against independent FP64 field formulas."""

import numpy as np
import pytest
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom.fft.nonlinear.diagnostics import FusedDiagnostics
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization
from cufhom.materials.drucker_prager import DruckerPragerMaterial
from cufhom.materials.plasticity import J2Material, MaterialResponse, PlasticState


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp


def _fields(cp, dtype, shape, pressure):
    rng = np.random.default_rng(261023)
    n = np.prod(shape)
    ids = rng.integers(0, 2, size=shape, dtype=np.int32)
    strain = rng.normal(scale=0.005, size=(6, *shape)).astype(dtype)
    stress = rng.normal(scale=0.01, size=(6, *shape)).astype(dtype)
    old_alpha = rng.uniform(0, 0.02, shape).astype(dtype)
    delta = rng.uniform(0, 0.002, shape).astype(dtype)
    delta[ids != 0] = 0
    ep = rng.normal(scale=0.001, size=strain.shape).astype(dtype)
    direction = rng.normal(scale=0.1, size=strain.shape).astype(dtype)
    material = DruckerPragerMaterial() if pressure else J2Material()
    state = PlasticState(cp.asarray(ep), cp.asarray(old_alpha))
    response = MaterialResponse(
        cp.asarray(stress), cp.asarray(direction), None, None, cp.asarray(delta)
    )
    stress64, strain64 = stress.astype(float), strain.astype(float)
    mean = stress64[:3].sum(axis=0) / 3
    dev = stress64.copy()
    dev[:3] -= mean
    alpha64, delta64 = old_alpha.astype(float), delta.astype(float)
    f = (
        np.sqrt(1.5 * np.sum(dev * dev, axis=0))
        - material.yield_stress
        - material.hardening * (alpha64 + delta64)
    )
    if pressure:
        f += material.friction * mean
    work = (
        delta64 * (material.yield_stress + (material.beta - material.friction) * mean)
        if pressure
        else material.yield_stress * delta64
    )
    expected = dict(
        mean_strain=strain64.reshape(6, -1).mean(axis=1),
        mean_stress=stress64.reshape(6, -1).mean(axis=1),
        stress_work=np.sum(strain64 * stress64) / n,
        strain_norm=np.sqrt(np.sum(strain64 * strain64) / n),
        equivalent_mean=np.mean(alpha64 + delta64),
        plastic_fraction=np.mean(alpha64 + delta64 > 1e-10),
        yield_violation=np.max(np.where(ids == 0, np.maximum(f, 0.0), 0.0))
        / material.yield_stress,
        dissipation=np.mean(work),
        plastic_volumetric_mean=(
            np.mean(
                np.sum(ep[:3].astype(float), axis=0)
                + np.sqrt(1.5) * np.sum(direction[:3].astype(float), axis=0) * delta64
            )
            if pressure
            else 0.0
        ),
    )
    return ids, strain, material, state, response, expected


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("pressure", [False, True])
@pytest.mark.parametrize("shape", [(3, 5, 7), (8, 8, 8), (17, 19, 11)])
def test_fused_diagnostics_matches_independent_fp64(cuda, dtype, pressure, shape):
    ids, e, material, state, response, expected = _fields(cuda, dtype, shape, pressure)
    engine = FusedDiagnostics(
        cuda, material, cuda.asarray(ids), dtype, pressure=pressure
    )
    before = [
        cuda.asnumpy(x)
        for x in (state.plastic_strain, state.equivalent, response.stress, response.dp)
    ]
    actual = engine.evaluate(cuda.asarray(e), response, state)
    for name in expected:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=3e-13, atol=1e-17, err_msg=name
        )
    for field, saved in zip(
        (state.plastic_strain, state.equivalent, response.stress, response.dp), before
    ):
        np.testing.assert_array_equal(cuda.asnumpy(field), saved)
    assert engine.calls == 1
    assert engine.workspace_bytes <= 22 * (4096 + 1) * 8


@pytest.mark.gpu
@pytest.mark.parametrize(
    "field", ["strain", "stress", "dp", "alpha", "direction", "plastic_strain"]
)
def test_fused_diagnostics_rejects_local_nonfinite(cuda, field):
    ids, e, material, state, response, _ = _fields(cuda, "float32", (3, 5, 7), True)
    arrays = dict(
        strain=cuda.asarray(e),
        stress=response.stress,
        dp=response.dp,
        alpha=state.equivalent,
        direction=response.direction,
        plastic_strain=state.plastic_strain,
    )
    arrays[field].flat[0] = cuda.nan
    with pytest.raises(NonlinearConvergenceError, match="diagnostic"):
        FusedDiagnostics(
            cuda, material, cuda.asarray(ids), "float32", pressure=True
        ).evaluate(arrays["strain"], response, state)


@pytest.mark.gpu
def test_local_negative_dp_work_cannot_hide_in_positive_mean(cuda):
    shape = (2, 2, 2)
    ids = cuda.zeros(shape, cuda.int32)
    e = cuda.zeros((6, *shape), cuda.float64)
    m = DruckerPragerMaterial(friction=0.3, dilatancy=0.0)
    stress = cuda.zeros_like(e)
    stress[:3, 0, 0, 0] = 1.0
    dp = cuda.full(shape, 0.001)
    state = PlasticState(cuda.zeros_like(e), cuda.zeros(shape))
    response = MaterialResponse(stress, cuda.zeros_like(e), None, None, dp)
    with pytest.raises(NonlinearConvergenceError, match="Negative local"):
        FusedDiagnostics(cuda, m, ids, "float64", pressure=True).evaluate(
            e, response, state
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "model,extra", [("j2", {}), ("drucker_prager", {"friction": 0.3, "dilatancy": 0.1})]
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_complete_curve_fused_diagnostics_matches_reference(cuda, model, extra, dtype):
    shape = (6, 6, 8)
    phase = np.zeros(shape, np.uint8)
    phase[2:4, 2:4, :] = 1
    ori = np.zeros((*shape, 3))
    ori[..., 2] = 1
    p = dict(
        phase_array=phase,
        ori_array=ori,
        model=model,
        matrix=extra,
        backend="cupy",
        dtype=dtype,
        path=[0.0, 0.006, 0.004],
        initial_increment=0.001,
        rtol=1e-5 if dtype == "float32" else 1e-8,
        return_fields=True,
    )
    old = solve_nonlinear_homogenization(dict(p, diagnostic_reduction="reference"))
    new = solve_nonlinear_homogenization(dict(p, diagnostic_reduction="fused"))
    assert new["diagnostic_reduction"] == "fused"
    assert len(old["history"]) == len(new["history"])
    assert new["diagnostic_calls"] == len(new["history"])
    assert old["rejected_steps"] == new["rejected_steps"]
    for name in ["stress", "strain", "residual", "newton_iterations", "cg_iterations"]:
        np.testing.assert_allclose(
            [r[name] for r in old["history"]],
            [r[name] for r in new["history"]],
            rtol=2e-12,
            atol=1e-15,
        )
    np.testing.assert_array_equal(
        old["state"].plastic_strain, new["state"].plastic_strain
    )
    np.testing.assert_array_equal(old["state"].equivalent, new["state"].equivalent)
    np.testing.assert_array_equal(old["fields"]["strain"], new["fields"]["strain"])


def test_explicit_fused_diagnostics_rejects_cpu():
    with pytest.raises(ValueError, match="Fused diagnostics require"):
        solve_nonlinear_homogenization(
            dict(
                phase_array=np.zeros((2, 2, 2), int),
                ori_array=np.zeros((2, 2, 2, 3)),
                diagnostic_reduction="fused",
            )
        )
