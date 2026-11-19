"""Paraboloidal (Tschoegl/Melro) matrix: closed-form uniaxial tests, exact
tangent, CPU/GPU parity and solver contracts.

In homogeneous uniaxial stress f = (sigma - sigma_t)(sigma + sigma_c), the
flow direction is fixed and nu_p is the plastic Poisson ratio, so backward
Euler reproduces sigma = sigma_t(ep_11), eps_22 = -nu sigma/E - nu_p ep_11.
"""

import numpy as np
import pytest
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization
from cufhom.materials import paraboloidal as pb
from cufhom.materials.plasticity import PlasticState
from scipy.optimize import brentq

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
except Exception:
    cp = None

MATRIX = dict(
    E=3.76,
    nu=0.39,
    plastic_poisson=0.32,
    tension_curve=[[0, 0.029], [0.02, 0.05], [0.1, 0.07]],
    compression_curve=[[0, 0.067], [0.03, 0.1], [0.2, 0.12]],
)


def _random(n, seed=1):
    rng = np.random.default_rng(seed)
    state = PlasticState(rng.normal(0, 3e-3, (6, n)), np.abs(rng.normal(0, 0.02, n)))
    return rng.normal(0, 2e-2, (6, n)), state, np.zeros(n, np.int32), np.eye(6)[None]


def test_parameters():
    material = pb.ParaboloidalMaterial(**MATRIX)
    assert material.yield_stress == 0.029 and material.tension_curve[1] == (0.02, 0.05)
    assert material.alpha == pytest.approx(4.5 * (1 - 0.64) / 1.32)
    assert material.configuration()["compression_curve"][2] == [0.2, 0.12]
    st, dst, sc, dsc = material.strengths(np.array([0.0, 0.01, 0.5]))
    assert np.allclose(st, [0.029, 0.0395, 0.07]) and np.allclose(dst, [1.05, 1.05, 0])
    for bad in (
        dict(plastic_poisson=0.6),
        dict(tension_curve=[[0.1, 0.02]]),
        dict(compression_curve=[[0, 0.06], [0.1, 0.05]]),
        dict(tension_curve=[[0, -0.01]]),
    ):
        with pytest.raises(ValueError):
            pb.ParaboloidalMaterial(**dict(MATRIX, **bad))


@pytest.mark.parametrize("sign", [1, -1])
def test_uniaxial_closed_form(sign):
    n = 4
    material = pb.ParaboloidalMaterial(**MATRIX)
    result = solve_nonlinear_homogenization(
        dict(
            phase_array=np.zeros((n, n, n), np.int8),
            ori_array=np.zeros((n, n, n, 3)),
            backend="scipy",
            model="paraboloidal",
            matrix=MATRIX,
            path=[0, sign * 0.05],
            initial_increment=0.005,
        )
    )
    assert (
        result["linear_solver"] == "bicgstab"
        and result["integrator"] == "backward_euler"
    )
    for row in result["history"][1:]:
        eps = row["strain"][0]
        curve = material.tension_curve if sign > 0 else material.compression_curve

        def strength(p):
            return pb._interpolate(curve, np.array([p]), np)[0][0]

        plastic = (
            brentq(lambda p: material.E * (abs(eps) - p) - strength(p), 0, abs(eps))
            if material.E * abs(eps) > strength(0.0)
            else 0.0
        )
        stress = sign * material.E * (abs(eps) - plastic)
        assert row["stress"][0] == pytest.approx(stress, rel=1e-9)
        assert row["strain"][1] == pytest.approx(
            -material.nu * stress / material.E
            - sign * material.plastic_poisson * plastic,
            rel=1e-9,
            abs=1e-14,
        )
        assert row["equivalent_plastic_mean"] == pytest.approx(
            plastic, rel=1e-9, abs=1e-14
        )
        assert row["yield_violation"] < 1e-12


def test_exact_tangent_and_flow():
    strain, state, ids, table = _random(2000)
    material = pb.ParaboloidalMaterial(**MATRIX)
    r = pb.update_cpu(strain, state, material, ids, table)
    plastic = r.dp > 0
    assert plastic.sum() > 1000
    violation, work = pb.audit_cpu(r, state, material, ids)
    assert violation.max() < 1e-14 and work[plastic].min() > 0
    increment = pb.commit_cpu(state, r, material).plastic_strain - state.plastic_strain
    assert np.allclose(
        np.sqrt(np.sum(increment**2, axis=0)) * material.equivalence, r.dp, atol=1e-15
    )
    h = 1e-8
    for component in range(6):
        v = np.zeros_like(strain)
        v[component] = 1
        plus = pb.update_cpu(strain + h * v, state, material, ids, table).stress
        minus = pb.update_cpu(strain - h * v, state, material, ids, table).stress
        assert np.allclose(
            pb.tangent_cpu(v, r, material, ids, table),
            (plus - minus) / (2 * h),
            rtol=1e-5,
            atol=1e-6,
        )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_gpu_matches_cpu():
    strain, state, ids, table = _random(4096, seed=7)
    material = pb.ParaboloidalMaterial(**MATRIX)
    cpu = pb.update_cpu(strain, state, material, ids, table)
    gpu_law = pb.CuPyParaboloidal(
        cp, material, cp.asarray(ids), cp.asarray(table), np.float64
    )
    gstate = PlasticState(
        cp.asarray(state.plastic_strain), cp.asarray(state.equivalent)
    )
    gpu = gpu_law.update(cp.asarray(strain), gstate)
    for name in (
        "stress",
        "direction",
        "a",
        "b",
        "dp",
        "bulk",
        "coupling_to_direction",
        "coupling_to_pressure",
    ):
        assert np.allclose(
            cp.asnumpy(getattr(gpu, name)), getattr(cpu, name), rtol=1e-9, atol=1e-12
        ), name
    assert np.allclose(
        cp.asnumpy(gpu_law.commit(gstate, gpu).plastic_strain),
        pb.commit_cpu(state, cpu, material).plastic_strain,
        atol=1e-15,
    )
    violation, work, volume = gpu_law.audit(gpu, gstate)
    assert np.allclose(
        cp.asnumpy(work), pb.audit_cpu(cpu, state, material, ids)[1], atol=1e-15
    )
    assert float(cp.max(violation)) < 1e-14
    v = np.random.default_rng(0).normal(size=strain.shape)
    assert np.allclose(
        cp.asnumpy(gpu_law.tangent(cp.asarray(v), gpu)),
        pb.tangent_cpu(v, cpu, material, ids, table),
    )


def test_solver_contracts():
    n = 4
    base = dict(
        phase_array=np.zeros((n, n, n), np.int8),
        ori_array=np.zeros((n, n, n, 3)),
        model="paraboloidal",
        matrix=MATRIX,
        path=[0, 0.01],
    )
    with pytest.raises(ValueError):
        solve_nonlinear_homogenization(dict(base, integrator="midpoint"))
    with pytest.raises(ValueError):
        solve_nonlinear_homogenization(dict(base, damage=dict()))
    with pytest.raises(ValueError):
        solve_nonlinear_homogenization(dict(base, load_rate=1.0))
