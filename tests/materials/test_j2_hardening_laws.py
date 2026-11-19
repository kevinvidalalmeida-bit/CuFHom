"""J2 hardening laws of the FANS small-strain models: saturation (Voce),
Prager kinematic hardening, Perzyna viscosity and perfect plasticity.

Homogeneous uniaxial-stress paths have closed forms: with X = 2/3 C dev(ep)
the relative stress is q = |sigma - C ep_11|, so monotone tension gives
sigma = R(p) + C p and a reversal yields again at sigma = C ep_11 - R(p).
"""

import numpy as np
import pytest
from cufhom.fft.nonlinear.solver import solve_nonlinear_homogenization
from cufhom.materials import plasticity as j2
from cufhom.materials.drucker_prager import DruckerPragerMaterial
from scipy.optimize import brentq

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
except Exception:
    cp = None

GENERAL = dict(
    E=3.0,
    nu=0.3,
    yield_stress=0.02,
    hardening=0.05,
    saturation_stress=0.04,
    saturation_exponent=200.0,
    kinematic_hardening=0.3,
)


def _homogeneous(n=4):
    return np.zeros((n, n, n), np.int8), np.zeros((n, n, n, 3))


def _solve(material, backend="scipy", **kwargs):
    phase, ori = _homogeneous()
    return solve_nonlinear_homogenization(
        dict(
            phase_array=phase, ori_array=ori, backend=backend, matrix=material, **kwargs
        )
    )


def _random_state(rng, n, scale=4e-3):
    strain = rng.normal(0, scale, (6, n))
    ep = rng.normal(0, scale / 4, (6, n))
    ep[:3] -= ep[:3].mean(axis=0)
    return strain, j2.PlasticState(ep, np.abs(rng.normal(0, scale / 2, n)))


def test_parameters_and_legacy_configuration():
    assert j2.J2Material().configuration() == dict(
        E=3.0, nu=0.3, yield_stress=0.015, hardening=0.03
    )
    assert j2.J2Material().model_name == "small_strain_J2_linear_isotropic_hardening"
    assert (
        j2.J2Material(hardening=0.0).model_name
        == "small_strain_J2_linear_isotropic_hardening"
    )
    material = j2.J2Material(**GENERAL, viscosity=1e-3)
    assert (
        material.model_name
        == "small_strain_J2_Voce_isotropic_Prager_kinematic_Perzyna_viscoplastic"
    )
    assert material.configuration()["kinematic_hardening"] == 0.3
    p = np.array([0.0, 0.01, 1.0])
    assert np.allclose(
        material.radius(p), 0.02 + 0.05 * p + 0.02 * (1 - np.exp(-200 * p))
    )
    for bad in (
        dict(saturation_stress=0.01),
        dict(saturation_stress=0.04),
        dict(saturation_exponent=1.0),
        dict(kinematic_hardening=-1.0),
        dict(viscosity=-1.0),
        dict(hardening=-0.1),
    ):
        with pytest.raises(ValueError):
            j2.J2Material(yield_stress=0.02, **bad)
    with pytest.raises(ValueError):
        DruckerPragerMaterial(kinematic_hardening=0.1)
    with pytest.raises(ValueError):
        DruckerPragerMaterial(hardening=0.0)
    with pytest.raises(ValueError):
        material.viscous_modulus(0.0)


@pytest.mark.parametrize("extra", [{}, dict(viscosity=2e-3)])
def test_backward_euler_tangent_is_consistent(extra):
    rng = np.random.default_rng(3)
    material = j2.J2Material(**GENERAL, **extra)
    viscous = material.viscous_modulus(0.5) if extra else 0.0
    strain, state = _random_state(rng, 64)
    ids, table = np.zeros(64, np.int32), np.eye(6)[None]
    response = j2.update_cpu(strain, state, material, ids, table, viscous=viscous)
    assert np.count_nonzero(response.dp) > 20
    h = 1e-8
    for component in range(6):
        v = np.zeros_like(strain)
        v[component] = 1
        plus = j2.update_cpu(
            strain + h * v, state, material, ids, table, viscous=viscous
        ).stress
        minus = j2.update_cpu(
            strain - h * v, state, material, ids, table, viscous=viscous
        ).stress
        exact = j2.tangent_cpu(v, response, material, ids, table)
        assert np.allclose(exact, (plus - minus) / (2 * h), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("start", [False, True])
def test_return_satisfies_yield_and_flow(start):
    rng = np.random.default_rng(5)
    material = j2.J2Material(**GENERAL, viscosity=1e-3)
    viscous = material.viscous_modulus(0.2)
    strain, state = _random_state(rng, 200)
    begin = strain - rng.normal(0, 1e-3, strain.shape) if start else None
    ids, table = np.zeros(200, np.int32), np.eye(6)[None]
    r = j2.update_cpu(strain, state, material, ids, table, begin, viscous=viscous)
    plastic = r.dp > 0
    assert plastic.sum() > 50
    committed = j2.commit_state(state, r)
    xi = r.stress.copy()
    xi[:3] -= xi[:3].mean(axis=0)
    xi -= material.backstress(committed.plastic_strain)
    q = np.sqrt(1.5 * np.sum(xi * xi, axis=0))
    overstress = q - material.radius(committed.equivalent) - viscous * r.dp
    assert np.abs(overstress[plastic]).max() < 1e-12
    assert overstress[~plastic].max() <= 1e-12
    # Elastic law with the committed plastic strain reproduces the stress.
    elastic = j2.tangent_cpu(
        strain - committed.plastic_strain,
        j2.MaterialResponse(
            None, r.direction * 0, np.full(200, 2 * material.mu), np.zeros(200), None
        ),
        material,
        ids,
        table,
    )
    assert np.allclose(elastic, r.stress, atol=1e-14)


@pytest.mark.parametrize("integrator", ["backward_euler", "midpoint"])
def test_uniaxial_saturation_and_kinematic_closed_form(integrator):
    material = j2.J2Material(**GENERAL)
    result = _solve(
        GENERAL, path=[0, 0.02], integrator=integrator, initial_increment=0.002
    )
    final = result["history"][-1]
    eps = final["strain"][0]
    p = brentq(
        lambda p: (
            material.radius(p)
            + material.kinematic_hardening * p
            - material.E * (eps - p)
        ),
        0,
        eps,
    )
    assert final["stress"][0] == pytest.approx(material.E * (eps - p), rel=1e-10)
    assert max(abs(s) for s in final["stress"][1:]) < 1e-10


def test_bauschinger_reversal_and_perfect_plasticity():
    C, sy, E = 0.4, 0.02, 3.0
    result = _solve(
        dict(E=E, nu=0.3, yield_stress=sy, hardening=0.0, kinematic_hardening=C),
        path=[0, 0.01, -0.01],
        initial_increment=0.002,
    )
    eps = result["history"][-1]["strain"][0]
    plastic = (E * eps + sy) / (E + C)  # compression branch sigma = C ep - sy
    assert result["history"][-1]["stress"][0] == pytest.approx(
        C * plastic - sy, rel=1e-10
    )
    perfect = _solve(dict(E=E, nu=0.3, yield_stress=sy, hardening=0.0), path=[0, 0.01])
    assert perfect["history"][-1]["stress"][0] == pytest.approx(sy, rel=1e-12)


def test_viscous_uniaxial_matches_discrete_recursion():
    material = j2.J2Material(
        E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05, viscosity=5e-3
    )
    rate = 1e-3
    result = _solve(
        dict(E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05, viscosity=5e-3),
        path=[0, 0.01],
        load_rate=rate,
        integrator="backward_euler",
        initial_increment=0.001,
    )
    assert result["load_rate"] == rate and not result["history"][-1]["elastic_step"]
    ep = p = 0.0
    previous = 0.0
    for row in result["history"][1:]:
        eps = row["target_strain"]
        V = material.viscous_modulus((eps - previous) / rate)
        trial = material.E * (eps - ep)
        d = max(
            0.0, (trial - material.radius(p)) / (material.E + material.hardening + V)
        )
        ep, p, previous = ep + d, p + d, eps
        assert row["stress"][0] == pytest.approx(material.E * (eps - ep), rel=1e-10)
    slow = _solve(
        dict(E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05, viscosity=5e-3),
        path=[0, 0.01],
        load_rate=1e-9,
        integrator="backward_euler",
        initial_increment=0.001,
    )
    independent = _solve(
        dict(E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05),
        path=[0, 0.01],
        integrator="backward_euler",
        initial_increment=0.001,
        elastic_segments=False,
    )
    assert slow["history"][-1]["stress"][0] == pytest.approx(
        independent["history"][-1]["stress"][0], rel=1e-6
    )
    assert result["history"][-1]["stress"][0] > independent["history"][-1]["stress"][0]
    with pytest.raises(ValueError):
        _solve(
            dict(E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05, viscosity=5e-3),
            path=[0, 0.01],
        )
    with pytest.raises(ValueError):
        _solve(
            dict(E=3.0, nu=0.3, yield_stress=0.02, hardening=0.05),
            path=[0, 0.01],
            load_rate=1.0,
        )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
@pytest.mark.parametrize("start", [False, True])
def test_gpu_return_matches_cpu(start):
    rng = np.random.default_rng(11)
    material = j2.J2Material(**GENERAL, viscosity=1e-3)
    viscous = material.viscous_modulus(0.3)
    n = 4096
    strain, state = _random_state(rng, n)
    begin = strain - rng.normal(0, 1e-3, strain.shape) if start else None
    ids, table = np.zeros(n, np.int32), np.eye(6)[None]
    cpu = j2.update_cpu(strain, state, material, ids, table, begin, viscous=viscous)
    gpu = j2.CuPyMaterial(
        cp, material, cp.asarray(ids), cp.asarray(table), np.float64
    ).update(
        cp.asarray(strain),
        j2.PlasticState(cp.asarray(state.plastic_strain), cp.asarray(state.equivalent)),
        None if begin is None else cp.asarray(begin),
        viscous=viscous,
    )
    assert np.count_nonzero(cpu.dp) > n // 4
    for name in ("stress", "direction", "a", "b", "dp"):
        assert np.allclose(
            cp.asnumpy(getattr(gpu, name)), getattr(cpu, name), rtol=1e-9, atol=1e-13
        ), name


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_fused_diagnostics_match_reference_on_composite():
    rng = np.random.default_rng(2)
    phase = (rng.random((12, 12, 12)) < 0.2).astype(np.int8)
    ori = np.zeros((12, 12, 12, 3))
    ori[..., 2] = 1
    rows = []
    for mode in ("fused", "reference"):
        result = solve_nonlinear_homogenization(
            dict(
                phase_array=phase,
                ori_array=ori,
                backend="cupy",
                dtype="float64",
                path=[0, 0.01, 0.002],
                matrix=dict(GENERAL, viscosity=2e-3),
                load_rate=1e-2,
                diagnostic_reduction=mode,
            )
        )
        rows.append(result["history"])
    for fused, reference in zip(*rows):
        assert fused["stress"] == pytest.approx(
            reference["stress"], rel=1e-12, abs=1e-15
        )
        for key in (
            "yield_violation",
            "plastic_dissipation",
            "equivalent_plastic_mean",
            "plastic_fraction",
        ):
            assert fused[key] == pytest.approx(reference[key], rel=1e-9, abs=1e-13), key
