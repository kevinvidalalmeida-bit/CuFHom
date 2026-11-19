"""CPU constitutive review: independent invariants, FD tangents and frozen state.

Cancellation regressions (all parameters satisfy the existing material bounds):
* E=3, nu=.3, sy=H=1e-18, e_Mandel[3]=.1, alpha=beta=0:
  DP formerly selected an impossible apex, returned dp=-1, and decreased k.
* nu=.499999, hydrostatic e_ii=.01 in float32: DP apex pressure formerly
  differed from (sy+H*(k+dp))/alpha by about 1.7 percent.
* J2 at sy=H=1e-18 lost its positive returned radius by cancellation.

The fixes only rearrange the original backward-Euler equations. Write
R=sy+H*k, Da=alpha*K*beta+H, D=3*mu+Da. Then
q_new=(Da*q+3*mu*(R-alpha*p_trial))/D on the smooth cone;
apex iff alpha*p_trial-R >= q*Da/(3*mu); p_apex=(R+H*dp)/alpha.
J2 has q_new=R+H*dp. Smooth tangent coefficients are b=2*mu*(Da/D-t)
and bulk=K*(3*mu+H)/D, with t=q_new/q. No tolerances or laws change.

No CuPy object, kernel, solver or FFT is instantiated by these tests.
"""

from dataclasses import fields

import numpy as np
import pytest
from cufhom.materials import drucker_prager as dp
from cufhom.materials import plasticity as j2

MANDEL_IDENTITY = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _data(strain):
    return np.zeros(strain.shape[1:], dtype=np.int32), np.eye(6)[None]


def _deviator(stress):
    result = stress.copy()
    result[:3] -= result[:3].mean(axis=0)
    return result


def _q(stress):
    return np.sqrt(1.5 * np.sum(_deviator(stress) ** 2, axis=0))


def _matrix(law, strain, state, material, response, start=None):
    ids, table = _data(strain)
    tangent, finite = [], []
    h = 1e-7
    for component in range(6):
        value = np.zeros_like(strain)
        value[component] = 1
        tangent.append(law.tangent_cpu(value, response, material, ids, table)[:, 0])
        plus = law.update_cpu(
            strain + h * value, state, material, ids, table, start
        ).stress
        minus = law.update_cpu(
            strain - h * value, state, material, ids, table, start
        ).stress
        finite.append(((plus - minus) / (2 * h))[:, 0])
    return np.stack(tangent, axis=1), np.stack(finite, axis=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-6, 1e-12, 1e-18])
def test_small_positive_radius_preserves_j2_return_and_dp_reduction(dtype, scale):
    strain = np.zeros((6, 1), dtype=dtype)
    strain[3] = 0.1
    state = j2.initial_state((1,), dtype=dtype)
    ids, table = _data(strain)
    material = j2.J2Material(yield_stress=scale, hardening=scale)
    cone = dp.DruckerPragerMaterial(friction=0.0, yield_stress=scale, hardening=scale)
    radial = j2.update_cpu(strain, state, material, ids, table)
    response = dp.update_cpu(strain, state, cone, ids, table)
    for result in (radial, response):
        assert np.all(result.dp > 0)
        assert np.all(result.a > 0)
        committed = j2.commit_state(state, result)
        assert np.all(committed.equivalent > state.equivalent)
        radius = material.yield_stress + material.hardening * committed.equivalent
        # Relative tolerance measures the tiny physical radius, with no
        # absolute tolerance that would let an incorrectly zero stress pass.
        np.testing.assert_allclose(_q(result.stress), radius, rtol=3e-7, atol=0.0)
    for name in ("stress", "dp", "a", "b"):
        np.testing.assert_allclose(
            getattr(response, name), getattr(radial, name), rtol=3e-7, atol=0.0
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nu", [0.3, 0.499999, 0.499999999999])
def test_apex_pressure_satisfies_yield_near_incompressibility(dtype, nu):
    material = dp.DruckerPragerMaterial(nu=nu, dilatancy=0.1)
    strain = np.zeros((6, 1), dtype=dtype)
    strain[:3] = 0.01
    state = j2.initial_state((1,), dtype=dtype)
    ids, table = _data(strain)
    response = dp.update_cpu(strain, state, material, ids, table)
    assert response.dp[0] > 0
    np.testing.assert_array_equal(_deviator(response.stress), 0.0)
    radius = material.yield_stress + material.hardening * response.dp
    np.testing.assert_allclose(
        material.friction * response.stress[0], radius, rtol=3e-7, atol=0.0
    )
    # The apex tangent remains the derivative of the same return map.
    if dtype == np.float64:
        tangent, finite = _matrix(dp, strain, state, material, response)
        np.testing.assert_allclose(tangent, finite, rtol=3e-7, atol=1e-10)
        np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=0.0)


def test_dp_smooth_tangent_keeps_finite_bulk_near_incompressibility():
    material = dp.DruckerPragerMaterial(nu=0.499999999999, dilatancy=0.1)
    strain = np.array([0.018, -0.009, -0.009, 0.0, 0.0, 0.0])[:, None]
    state = j2.initial_state((1,))
    ids, table = _data(strain)
    response = dp.update_cpu(strain, state, material, ids, table)
    assert response.dp[0] > 0 and response.a[0] > 0
    # Eliminating the multiplier from the volumetric compliance gives this
    # independent reference; the old K-K*K*alpha*beta/D lost ~1e-5 here.
    compliance = 1 / material.K + material.friction * material.beta / (
        3 * material.mu + material.hardening
    )
    np.testing.assert_allclose(response.bulk, 1 / compliance, rtol=3e-15, atol=0.0)


@pytest.mark.parametrize("dilatancy", [None, 0.1, 0.0])
@pytest.mark.parametrize("branch", ["elastic", "smooth", "apex"])
def test_dp_exact_tangent_and_nonassociation(dilatancy, branch):
    material = dp.DruckerPragerMaterial(dilatancy=dilatancy)
    strain = np.array([0.018, -0.006, -0.006, 0.003, 0.001, 0.004])[:, None]
    if branch == "elastic":
        strain *= 0.01
    elif branch == "apex":
        strain[:] = MANDEL_IDENTITY[:, None] * 0.1
    state = j2.initial_state((1,))
    ids, table = _data(strain)
    response = dp.update_cpu(strain, state, material, ids, table)
    if branch == "elastic":
        assert response.dp[0] == 0
    else:
        assert response.dp[0] > 0
        if branch == "smooth":
            assert _q(response.stress)[0] > 1e-3
        else:
            np.testing.assert_allclose(_q(response.stress), 0.0, rtol=0.0, atol=1e-14)
    tangent, finite = _matrix(dp, strain, state, material, response)
    np.testing.assert_allclose(tangent, finite, rtol=3e-7, atol=3e-9)
    if branch == "smooth" and dilatancy is not None:
        assert np.max(np.abs(tangent - tangent.T)) > 1e-3
    else:
        np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=3e-15)


@pytest.mark.parametrize(
    "law,material",
    [
        (j2, j2.J2Material()),
        (dp, dp.DruckerPragerMaterial(dilatancy=0.1)),
    ],
)
def test_repeated_trials_and_commits_own_their_state(law, material):
    strain = np.array(
        [
            [0.00001, 0.018, 0.1],
            [0.0, -0.006, 0.1],
            [0.0, -0.006, 0.1],
            [0.0, 0.003, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.004, 0.0],
        ]
    )
    state = j2.initial_state((3,))
    state.equivalent[:] = 0.004
    old_plastic, old_equivalent = state.plastic_strain.copy(), state.equivalent.copy()
    ids, table = _data(strain)
    response = law.update_cpu(strain, state, material, ids, table)
    snapshot = {
        f.name: getattr(response, f.name).copy()
        for f in fields(response)
        if getattr(response, f.name) is not None
    }
    law.update_cpu(-strain, state, material, ids, table)
    repeated = law.update_cpu(strain, state, material, ids, table)
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(getattr(repeated, name), expected)
        np.testing.assert_array_equal(getattr(response, name), expected)
        assert np.all(np.isfinite(expected))
    committed = j2.commit_state(state, response)
    np.testing.assert_array_equal(state.plastic_strain, old_plastic)
    np.testing.assert_array_equal(state.equivalent, old_equivalent)
    assert not np.shares_memory(committed.plastic_strain, state.plastic_strain)
    assert not np.shares_memory(committed.equivalent, state.equivalent)
    assert np.all(committed.equivalent >= state.equivalent)
    # The returned stress is consistent with the committed elastic strain.
    elastic = strain - committed.plastic_strain
    expected_stress = 2 * material.mu * _deviator(elastic)
    expected_stress[:3] += material.K * elastic[:3].sum(axis=0)
    np.testing.assert_allclose(response.stress, expected_stress, rtol=3e-12, atol=3e-14)
    committed.plastic_strain[:] = 100
    committed.equivalent[:] = 200
    for name, expected in snapshot.items():
        np.testing.assert_array_equal(getattr(response, name), expected)
    np.testing.assert_array_equal(state.plastic_strain, old_plastic)
    np.testing.assert_array_equal(state.equivalent, old_equivalent)


@pytest.mark.parametrize(
    "law,material",
    [
        (j2, j2.J2Material()),
        (dp, dp.DruckerPragerMaterial(dilatancy=0.1)),
    ],
)
def test_elastic_fiber_never_commits_plastic_history(law, material):
    strain = np.array([0.1, -0.01, -0.01, 0.03, 0.01, 0.04])[:, None]
    state = j2.initial_state((1,))
    state.plastic_strain[:] = 0.0001
    state.equivalent[:] = 0.01
    ids = np.ones(1, dtype=np.int32)
    table = np.stack([np.eye(6), np.diag(np.arange(1.0, 7.0))])
    response = law.update_cpu(strain, state, material, ids, table)
    np.testing.assert_allclose(response.stress, table[1] @ strain, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(response.dp, 0.0)
    value = np.arange(6.0)[:, None]
    np.testing.assert_allclose(
        law.tangent_cpu(value, response, material, ids, table),
        table[1] @ value,
        rtol=0.0,
        atol=0.0,
    )
    committed = j2.commit_state(state, response)
    np.testing.assert_array_equal(committed.plastic_strain, state.plastic_strain)
    np.testing.assert_array_equal(committed.equivalent, state.equivalent)


def test_j2_radial_tangent_matches_finite_difference():
    material = j2.J2Material()
    strain = np.array([0.018, -0.006, -0.006, 0.003, 0.001, 0.004])[:, None]
    state = j2.initial_state((1,))
    ids, table = _data(strain)
    response = j2.update_cpu(strain, state, material, ids, table)
    tangent, finite = _matrix(j2, strain, state, material, response)
    np.testing.assert_allclose(tangent, finite, rtol=3e-7, atol=3e-9)
    np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=3e-15)


def test_j2_zero_midpoint_direction_falls_back_to_radial_without_mutation():
    material = j2.J2Material()
    state = j2.initial_state((1,))
    start = np.zeros((6, 1))
    # A zero average trial deviator cannot define a midpoint flow. Even
    # for an over-yield supplied start, the radial fallback stays finite.
    start[3] = 2 * material.yield_stress / (np.sqrt(1.5) * 2 * material.mu)
    strain = -start
    ids, table = _data(strain)
    radial = j2.update_cpu(strain, state, material, ids, table)
    midpoint = j2.update_cpu(strain, state, material, ids, table, start=start)
    for name in ("stress", "direction", "a", "b", "dp"):
        np.testing.assert_array_equal(getattr(midpoint, name), getattr(radial, name))
    np.testing.assert_array_equal(state.plastic_strain, 0.0)
    np.testing.assert_array_equal(state.equivalent, 0.0)


def test_j2_midpoint_preserves_documented_symmetric_tangent_approximation():
    material = j2.J2Material()
    state = j2.initial_state((1,))
    start = np.zeros((6, 1))
    start[3] = material.yield_stress / (np.sqrt(1.5) * 2 * material.mu)
    strain = start.copy()
    strain[4] = 0.004
    ids, table = _data(strain)
    response = j2.update_cpu(strain, state, material, ids, table, start=start)
    radius = material.yield_stress + material.hardening * response.dp
    np.testing.assert_allclose(_q(response.stress), radius, rtol=3e-13, atol=0.0)
    tangent, finite = _matrix(j2, strain, state, material, response, start=start)
    np.testing.assert_allclose(tangent, tangent.T, rtol=0.0, atol=3e-15)
    # This is the existing deliberate CG-compatible approximation. The
    # midpoint's exact Jacobian is nonsymmetric; no solver policy is changed.
    assert np.max(np.abs(finite - finite.T)) > 1e-3
    assert np.max(np.abs(tangent - finite)) > 1e-3


@pytest.mark.parametrize(
    "law,material",
    [
        (j2, j2.J2Material()),
        (dp, dp.DruckerPragerMaterial(dilatancy=0.1)),
    ],
)
@pytest.mark.parametrize("shape", [(), (1,), (1, 2, 3)])
def test_scalar_and_voxel_shapes_remain_supported(law, material, shape):
    strain = np.zeros((6,) + shape)
    strain[0] = 0.018
    strain[1:3] = -0.006
    strain[5] = 0.004
    state = j2.initial_state(shape)
    ids, table = _data(strain)
    response = law.update_cpu(strain, state, material, ids, table)
    assert response.stress.shape == strain.shape
    assert response.dp.shape == shape
    assert np.all(response.dp > 0)
    assert np.all(np.isfinite(law.tangent_cpu(strain, response, material, ids, table)))


@pytest.mark.parametrize(
    "law,material",
    [
        (j2, j2.J2Material()),
        (dp, dp.DruckerPragerMaterial(dilatancy=0.1)),
    ],
)
def test_zero_strain_stays_finite_under_strict_numpy_errors(law, material):
    strain = np.zeros((6, 2))
    state = j2.initial_state((2,))
    ids, table = _data(strain)
    with np.errstate(all="raise"):
        response = law.update_cpu(strain, state, material, ids, table)
        tangent = law.tangent_cpu(np.ones_like(strain), response, material, ids, table)
    assert np.all(np.isfinite(tangent))
    np.testing.assert_array_equal(response.stress, 0.0)
    np.testing.assert_array_equal(response.dp, 0.0)
