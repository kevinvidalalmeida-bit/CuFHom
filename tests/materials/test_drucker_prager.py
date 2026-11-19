"""Drucker-Prager parameter contracts and GPU checks against closed forms.

Run serially with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 on a CUDA host. CuPy and
the material modules are imported by fixtures, never by a CPU fallback.
NumPy is used only for small analytic oracles and comparisons, not solves.
"""

from types import SimpleNamespace

import numpy as np
import pytest

FIELDS = (
    "stress",
    "direction",
    "a",
    "b",
    "dp",
    "bulk",
    "coupling_to_direction",
    "coupling_to_pressure",
)
DTYPES = [np.float32, np.float64]
BETAS = [None, 0.1, 0.0]
I_MANDEL = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA device unavailable: {exc}")
    if count == 0:
        pytest.skip("CUDA device unavailable")
    return cupy


@pytest.fixture(scope="module")
def api():
    from cufhom.materials.drucker_prager import (
        CuPyDruckerPrager,
        DruckerPragerMaterial,
    )
    from cufhom.materials.plasticity import (
        CuPyMaterial,
        J2Material,
        PlasticState,
        commit_state,
        initial_state,
    )

    return SimpleNamespace(
        CuPyDruckerPrager=CuPyDruckerPrager,
        DruckerPragerMaterial=DruckerPragerMaterial,
        CuPyMaterial=CuPyMaterial,
        J2Material=J2Material,
        PlasticState=PlasticState,
        commit_state=commit_state,
        initial_state=initial_state,
    )


def _moduli(material):
    # Deliberately independent of the implementation's K/mu properties.
    return (
        material.E / (3 * (1 - 2 * material.nu)),
        material.E / (2 * (1 + material.nu)),
    )


def _beta(material):
    return material.friction if material.dilatancy is None else material.dilatancy


def _deviator(value):
    result = np.array(value, dtype=np.float64, copy=True)
    result[:3] -= result[:3].mean(axis=0)
    return result


def _elastic_action(value, material):
    bulk, shear = _moduli(material)
    result = 2 * shear * np.asarray(value, dtype=np.float64)
    result[:3] += (bulk - 2 * shear / 3) * np.sum(value[:3], axis=0)
    return result


def _reference_return(strain, plastic_strain, equivalent, material):
    """Scalar backward-Euler smooth/apex formulas, without a material call."""
    shape = strain.shape[1:]
    elastic = np.asarray(strain, dtype=np.float64) - plastic_strain
    bulk, shear = _moduli(material)
    alpha, beta, hardening = material.friction, _beta(material), material.hardening
    trial_dev = 2 * shear * _deviator(elastic)
    trial_mean = bulk * elastic[:3].sum(axis=0)
    trial_q = np.sqrt(1.5 * np.sum(trial_dev**2, axis=0))
    stress = np.empty_like(elastic)
    increment = np.zeros_like(elastic)
    gamma = np.zeros(shape)
    branch = np.full(shape, "elastic", dtype="U7")
    for index in np.ndindex(shape):
        column = (slice(None),) + index
        radius = material.yield_stress + hardening * equivalent[index]
        q, mean, dev = trial_q[index], trial_mean[index], trial_dev[column]
        excess = q + alpha * mean - radius
        returned_dev = dev.copy()
        if excess > 0:
            dg = excess / (3 * shear + alpha * bulk * beta + hardening)
            if q - 3 * shear * dg > 0:
                branch[index] = "smooth"
                increment[column] = 1.5 * dg * dev / q
                returned_dev *= 1 - 3 * shear * dg / q
            else:
                branch[index] = "apex"
                dg = (alpha * mean - radius) / (alpha * bulk * beta + hardening)
                increment[column] = dev / (2 * shear)
                returned_dev[:] = 0
            gamma[index] = dg
            increment[column] += beta * dg * I_MANDEL / 3
            mean -= bulk * beta * dg
        stress[column] = returned_dev + mean * I_MANDEL
    return SimpleNamespace(stress=stress, increment=increment, dp=gamma, branch=branch)


def _branch_strains():
    # Elastic, hydrocompression, smooth, apex with deviator, pure hydroapex.
    return np.array(
        [
            [0.0001, 0.0002, -0.0001, 0.0001, -0.0001, 0.0002],
            [-0.03, -0.03, -0.03, 0.0, 0.0, 0.0],
            [0.022, -0.007, -0.006, 0.005, 0.002, -0.003],
            [0.052, 0.048, 0.050, 0.001, -0.0007, 0.0004],
            [0.040, 0.040, 0.040, 0.0, 0.0, 0.0],
        ]
    ).T.copy()


def _gpu(cp, api, material, shape, dtype=np.float64, ids=None, table=None):
    if ids is None:
        ids = np.zeros(shape, dtype=np.int32)
    if table is None:
        # Matrix behavior must come from E/nu, not table[0].
        table = np.zeros((1, 6, 6))
    return api.CuPyDruckerPrager(
        cp,
        material,
        cp.asarray(ids, dtype=cp.int32),
        cp.asarray(table, dtype=dtype),
        dtype,
    )


def _state(cp, api, plastic, equivalent, dtype=np.float64):
    return api.PlasticState(
        cp.asarray(plastic, dtype=dtype), cp.asarray(equivalent, dtype=dtype)
    )


def _tol(dtype):
    return (2e-5, 2e-7) if dtype == np.float32 else (3e-12, 3e-14)


def _assert_reference(cp, response, expected, dtype=np.float64):
    rtol, atol = _tol(dtype)
    np.testing.assert_allclose(
        cp.asnumpy(response.stress), expected.stress, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        cp.asnumpy(response.dp), expected.dp, rtol=rtol, atol=atol
    )
    increment = np.sqrt(1.5) * cp.asnumpy(response.direction) * cp.asnumpy(response.dp)
    np.testing.assert_allclose(increment, expected.increment, rtol=rtol, atol=atol)
    shape = expected.dp.shape
    for name in FIELDS:
        value = getattr(response, name)
        assert isinstance(value, cp.ndarray), name
        assert value.dtype == np.dtype(dtype), name
        assert value.shape == (
            (6,) + shape if name in ("stress", "direction") else shape
        ), name
        assert bool(cp.all(cp.isfinite(value))), name


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_analytic_elasticity_with_nonzero_frozen_state(cp, api, dtype):
    material = api.DruckerPragerMaterial()
    shape = (2, 3, 2)
    rng = np.random.default_rng(42)
    plastic = rng.normal(size=(6,) + shape) * 0.0003  # Includes plastic volume.
    equivalent = np.full(shape, 0.02)
    elastic = rng.normal(size=plastic.shape) * 0.0002
    strain = plastic + elastic
    state = _state(cp, api, plastic, equivalent, dtype)
    gpu = _gpu(cp, api, material, shape, dtype)
    response = gpu.update(cp.asarray(strain, dtype=dtype), state)
    expected = _reference_return(strain, plastic, equivalent, material)
    assert np.all(expected.branch == "elastic")
    _assert_reference(cp, response, expected, dtype)
    np.testing.assert_array_equal(cp.asnumpy(response.dp), np.zeros(shape))
    np.testing.assert_array_equal(cp.asnumpy(response.direction), np.zeros_like(strain))
    value = rng.normal(size=strain.shape)
    rtol, atol = _tol(dtype)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.tangent(cp.asarray(value, dtype=dtype), response)),
        _elastic_action(value, material),
        rtol=rtol,
        atol=atol * 10,
    )


@pytest.mark.gpu
def test_defaults_are_associated(cp, api):
    default = api.DruckerPragerMaterial()
    assert (
        default.E,
        default.nu,
        default.yield_stress,
        default.hardening,
        default.friction,
    ) == (3.0, 0.3, 0.015, 0.03, 0.3)
    explicit = api.DruckerPragerMaterial(dilatancy=0.3)
    strain = cp.asarray(_branch_strains())
    state = api.initial_state((5,), cp, np.float64)
    responses = [
        _gpu(cp, api, material, (5,)).update(strain, state)
        for material in (default, explicit)
    ]
    for name in FIELDS:
        cp.testing.assert_array_equal(
            getattr(responses[0], name), getattr(responses[1], name)
        )


def test_calibration_from_initial_uniaxial_strengths(api):
    m = api.DruckerPragerMaterial.from_strengths(0.012, 0.018, dilatancy=0.1)
    np.testing.assert_allclose([m.friction, m.yield_stress], [0.6, 0.0144], rtol=1e-14)
    assert abs(0.012 + m.friction * (0.012 / 3) - m.yield_stress) < 1e-15
    assert abs(0.018 - m.friction * (0.018 / 3) - m.yield_stress) < 1e-15
    equal = api.DruckerPragerMaterial.from_strengths(0.015, 0.015)
    assert equal.friction == 0 and equal.beta == 0
    for a, b in ((0, 0.018), (0.018, 0.012), (np.nan, 0.018)):
        with pytest.raises(ValueError):
            api.DruckerPragerMaterial.from_strengths(a, b)
    with pytest.raises(ValueError, match="determine"):
        api.DruckerPragerMaterial.from_strengths(0.012, 0.018, friction=0.3)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_uniaxial_tension_and_compression_have_different_yield_thresholds(
    cp, api, dtype
):
    material = api.DruckerPragerMaterial()
    tensile = material.yield_stress / (material.E * (1 + material.friction / 3))
    compressive = material.yield_stress / (material.E * (1 - material.friction / 3))
    assert tensile < material.yield_stress / material.E < compressive
    axial = np.array(
        [
            0.8 * tensile,
            1.2 * tensile,
            -0.8 * compressive,
            -1.2 * compressive,
            material.yield_stress / material.E,
            -material.yield_stress / material.E,
        ]
    )
    # These strains give an elastic uniaxial trial stress, including Poisson contraction.
    strain = np.zeros((6, axial.size))
    strain[0], strain[1], strain[2] = axial, -material.nu * axial, -material.nu * axial
    shape = (axial.size,)
    expected = _reference_return(
        strain, np.zeros_like(strain), np.zeros(shape), material
    )
    np.testing.assert_array_equal(
        expected.branch, ["elastic", "smooth", "elastic", "smooth", "smooth", "elastic"]
    )
    response = _gpu(cp, api, material, shape, dtype).update(
        cp.asarray(strain, dtype=dtype), api.initial_state(shape, cp, dtype)
    )
    _assert_reference(cp, response, expected, dtype)
    np.testing.assert_array_equal(
        cp.asnumpy(response.dp) > 0, [False, True, False, True, True, False]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dilatancy", BETAS)
def test_hydrocompression_is_elastic_and_hydrotension_returns_to_apex(
    cp, api, dtype, dilatancy
):
    material = api.DruckerPragerMaterial(dilatancy=dilatancy)
    strain = I_MANDEL[:, None] * np.array([-0.03, 0.03])[None, :]
    state = api.initial_state((2,), cp, dtype)
    response = _gpu(cp, api, material, (2,), dtype).update(
        cp.asarray(strain, dtype=dtype), state
    )
    expected = _reference_return(strain, np.zeros_like(strain), np.zeros(2), material)
    np.testing.assert_array_equal(expected.branch, ["elastic", "apex"])
    _assert_reference(cp, response, expected, dtype)
    committed = api.commit_state(state, response)
    rtol, atol = _tol(dtype)
    np.testing.assert_allclose(
        cp.asnumpy(committed.plastic_strain[:3].sum(axis=0)),
        _beta(material) * expected.dp,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        cp.asnumpy(committed.equivalent), expected.dp, rtol=rtol, atol=atol
    )
    assert float(committed.equivalent[1]) > 0
    if dilatancy == 0:
        # k counts the apex multiplier even with zero plastic strain/volume.
        np.testing.assert_allclose(
            cp.asnumpy(committed.plastic_strain),
            np.zeros_like(strain),
            rtol=0,
            atol=2 * np.finfo(dtype).eps * 0.03,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dilatancy", BETAS)
def test_fused_commit_is_bitwise_original_and_keeps_inputs(cp, api, dtype, dilatancy):
    m = api.DruckerPragerMaterial(dilatancy=dilatancy)
    g = _gpu(cp, api, m, (5,), dtype)
    state = api.initial_state((5,), cp, dtype)
    response = g.update(cp.asarray(_branch_strains(), dtype=dtype), state)
    before = [
        a.copy()
        for a in (
            state.plastic_strain,
            state.equivalent,
            response.direction,
            response.dp,
        )
    ]
    reference = api.commit_state(state, response)
    actual = g.commit(state, response)
    for a, b in (
        (reference.plastic_strain, actual.plastic_strain),
        (reference.equivalent, actual.equivalent),
    ):
        cp.testing.assert_array_equal(a.view(cp.uint8), b.view(cp.uint8))
    for a, b in zip(
        (state.plastic_strain, state.equivalent, response.direction, response.dp),
        before,
    ):
        cp.testing.assert_array_equal(a.view(cp.uint8), b.view(cp.uint8))
        assert not cp.shares_memory(a, actual.plastic_strain)
        assert not cp.shares_memory(a, actual.equivalent)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dilatancy", BETAS)
def test_smooth_and_apex_flow_increment_and_plastic_volume(cp, api, dtype, dilatancy):
    material = api.DruckerPragerMaterial(dilatancy=dilatancy)
    strain = _branch_strains()
    shape = (5,)
    state = api.initial_state(shape, cp, dtype)
    response = _gpu(cp, api, material, shape, dtype).update(
        cp.asarray(strain, dtype=dtype), state
    )
    expected = _reference_return(
        strain, np.zeros_like(strain), np.zeros(shape), material
    )
    np.testing.assert_array_equal(
        expected.branch, ["elastic", "elastic", "smooth", "apex", "apex"]
    )
    _assert_reference(cp, response, expected, dtype)
    committed = api.commit_state(state, response)
    rtol, atol = _tol(dtype)
    increment = cp.asnumpy(committed.plastic_strain)
    np.testing.assert_allclose(
        increment[:3].sum(axis=0), _beta(material) * expected.dp, rtol=rtol, atol=atol
    )
    # At the apex the entire trial deviator is removed; its magnitude is not gamma.
    np.testing.assert_allclose(
        _deviator(increment)[:, 3], _deviator(strain)[:, 3], rtol=rtol, atol=atol
    )
    assert not np.isclose(
        np.sqrt(2 / 3) * np.linalg.norm(_deviator(increment)[:, 3]),
        float(committed.equivalent[3]),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_matrix_and_anisotropic_fiber_table(cp, api, dtype):
    material = api.DruckerPragerMaterial(dilatancy=0.1)
    shape = (5, 7, 9)  # Cross a CUDA block boundary and exercise the partial tail.
    n = np.prod(shape)
    indices = np.arange(n)
    ids = (indices % 3).astype(np.int32).reshape(shape)
    factor = np.arange(36, dtype=float).reshape(6, 6) / 30 + np.eye(6)
    table = np.stack(
        [
            np.zeros((6, 6)),
            factor.T @ factor + np.eye(6),
            2 * factor @ factor.T + np.diag([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ]
    )
    plastic = np.full((6,) + shape, 0.0002)
    equivalent = np.full(shape, 0.01)
    strain = plastic + _branch_strains()[:, indices % 5].reshape((6,) + shape)
    gpu = _gpu(cp, api, material, shape, dtype, ids, table)
    state = _state(cp, api, plastic, equivalent, dtype)
    response = gpu.update(cp.asarray(strain, dtype=dtype), state)
    expected = _reference_return(strain, plastic, equivalent, material)
    fiber = ids != 0
    indexed = np.einsum(
        "nij,jn->in", table[ids.ravel()], strain.reshape(6, -1)
    ).reshape(strain.shape)
    expected.stress[:, fiber] = indexed[:, fiber]
    expected.dp[fiber] = 0
    expected.increment[:, fiber] = 0
    _assert_reference(cp, response, expected, dtype)
    assert set(expected.branch[~fiber]) == {"elastic", "smooth", "apex"}
    np.testing.assert_array_equal(cp.asnumpy(response.dp)[fiber], np.zeros(fiber.sum()))
    rng = np.random.default_rng(17)
    value = rng.normal(size=strain.shape)
    tangent = cp.asnumpy(gpu.tangent(cp.asarray(value, dtype=dtype), response))
    indexed = np.einsum("nij,jn->in", table[ids.ravel()], value.reshape(6, -1)).reshape(
        value.shape
    )
    rtol, atol = _tol(dtype)
    np.testing.assert_allclose(
        tangent[:, fiber], indexed[:, fiber], rtol=rtol, atol=atol * 10
    )
    committed = api.commit_state(state, response)
    np.testing.assert_array_equal(
        cp.asnumpy(committed.plastic_strain)[:, fiber],
        cp.asnumpy(state.plastic_strain)[:, fiber],
    )
    np.testing.assert_array_equal(
        cp.asnumpy(committed.equivalent)[fiber], cp.asnumpy(state.equivalent)[fiber]
    )


@pytest.mark.gpu
def test_commit_allocates_new_state_and_preserves_inputs_and_response(cp, api):
    material = api.DruckerPragerMaterial(dilatancy=0.1)
    plastic = np.full((6, 5), 0.0002)
    equivalent = np.full(5, 0.007)
    strain_host = plastic + _branch_strains()
    strain = cp.asarray(strain_host)
    state = _state(cp, api, plastic, equivalent)
    gpu = _gpu(cp, api, material, (5,))
    inputs = [strain, state.plastic_strain, state.equivalent, gpu.ids, gpu.table]
    snapshots = [value.copy() for value in inputs]
    response = gpu.update(strain, state)
    response_snapshots = {name: getattr(response, name).copy() for name in FIELDS}
    expected = _reference_return(strain_host, plastic, equivalent, material)
    committed = api.commit_state(state, response)
    assert committed is not state
    assert committed.plastic_strain.data.ptr != state.plastic_strain.data.ptr
    assert committed.equivalent.data.ptr != state.equivalent.data.ptr
    np.testing.assert_allclose(
        cp.asnumpy(committed.plastic_strain),
        plastic + expected.increment,
        rtol=3e-12,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        cp.asnumpy(committed.equivalent),
        equivalent + expected.dp,
        rtol=3e-12,
        atol=3e-14,
    )
    value = cp.asarray(np.random.default_rng(10).normal(size=strain_host.shape))
    value_snapshot = value.copy()
    gpu.tangent(value, response)
    # Mutating the newly allocated state also must not affect its inputs.
    committed.plastic_strain.fill(123)
    committed.equivalent.fill(456)
    for original, snapshot in zip(inputs, snapshots):
        cp.testing.assert_array_equal(original, snapshot)
    cp.testing.assert_array_equal(value, value_snapshot)
    for name in FIELDS:
        cp.testing.assert_array_equal(getattr(response, name), response_snapshots[name])


@pytest.mark.gpu
@pytest.mark.parametrize("dilatancy", [None, 0.1])
def test_load_unload_history_repeated_updates_use_only_frozen_state(cp, api, dilatancy):
    material = api.DruckerPragerMaterial(dilatancy=dilatancy)
    shape = (1, 1, 5)
    base = _branch_strains().reshape((6,) + shape)
    rotated = base.copy()
    rotated[3:] *= -1
    path = [base * 0.005, base, base * 0.6, rotated, base * 1.4]
    plastic, equivalent = np.zeros_like(base), np.zeros(shape)
    state = api.initial_state(shape, cp, np.float64)
    gpu = _gpu(cp, api, material, shape)
    branches = set()
    for strain_host in path:
        strain = cp.asarray(strain_host)
        old_plastic, old_equivalent = (
            state.plastic_strain.copy(),
            state.equivalent.copy(),
        )
        response = gpu.update(strain, state, start=None)
        snapshots = {name: getattr(response, name).copy() for name in FIELDS}
        expected = _reference_return(strain_host, plastic, equivalent, material)
        branches.update(expected.branch.ravel())
        _assert_reference(cp, response, expected)
        # An intervening equilibrium trial cannot contaminate a repeated trial.
        gpu.update(cp.asarray(strain_host * 0.7), state)
        repeated = gpu.update(strain, state)
        for name in FIELDS:
            cp.testing.assert_array_equal(getattr(repeated, name), snapshots[name])
            cp.testing.assert_array_equal(getattr(response, name), snapshots[name])
        committed = api.commit_state(state, response)
        cp.testing.assert_array_equal(state.plastic_strain, old_plastic)
        cp.testing.assert_array_equal(state.equivalent, old_equivalent)
        plastic = plastic + expected.increment
        equivalent = equivalent + expected.dp
        np.testing.assert_allclose(
            cp.asnumpy(committed.plastic_strain), plastic, rtol=3e-12, atol=3e-14
        )
        np.testing.assert_allclose(
            cp.asnumpy(committed.equivalent), equivalent, rtol=3e-12, atol=3e-14
        )
        assert bool(cp.all(committed.equivalent >= state.equivalent))
        state = committed
    assert branches == {"elastic", "smooth", "apex"}


@pytest.mark.gpu
@pytest.mark.parametrize("branch,column", [("smooth", 2), ("apex", 3)])
@pytest.mark.parametrize("dilatancy", BETAS)
def test_fp64_tangent_matches_finite_differences_and_expected_symmetry(
    cp, api, branch, column, dilatancy
):
    material = api.DruckerPragerMaterial(dilatancy=dilatancy)
    shape = (2, 2, 3)
    plastic = np.full((6,) + shape, 0.0001)
    equivalent = np.full(shape, 0.004)
    scale = np.linspace(0.98, 1.02, np.prod(shape)).reshape(shape)
    strain_host = plastic + _branch_strains()[:, column, None, None, None] * scale
    strain = cp.asarray(strain_host)
    state = _state(cp, api, plastic, equivalent)
    gpu = _gpu(cp, api, material, shape)
    response = gpu.update(strain, state)
    expected = _reference_return(strain_host, plastic, equivalent, material)
    assert np.all(expected.branch == branch)
    _assert_reference(cp, response, expected)
    h = 1e-7
    tangent_columns, fd_columns = [], []
    for component in range(6):
        value_host = np.zeros_like(strain_host)
        value_host[component] = 1
        # Both FD points stay strictly inside the same return branch.
        for sign in (-1, 1):
            perturbed = _reference_return(
                strain_host + sign * h * value_host, plastic, equivalent, material
            )
            assert np.all(perturbed.branch == branch)
        value = cp.asarray(value_host)
        tangent_columns.append(cp.asnumpy(gpu.tangent(value, response)))
        plus = gpu.update(strain + h * value, state).stress
        minus = gpu.update(strain - h * value, state).stress
        fd_columns.append(cp.asnumpy((plus - minus) / (2 * h)))
    tangent = np.stack(tangent_columns, axis=1)
    finite_difference = np.stack(fd_columns, axis=1)
    np.testing.assert_allclose(tangent, finite_difference, rtol=3e-7, atol=3e-9)
    transpose = np.swapaxes(tangent, 0, 1)
    if branch == "apex":
        bulk, _ = _moduli(material)
        apex_bulk = (
            bulk
            * material.hardening
            / (material.friction * bulk * _beta(material) + material.hardening)
        )
        matrix = apex_bulk * np.outer(I_MANDEL, I_MANDEL)
        expected_tangent = np.broadcast_to(
            matrix[:, :, None, None, None], tangent.shape
        )
        np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-12, atol=3e-14)
        # Apex tangents are purely volumetric and symmetric even without association.
        np.testing.assert_allclose(tangent, transpose, rtol=3e-12, atol=3e-14)
    elif dilatancy is None:
        np.testing.assert_allclose(tangent, transpose, rtol=3e-12, atol=3e-14)
    else:
        assert np.max(np.abs(tangent - transpose)) > 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("dilatancy", BETAS)
def test_local_admissibility_complementarity_and_plastic_work(cp, api, dilatancy):
    material = api.DruckerPragerMaterial(dilatancy=dilatancy)
    cases = _branch_strains()
    compressive_smooth = cases[:, 2] - 0.02 * I_MANDEL
    strain = np.column_stack([cases, compressive_smooth])
    shape = (strain.shape[1],)
    equivalent = np.full(shape, 0.006)
    state = _state(cp, api, np.zeros_like(strain), equivalent)
    gpu = _gpu(cp, api, material, shape)
    response = gpu.update(cp.asarray(strain), state)
    stress, gamma = cp.asnumpy(response.stress), cp.asnumpy(response.dp)
    mean = stress[:3].mean(axis=0)
    q = np.sqrt(1.5 * np.sum(_deviator(stress) ** 2, axis=0))
    radius = material.yield_stress + material.hardening * (equivalent + gamma)
    residual = q + material.friction * mean - radius
    assert np.all(gamma >= 0)
    assert np.all(residual <= 3e-14)
    assert np.any(gamma == 0) and np.any(gamma > 0)
    np.testing.assert_allclose(residual[gamma > 0], 0, atol=3e-14)
    np.testing.assert_allclose(gamma * residual, 0, atol=3e-14)
    increment = np.sqrt(1.5) * cp.asnumpy(response.direction) * gamma
    plastic_work = np.sum(stress * increment, axis=0)
    assert np.all(plastic_work >= -3e-14)
    np.testing.assert_allclose(
        plastic_work, gamma * (q + _beta(material) * mean), rtol=3e-12, atol=3e-14
    )
    violation, dissipation = gpu.audit(response, state)
    np.testing.assert_allclose(
        cp.asnumpy(violation), np.maximum(residual, 0), atol=3e-14
    )
    expected_dissipation = (
        plastic_work - material.hardening * (equivalent + gamma) * gamma
    )
    np.testing.assert_allclose(
        cp.asnumpy(dissipation), expected_dissipation, rtol=3e-12, atol=3e-14
    )
    if dilatancy is None:
        np.testing.assert_allclose(
            cp.asnumpy(dissipation),
            material.yield_stress * gamma,
            rtol=3e-12,
            atol=3e-14,
        )
        assert np.all(cp.asnumpy(dissipation) >= 0)
        stored_increment = (
            0.5 * material.hardening * ((equivalent + gamma) ** 2 - equivalent**2)
        )
        assert np.all(plastic_work - stored_increment >= -3e-14)
    else:
        # Nonassociated apex flow can have negative net hardening dissipation;
        # check the audit formula rather than assuming thermodynamic positivity.
        compression = mean < 0
        assert np.any(compression & (gamma > 0))
        assert np.all(cp.asnumpy(dissipation)[compression] >= -3e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_friction_and_dilatancy_reduce_to_gpu_j2_backward_euler(cp, api, dtype):
    dp_material = api.DruckerPragerMaterial(friction=0.0, dilatancy=0.0)
    j2_material = api.J2Material()
    shape = (3, 4, 5)
    rng = np.random.default_rng(81)
    plastic = _deviator(rng.normal(size=(6,) + shape) * 0.0004)
    equivalent = rng.uniform(0, 0.02, size=shape)
    strain = plastic + rng.normal(size=plastic.shape) * 0.01
    strain[:, 0, 0, 0] = plastic[:, 0, 0, 0] + I_MANDEL * 0.02
    strain[:, 0, 0, 1] = plastic[:, 0, 0, 1] + I_MANDEL * 0.0001
    ids = np.zeros(shape, dtype=np.int32)
    ids[1, :, :] = 1
    factor = rng.normal(size=(6, 6))
    table = np.stack([np.zeros((6, 6)), factor.T @ factor + 2 * np.eye(6)])
    dp_gpu = _gpu(cp, api, dp_material, shape, dtype, ids, table)
    j2_gpu = api.CuPyMaterial(
        cp, j2_material, cp.asarray(ids), cp.asarray(table, dtype=dtype), dtype
    )
    state = _state(cp, api, plastic, equivalent, dtype)
    strain_gpu = cp.asarray(strain, dtype=dtype)
    # No start strain: the existing J2 map is explicitly backward Euler.
    dp_response, j2_response = (
        dp_gpu.update(strain_gpu, state),
        j2_gpu.update(strain_gpu, state),
    )
    rtol, atol = _tol(dtype)
    for name in ("stress", "a", "b", "dp"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(dp_response, name)),
            cp.asnumpy(getattr(j2_response, name)),
            rtol=rtol,
            atol=atol * 10,
        )
    active = cp.asnumpy(j2_response.dp) > 0
    assert np.any(active)
    np.testing.assert_allclose(
        cp.asnumpy(dp_response.direction)[:, active],
        cp.asnumpy(j2_response.direction)[:, active],
        rtol=rtol,
        atol=atol,
    )
    value = cp.asarray(rng.normal(size=strain.shape), dtype=dtype)
    np.testing.assert_allclose(
        cp.asnumpy(dp_gpu.tangent(value, dp_response)),
        cp.asnumpy(j2_gpu.tangent(value, j2_response)),
        rtol=rtol,
        atol=atol * 10,
    )
    dp_state, j2_state = (
        api.commit_state(state, dp_response),
        api.commit_state(state, j2_response),
    )
    for name in ("plastic_strain", "equivalent"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(dp_state, name)),
            cp.asnumpy(getattr(j2_state, name)),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"E": 0},
        {"E": -1},
        {"nu": -1},
        {"nu": -1.1},
        {"nu": 0.5},
        {"nu": 0.6},
        {"yield_stress": 0},
        {"yield_stress": -0.01},
        {"hardening": 0},
        {"hardening": -0.01},
        {"friction": -0.1},
        {"friction": 3},
        {"friction": 3.1},
        {"dilatancy": -0.1},
        {"friction": 0.3, "dilatancy": 0.31},
        {"friction": 0, "dilatancy": 0.01},
    ],
)
def test_invalid_parameter_ranges_are_rejected(api, parameters):
    with pytest.raises(ValueError):
        api.DruckerPragerMaterial(**parameters)


@pytest.mark.parametrize(
    "name", ["E", "nu", "yield_stress", "hardening", "friction", "dilatancy"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_parameters_are_rejected(api, name, value):
    with pytest.raises(ValueError):
        api.DruckerPragerMaterial(**{name: value})


@pytest.mark.parametrize(
    "friction,dilatancy", [(0.0, None), (0.0, 0.0), (0.3, 0.0), (2.999, 2.999)]
)
def test_valid_friction_and_dilatancy_bounds(api, friction, dilatancy):
    material = api.DruckerPragerMaterial(friction=friction, dilatancy=dilatancy)
    assert 0 <= _beta(material) <= material.friction < 3


@pytest.mark.gpu
def test_midpoint_start_is_rejected_without_mutating_state(cp, api):
    material = api.DruckerPragerMaterial()
    gpu = _gpu(cp, api, material, (5,))
    strain = cp.asarray(_branch_strains())
    state = api.initial_state((5,), cp, np.float64)
    with pytest.raises(ValueError, match="backward_euler"):
        gpu.update(strain, state, start=cp.zeros_like(strain))
    cp.testing.assert_array_equal(state.plastic_strain, cp.zeros_like(strain))
    cp.testing.assert_array_equal(state.equivalent, cp.zeros((5,)))
