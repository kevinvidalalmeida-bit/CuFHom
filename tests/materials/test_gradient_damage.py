"""GPU-only tests of the effective law and its frozen-state contract."""

import numpy as np
import pytest
from cufhom.materials.drucker_prager import CuPyDruckerPrager, DruckerPragerMaterial
from cufhom.materials.gradient_damage import (
    CuPyGradientDamage,
    GradientDamageParameters,
)
from cufhom.materials.plasticity import initial_state

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


def engine(dtype="float64", **parameters):
    m = DruckerPragerMaterial(friction=0.3, dilatancy=0.1)
    ids = cp.zeros((4, 4, 4), dtype=cp.int32)
    table = cp.eye(6, dtype=dtype)[None]
    return CuPyGradientDamage(cp, m, ids, table, dtype, parameters, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_undamaged_return_matches_existing_complete_dp(dtype):
    g = engine(dtype)
    reference = CuPyDruckerPrager(cp, g.material, g.ids, g.table, dtype)
    state = initial_state(g.shape, cp, dtype)
    e = cp.zeros((6,) + g.shape, dtype=dtype)
    e[0] = 0.013
    e[1] = -0.002
    e[5] = 0.003
    r, q = g.update(e, state), reference.update(e, state)
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
        cp.testing.assert_array_equal(getattr(r, name), getattr(q, name))
    actual, expected = g.audit(r, state), reference.audit(q, state)
    for a, b in zip(actual, expected):
        cp.testing.assert_allclose(a, b, atol=2e-8 if dtype == "float32" else 2e-16)


def test_modified_von_mises_uniaxial_tension_compression():
    g = engine(strength_ratio=2.5)
    e = cp.zeros((6,) + g.shape)
    e[0] = 0.012
    e[1:3] = -0.012 * g.material.nu
    cp.testing.assert_allclose(g.source(e), 0.012, rtol=1e-13)
    cp.testing.assert_allclose(g.source(-e), 0.012 / 2.5, rtol=1e-13)


def test_complete_fixed_damage_tangent_matches_finite_differences():
    g = engine()
    g.set_trial(cp.full(g.shape, 0.37))
    state = initial_state(g.shape, cp, "float64")
    e = cp.zeros((6,) + g.shape)
    e[0] = 0.018
    e[1:3] = -0.006
    e[5] = 0.004
    r = g.update(e, state)
    for j in range(6):
        v = cp.zeros_like(e)
        v[j] = 1.0
        finite = (
            g.update(e + 1e-7 * v, state).stress - g.update(e - 1e-7 * v, state).stress
        ) / 2e-7
        cp.testing.assert_allclose(g.tangent(v, r), finite, rtol=2e-7, atol=2e-8)
    assert float(cp.max(r.dp)) > 0


def test_trial_history_rollback_and_nonnegative_damage_work():
    g = engine(strain_threshold=0.003, softening_strain=0.008)
    state = initial_state(g.shape, cp, "float64")
    e = cp.zeros((6,) + g.shape)
    e[0] = 0.01
    e[1:3] = -0.003
    d, info = g.propose(e)
    assert info["damage_gap"] > 0
    g.set_trial(d)
    r = g.update(e, state)
    assert float(cp.min(g.damage_work(e, state, r))) >= 0
    cp.testing.assert_array_equal(g.confirmed.value, 0)
    cp.testing.assert_array_equal(g.confirmed.history, 0)
    g.begin()
    cp.testing.assert_array_equal(g.trial, 0)
    cp.testing.assert_array_equal(g.degradation, 1)
    cp.testing.assert_array_equal(state.equivalent, 0)
    d, _ = g.propose(e)
    g.set_trial(d)
    g.propose(e)
    g.accept()
    old = g.confirmed.value.copy()
    g.begin()
    lower, _ = g.propose(e * 0.1)
    cp.testing.assert_array_equal(lower, old)


@pytest.mark.parametrize(
    "bad",
    [
        dict(length_scale=0),
        dict(strain_threshold=-1),
        dict(strength_ratio=0.9),
        dict(residual_stiffness=0),
        dict(max_iterations=1.5),
        dict(softening_strain=float("nan")),
    ],
)
def test_invalid_parameters(bad):
    with pytest.raises(ValueError):
        GradientDamageParameters(**bad)


def test_corrupt_damage_checkpoint_is_rejected():
    g = engine()
    with pytest.raises(ValueError, match="values"):
        g.restore(cp.full(g.shape, -0.1), cp.zeros(g.shape))
    with pytest.raises(ValueError, match="mismatch"):
        g.restore(cp.zeros(g.shape), cp.full(g.shape, 0.1))


def test_small_residual_stiffness_and_precise_history_are_preserved():
    g = engine("float32", residual_stiffness=1e-10, softening_strain=1e-12)
    g.set_trial(cp.ones(g.shape, dtype="float32"))
    cp.testing.assert_allclose(g.degradation, 1e-10, rtol=1e-7)
    field = cp.full(g.shape, 0.01000000001, dtype="float64")
    g.regularizer.solve = lambda *a, **kw: (
        field,
        dict(iterations=0, relative_residual=0.0),
    )
    d, _ = g.propose(cp.zeros((6,) + g.shape, dtype="float32"))
    g.set_trial(d)
    g.accept()
    assert g.confirmed.history.dtype == cp.float64
    assert float(cp.min(g.confirmed.value)) > 0.99
    g.restore(g.confirmed.value, g.confirmed.history)


def test_nonfinite_damage_source_is_not_hidden():
    g = engine()
    e = cp.zeros((6,) + g.shape)
    e[0] = cp.nan
    assert bool(cp.all(cp.isnan(g.source(e))))
    with pytest.raises(ValueError, match="finite"):
        g.propose(e)


def paraboloidal_engine(dtype="float64", **parameters):
    from cufhom.materials.gradient_damage import CuPyParaboloidalDamage
    from cufhom.materials.paraboloidal import ParaboloidalMaterial

    m = ParaboloidalMaterial(
        tension_curve=[[0, 0.029], [0.02, 0.05], [0.1, 0.07]],
        compression_curve=[[0, 0.067], [0.2, 0.12]],
    )
    ids = cp.zeros((4, 4, 4), dtype=cp.int32)
    table = cp.eye(6, dtype=dtype)[None]
    return CuPyParaboloidalDamage(cp, m, ids, table, dtype, parameters, (1.0, 1.0, 1.0))


def test_paraboloidal_damage_tangent_matches_finite_differences():
    g = paraboloidal_engine()
    g.set_trial(cp.full(g.shape, 0.37))
    state = initial_state(g.shape, cp, "float64")
    e = cp.zeros((6,) + g.shape)
    e[0] = 0.018
    e[1:3] = -0.006
    e[5] = 0.004
    r = g.update(e, state)
    assert float(cp.max(r.dp)) > 0
    for j in range(6):
        v = cp.zeros_like(e)
        v[j] = 1.0
        finite = (
            g.update(e + 1e-7 * v, state).stress - g.update(e - 1e-7 * v, state).stress
        ) / 2e-7
        cp.testing.assert_allclose(g.tangent(v, r), finite, rtol=2e-6, atol=2e-8)
    violation, work, _ = g.audit(r, state)
    assert float(cp.max(violation)) < 1e-12 and float(cp.min(work)) >= 0


def test_paraboloidal_damage_scales_the_undamaged_return():
    from cufhom.materials.paraboloidal import CuPyParaboloidal

    g = paraboloidal_engine()
    reference = CuPyParaboloidal(cp, g.material, g.ids, g.table, "float64")
    state = initial_state(g.shape, cp, "float64")
    e = cp.zeros((6,) + g.shape)
    e[0] = 0.02
    e[1] = -0.004
    e[5] = 0.003
    q = reference.update(e, state)
    for d in (0.0, 0.6):
        g.set_trial(cp.full(g.shape, d))
        r = g.update(e, state)
        scale = 1 - (1 - g.parameters.residual_stiffness) * d
        cp.testing.assert_array_equal(r.dp, q.dp)
        cp.testing.assert_allclose(r.stress, scale * q.stress, rtol=1e-14)
        cp.testing.assert_allclose(r.a, scale * q.a, rtol=1e-14)


def test_interphase_mask_is_a_city_block_shell():
    from cufhom.materials.gradient_damage import interphase_mask

    ids = cp.zeros((8, 8, 8), dtype=cp.int32)
    ids[4, 4, 4] = 1
    mask = interphase_mask(cp, ids, (1.0, 1.0, 1.0), 0.125)
    expected = np.zeros((8, 8, 8), bool)
    for z in np.argwhere(np.ones((8, 8, 8))):
        expected[tuple(z)] = 0 < np.abs(z - 4).sum() <= 1
    np.testing.assert_array_equal(cp.asnumpy(mask).astype(bool), expected)
    assert mask.dtype == cp.uint8


@pytest.mark.parametrize(
    "bad",
    [
        dict(interphase_thickness=0.1),
        dict(interphase_strain_threshold=0.01, interphase_softening_strain=0.01),
        dict(
            interphase_thickness=-1.0,
            interphase_strain_threshold=0.01,
            interphase_softening_strain=0.01,
        ),
    ],
)
def test_invalid_interphase_parameters(bad):
    with pytest.raises(ValueError):
        GradientDamageParameters(**bad)
