"""Midpoint (second-order) and backward Euler J2 return maps."""

import numpy as np
import pytest
from cufhom.fft.nonlinear.solver import cp, solve_nonlinear_homogenization
from cufhom.materials.plasticity import (
    CuPyMaterial,
    J2Material,
    PlasticState,
    update_cpu,
)


def _fiber_case(**extra):
    phase = np.zeros((6, 6, 6), dtype=np.uint8)
    phase[2:4, 2:4, :] = 1
    orientation = np.zeros(phase.shape + (3,))
    orientation[..., 2] = 1
    return dict(
        phase_array=phase, ori_array=orientation, path=[0.0, 0.006], rtol=1e-11, **extra
    )


def _curve(**parameters):
    history = solve_nonlinear_homogenization(parameters)["history"]
    return np.array([r["target_strain"] for r in history]), np.array(
        [r["stress"][0] for r in history]
    )


def test_midpoint_equals_radial_return_on_proportional_paths():
    # Homogeneous J2 under uniaxial stress: the deviator keeps its direction,
    # so both maps flow along it and return the same, exact, curve.
    p = dict(
        phase_array=np.zeros((4, 4, 4), np.uint8),
        ori_array=np.zeros((4, 4, 4, 3)),
        path=[0.0, 0.01],
        initial_increment=0.001,
        rtol=1e-11,
    )
    _, euler = _curve(**p, integrator="backward_euler")
    _, midpoint = _curve(**p, integrator="midpoint")
    np.testing.assert_allclose(midpoint, euler, rtol=1e-10, atol=1e-14)


def test_midpoint_is_more_accurate_than_euler_with_four_times_larger_steps():
    strain, reference = _curve(
        **_fiber_case(initial_increment=0.0005 / 16, integrator="midpoint")
    )
    errors = {}
    for integrator in ("backward_euler", "midpoint"):
        for h in (0.001, 0.00025):
            e, s = _curve(**_fiber_case(initial_increment=h, integrator=integrator))
            errors[integrator, h] = np.max(np.abs(s - np.interp(e, strain, reference)))
    # First order: halving twice divides the error by about four (here ~3).
    assert 2.5 < errors["backward_euler", 0.001] / errors["backward_euler", 0.00025] < 6
    assert errors["midpoint", 0.001] < errors["backward_euler", 0.00025] / 10


def _random_state(shape, seed):
    rng = np.random.default_rng(seed)
    start = rng.normal(size=(6,) + shape) * 0.006
    strain = start + rng.normal(size=(6,) + shape) * 0.0008
    plastic = rng.normal(size=(6,) + shape) * 0.0005
    plastic[:3] -= plastic[:3].mean(axis=0)
    return strain, start, PlasticState(plastic, np.abs(rng.normal(size=shape)) * 0.002)


def test_midpoint_update_yields_exactly_at_the_end_of_the_step():
    material = J2Material()
    strain, start, state = _random_state((5, 4, 3), 2)
    ids, table = np.zeros((5, 4, 3), np.int32), np.zeros((1, 6, 6))
    response = update_cpu(strain, state, material, ids, table, start)
    plastic = response.dp > 0
    assert plastic.mean() > 0.5
    deviator = response.stress.copy()
    deviator[:3] -= deviator[:3].mean(axis=0)
    radius = material.yield_stress + material.hardening * (
        state.equivalent + response.dp
    )
    np.testing.assert_allclose(
        np.sqrt(1.5 * np.sum(deviator * deviator, axis=0))[plastic],
        radius[plastic],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(response.direction**2, axis=0)[plastic], 1, rtol=1e-12
    )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_midpoint_cuda_update_matches_cpu():
    material = J2Material()
    strain, start, state = _random_state((5, 4, 3), 3)
    ids, table = np.zeros((5, 4, 3), np.int32), np.zeros((1, 6, 6))
    expected = update_cpu(strain, state, material, ids, table, start)
    gpu = CuPyMaterial(cp, material, cp.asarray(ids), cp.asarray(table), np.float64)
    actual = gpu.update(
        cp.asarray(strain),
        PlasticState(cp.asarray(state.plastic_strain), cp.asarray(state.equivalent)),
        cp.asarray(start),
    )
    for name in ("stress", "direction", "a", "b", "dp"):
        np.testing.assert_allclose(
            cp.asnumpy(getattr(actual, name)),
            getattr(expected, name),
            rtol=1e-12,
            atol=1e-15,
        )


def test_integrator_option_is_validated_and_reported():
    p = dict(_fiber_case(), path=[0.0, 0.002], initial_increment=0.001, rtol=1e-8)
    assert solve_nonlinear_homogenization(p)["integrator"] == "midpoint"
    with pytest.raises(ValueError, match="integrator must"):
        solve_nonlinear_homogenization(dict(p, integrator="rk4"))


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
def test_float32_midpoint_keeps_the_yield_surface_for_large_increments():
    # Fiber tips see large local increments: the quadratic alone cancels in
    # float32; the Newton refinement keeps the surface below the step audit.
    material = J2Material(E=4.34, nu=0.35)
    rng = np.random.default_rng(5)
    shape = (200_000,)
    plastic = rng.normal(size=(6,) + shape) * 0.002
    plastic[:3] -= plastic[:3].mean(axis=0)
    equivalent = np.abs(rng.normal(size=shape)) * 0.003
    direction = rng.normal(size=(6,) + shape)
    direction[:3] -= direction[:3].mean(axis=0)
    direction /= np.linalg.norm(direction, axis=0)
    radius = material.yield_stress + material.hardening * equivalent
    start = plastic + direction * radius / (np.sqrt(1.5) * 2 * material.mu)
    strain = start + rng.normal(size=(6,) + shape) * 3e-3
    gpu = CuPyMaterial(
        cp,
        material,
        cp.zeros(shape, cp.int32),
        cp.zeros((1, 6, 6), cp.float32),
        np.float32,
    )

    def f32(a):
        return cp.asarray(a, cp.float32)

    response = gpu.update(
        f32(strain), PlasticState(f32(plastic), f32(equivalent)), f32(start)
    )
    stress, dp = (
        cp.asnumpy(response.stress).astype(float),
        cp.asnumpy(response.dp).astype(float),
    )
    deviator = (
        stress
        - np.r_[
            stress[:3].mean(axis=0, keepdims=True).repeat(3, 0), np.zeros((3,) + shape)
        ]
    )
    q = np.sqrt(1.5 * np.sum(deviator**2, axis=0))
    surface = material.yield_stress + material.hardening * (
        np.float32(equivalent).astype(float) + dp
    )
    assert (dp > 0).mean() > 0.9
    assert np.max(np.abs(q - surface)[dp > 0]) / material.yield_stress < 2e-6


def test_quadratic_predictor_does_not_change_the_converged_curve():
    p = _fiber_case(initial_increment=0.0005)
    selected = solve_nonlinear_homogenization(p)
    linear = solve_nonlinear_homogenization(dict(p, predictor=False))
    np.testing.assert_allclose(
        [r["stress"] for r in selected["history"]],
        [r["stress"] for r in linear["history"]],
        atol=1e-10,
    )
