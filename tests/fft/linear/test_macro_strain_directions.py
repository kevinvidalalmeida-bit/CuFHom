"""Arbitrary macroscopic load directions reproduce the unit-load superposition."""

import numpy as np
import pytest

from tests.fft.linear.test_linear_coordinates import case, solve_homogenization


def solve(parameters, device):
    strains, stresses = {}, {}
    to_host = (lambda f: f.get()) if device else np.asarray
    p = dict(
        parameters,
        solution_field_consumer=lambda i, f: strains.__setitem__(i, to_host(f)),
        stress_field_consumer=lambda i, f: stresses.__setitem__(i, to_host(f)),
        solution_field_device=device,
        stress_field_device=device,
        stress_field_equilibrate=True,
    )
    columns = np.asarray(solve_homogenization(p), dtype=float)
    return columns, strains, stresses


@pytest.mark.parametrize(
    "backend", [pytest.param("cupy-float32", marks=pytest.mark.gpu), "scipy-float64"]
)
def test_directions_match_unit_load_superposition(backend):
    gpu_backend = backend.startswith("cupy")
    p = dict(case((6, 4, 8)), solver_rtol=1e-5 if gpu_backend else 1e-11)
    if not gpu_backend:
        p.update(fft_backend="scipy", solver_real_dtype="float64")
    ceff, strains, stresses = solve(p, device=False)
    directions = np.random.default_rng(3).standard_normal((3, 6))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    columns, dir_strains, dir_stresses = solve(
        dict(p, macro_strains=directions), device=gpu_backend
    )
    tolerance = 2e-4 if gpu_backend else 1e-8
    np.testing.assert_allclose(
        columns,
        ceff @ directions.T,
        rtol=tolerance,
        atol=tolerance * np.abs(ceff).max(),
    )
    assert sorted(dir_strains) == sorted(dir_stresses) == [0, 1, 2]
    for k, direction in enumerate(directions):
        expected_strain = np.tensordot(
            direction, np.stack([strains[i] for i in range(6)]), 1
        )
        expected_stress = np.tensordot(
            direction, np.stack([stresses[i] for i in range(6)]), 1
        )
        for actual, expected in (
            (dir_strains[k], expected_strain),
            (dir_stresses[k], expected_stress),
        ):
            scale = np.abs(expected).max()
            assert np.abs(actual - expected).max() <= tolerance * scale


def test_invalid_directions_are_rejected():
    with pytest.raises(ValueError, match="macro_strains"):
        solve_homogenization(dict(case((6, 4, 8)), macro_strains=np.ones((7, 6))))
