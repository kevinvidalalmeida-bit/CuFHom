"""Fast algebraic checks for the composite-voxel interface formulation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from cufhom_core.mechanics.matcoef import ElasticTensor
from cufhom.pipeline.composite_voxels import (  # noqa: E402
    laminate_voxel_stiffness_mandel,
    rasterize_capsules_composite,
)
from cufhom.pipeline.fft_solver import (  # noqa: E402
    Enu_to_KG,
    TI_stiffness_voigt,
    solve_homogenization,
    voigt_to_mandel,
)


def _phase_stiffness() -> tuple[np.ndarray, np.ndarray]:
    bulk, shear = Enu_to_KG(3.0, 0.30)
    matrix = ElasticTensor(bulk=bulk, mu=shear).mandel
    fibre = voigt_to_mandel(TI_stiffness_voigt(70.0, 8.0, 0.25, 0.30, 5.0))
    return matrix, fibre


def test_laminate_recovers_pure_phases_and_is_spd() -> None:
    matrix, fibre = _phase_stiffness()
    fraction = np.array([[[0.0, 0.5, 1.0]]])
    orientation = np.zeros((1, 1, 3, 3))
    orientation[..., 0] = 1.0
    normal = np.zeros_like(orientation)
    normal[..., 1] = 1.0
    field = laminate_voxel_stiffness_mandel(
        matrix, fibre, fraction, orientation, normal, storage="full", dtype=np.float64
    )
    assert np.allclose(field[:, :, 0, 0, 0], matrix)
    assert np.allclose(field[:, :, 0, 0, 2], fibre)
    cut = field[:, :, 0, 0, 1]
    assert np.allclose(cut, cut.T)
    assert np.linalg.eigvalsh(cut)[0] > 0.0


def test_capsule_subvoxel_geometry_is_fractional_at_curved_boundary() -> None:
    fraction, orientation, normal = rasterize_capsules_composite(
        np.array([[8.0, 8.0, 8.0]]), np.array([[1.0, 0.0, 0.0]]),
        fiber_length=8.0, fiber_radius=2.0, grid_size=16, samples_per_axis=2,
    )
    cut = (fraction > 0.0) & (fraction < 1.0)
    assert np.any(cut)
    assert np.allclose(np.linalg.norm(orientation[fraction > 0.0], axis=1), 1.0)
    assert np.all(np.linalg.norm(normal[cut], axis=1) > 0.0)


def test_solver_builds_composite_cfield_from_fraction_and_normal() -> None:
    grid = 8
    fraction, orientation, normal = rasterize_capsules_composite(
        np.array([[4.0, 4.0, 4.0]]), np.array([[1.0, 0.0, 0.0]]),
        fiber_length=4.0, fiber_radius=1.25, grid_size=grid, samples_per_axis=2,
    )
    result = solve_homogenization(
        {
            "input_dir": str(ROOT),
            "seed": 0,
            "phase_array": (fraction >= 0.5).astype(np.uint8),
            "ori_array": orientation,
            "fiber_fraction_array": fraction,
            "interface_normal_array": normal,
            "Em": 3.0,
            "nu_m": 0.30,
            "Ef_L": 30.0,
            "Ef_T": 8.0,
            "nu_LT": 0.25,
            "nu_TT": 0.30,
            "G_LT": 5.0,
            "fft_backend": "scipy",
            "solver_real_dtype": "float64",
            "solver_rtol": 1.0e-7,
            "solver_maxiter": 1000,
            "require_convergence": True,
            "cfield_storage": "sym21",
        }
    )
    assert result.shape == (6, 6)
    assert np.allclose(result, result.T)
    assert np.linalg.eigvalsh(result)[0] > 0.0
