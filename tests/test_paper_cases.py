"""Verify the compact paper inputs without running a homogenization solver."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_cases", ROOT / "benchmarks/paper/cases.py"
)
cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cases)


@pytest.mark.parametrize("case", cases.CASES)
def test_small_grid_matches_frozen_input_checksums(case):
    checks = json.loads((cases.INPUTS / "geometry_checks.json").read_text())["checks"]
    expected = next(row for row in checks if row["case"] == case and row["n"] == 21)
    index, table, metadata = cases.prepare_case(case, 21)
    assert index.shape == (21, 21, 21)
    assert index.dtype == np.int32
    assert table.shape[1] == 21
    assert metadata["index_sha256"] == expected["index_sha256"]
    assert metadata["table_sha256"] == expected["table_sha256"]
    assert index.min() >= 0 and index.max() < len(table)


def test_public_image_axis_order_and_resampling():
    with np.load(cases.INPUTS / "public_phase192.npz") as data:
        raw = data["phase"]
    index, _, metadata = cases.prepare_case("public", 9)
    axes = np.floor((np.arange(9) + 0.5) * 192 / 9).astype(int)
    for i, j, k in ((0, 1, 2), (2, 7, 4), (8, 5, 1)):
        assert index[i, j, k] == raw[axes[k], axes[j], axes[i]]
    assert "XYZ transpose" in metadata["sampling"]


def test_copper_rotations_preserve_anisotropic_material():
    _, table, _ = cases.prepare_case("polycrystal", 5)
    definition = json.loads((cases.INPUTS / "periodic_families.json").read_text())
    copper = definition["polycrystal"]
    full = np.zeros((40, 6, 6))
    for column, (i, j) in enumerate((i, j) for i in range(6) for j in range(i, 6)):
        full[:, i, j] = full[:, j, i] = table[:, column]
    basis = np.zeros((6, 3, 3))
    for a, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))):
        basis[a, i, j] = basis[a, j, i] = 1 if i == j else 1 / np.sqrt(2)
    local = np.diag([copper["C11"]] * 3 + [2 * copper["C44"]] * 3)
    local[:3, :3] += copper["C12"] * (np.ones((3, 3)) - np.eye(3))
    for actual, rotation in zip(
        full, Rotation.from_quat(copper["quaternions_xyzw"]).as_matrix(), strict=True
    ):
        mandel = np.einsum("aij,ik,bkl,jl->ab", basis, rotation, basis, rotation)
        np.testing.assert_allclose(
            actual, mandel @ local @ mandel.T, rtol=0, atol=1e-13
        )
    assert np.ptp(full[:, 0, 0]) > 1
    assert np.linalg.eigvalsh(full).min() > 0


def test_float32_material_is_explicit_rounding_of_nominal_table():
    index64, table64, _ = cases.prepare_case("polycrystal", 5)
    index32, table32, _ = cases.prepare_case("polycrystal", 5, "float32")
    np.testing.assert_array_equal(index64, index32)
    np.testing.assert_array_equal(table32, table64.astype("float32"))
    assert table32.dtype == np.float32


@pytest.mark.parametrize("n", [True, False, np.bool_(True), 1, 0, -1, 3.5, "21"])
def test_invalid_grid_is_rejected(n):
    with pytest.raises(ValueError, match="integer"):
        cases.prepare_case("public", n)


def test_unknown_case_and_nonfloating_material_are_rejected():
    with pytest.raises(ValueError, match="Unknown case"):
        cases.prepare_case("aligned_short_fibres", 21)
    with pytest.raises(ValueError, match="float32 or float64"):
        cases.prepare_case("public", 21, "int32")
