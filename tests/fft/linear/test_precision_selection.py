"""Public precision choices preserve the established numerical configurations."""

import numpy as np
import pytest
from cufhom.fft import IndexedMaterial, homogenize


@pytest.fixture
def material():
    labels = np.indices((4, 4, 4)).sum(axis=0).astype(np.int32) % 2
    table = np.stack([2 * np.eye(6), 8 * np.eye(6)])
    table[0, :3, :3] += 1
    table[1, :3, :3] += 4
    return labels, table


@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_cpu_precision_matches_explicit_numerical_controls(material, kind, precision):
    labels, table = material
    dtype = "float32" if precision == "fp32" else "float64"
    tolerance = 1e-5 if precision == "fp32" else 1e-6
    actual = homogenize(
        labels, table, discretization=kind, backend="numpy", precision=precision
    )
    expected = homogenize(
        labels,
        table,
        discretization=kind,
        backend="numpy",
        dtype=dtype,
        working=dtype,
        rtol=tolerance,
    )
    np.testing.assert_array_equal(actual["C"], expected["C"])
    assert actual["precision_contract"] == expected["precision_contract"]
    assert actual["iterations"] == expected["iterations"]
    assert actual["precision"] == precision
    assert actual["dtype"] == actual["working"] == dtype
    for audit in actual["audits"]:
        assert audit["accepted"]
        assert audit["threshold"] == pytest.approx(tolerance * audit["rhs_norm"])


def test_default_call_works_without_cuda_and_reports_resolved_precision(
    material, monkeypatch
):
    monkeypatch.setattr(
        "cufhom.fft.linear.solver._default_fft_backend", lambda: "scipy"
    )
    result = homogenize(*material)
    assert result["backend"] == "numpy"
    assert result["precision"] == "fp64"
    assert result["dtype"] == result["working"] == "float64"


@pytest.mark.parametrize(
    "options, message",
    [
        ({"precision": "float16"}, "precision must"),
        ({"precision": "mixed", "backend": "numpy"}, "requires CUDA"),
        ({"precision": "fp64", "dtype": "float32"}, "conflicts"),
        ({"precision": "mixed", "working": "float64", "backend": "cupy"}, "conflicts"),
        ({"precision": "fp32", "working": "float64"}, "conflicts"),
    ],
)
def test_conflicting_modes_are_rejected_instead_of_silently_changed(
    material, options, message
):
    with pytest.raises(ValueError, match=message):
        homogenize(*material, **options)


def test_fp64_cannot_recover_coefficients_from_prepared_fp32_material(material):
    rounded = IndexedMaterial(*material, dtype="float32")
    with pytest.raises(ValueError, match="already defines.*precision"):
        homogenize(rounded, precision="fp64", backend="numpy")


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize("precision", ["fp32", "fp64", "mixed"])
def test_cuda_modes_match_existing_routes(material, kind, precision):
    dtype = "float32" if precision == "fp32" else "float64"
    working = "float64" if precision == "fp64" else "float32"
    rtol = 1e-5 if precision == "fp32" else 1e-6
    actual = homogenize(*material, discretization=kind, precision=precision)
    expected = homogenize(
        *material,
        discretization=kind,
        backend="cupy",
        dtype=dtype,
        working=working,
        rtol=rtol,
    )
    assert actual["backend"] == "cupy"
    assert actual["precision"] == precision
    assert actual["precision_contract"] == expected["precision_contract"]
    np.testing.assert_array_equal(actual["C"], expected["C"])


@pytest.mark.gpu
def test_default_cuda_call_uses_mixed_precision(material):
    result = homogenize(*material)
    assert result["backend"] == "cupy"
    assert result["precision"] == "mixed"
    assert result["dtype"] == "float64" and result["working"] == "float32"
