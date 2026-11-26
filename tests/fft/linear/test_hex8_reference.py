"""CPU-only checks for the staged reference, including independent quadrature."""

from itertools import product

import numpy as np
import pytest
from cufhom.fft.linear.hex8_general import (
    Hex8Operator,
    homogenize_hex8,
    solve_hex8_load,
)
from cufhom.fft.linear.hex8_reference import Hex8ReferenceOperator


def materials(seed=57, phases=5):
    a = np.random.default_rng(seed).normal(size=(phases, 6, 6))
    return a.transpose(0, 2, 1) @ a + 3 * np.eye(6)


def quadrature(shape, lengths):
    """Independent physical gradients and volume-normalized Gauss weights."""
    h = np.asarray(lengths) / shape
    corners = np.array(list(product((-1, 1), repeat=3)))
    gradients = []
    for point in product((-1 / np.sqrt(3), 1 / np.sqrt(3)), repeat=3):
        derivatives = np.empty((8, 3))
        for a, corner in enumerate(corners):
            for d in range(3):
                derivatives[a, d] = (
                    corner[d]
                    / (4 * h[d])
                    * np.prod([1 + corner[j] * point[j] for j in range(3) if j != d])
                )
        gradients.append(derivatives)
    weights = np.full(8, np.prod(h) / (8 * np.prod(lengths)))
    return np.asarray(gradients), weights


def independent_strains(u, lengths):
    """Cell-local vector gradients, without the production B or action tables."""
    shape = u.shape[1:]
    derivatives, weights = quadrature(shape, lengths)
    corners = np.array(list(product((0, 1), repeat=3)))
    strains = np.empty((*shape, 8, 6))
    for cell in np.ndindex(shape):
        values = np.array(
            [u[(slice(None), *tuple((cell + node) % shape))] for node in corners]
        )
        gradient = np.einsum("ac,gad->gcd", values, derivatives)
        strains[cell] = np.column_stack(
            (
                gradient[:, 0, 0],
                gradient[:, 1, 1],
                gradient[:, 2, 2],
                (gradient[:, 1, 2] + gradient[:, 2, 1]) / np.sqrt(2),
                (gradient[:, 0, 2] + gradient[:, 2, 0]) / np.sqrt(2),
                (gradient[:, 0, 1] + gradient[:, 1, 0]) / np.sqrt(2),
            )
        )
    return strains, weights


def weighted_energy(left, c, right, weights):
    return np.einsum("xyzgi,xyzij,xyzgj,g->", left, c, right, weights)


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 3, 3), (4, 3, 6)])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_indexed_anisotropic_action_matches_parent(shape, dtype):
    rng = np.random.default_rng(101)
    labels = rng.integers(0, 5, shape)
    c = materials()
    packed = np.stack([c[:, i, j] for i in range(6) for j in range(i, 6)], axis=1)
    kwargs = dict(
        lengths=(0.7, 1.4, 2.3),
        reference=materials(71, 1)[0],
        backend="numpy",
        dtype=dtype,
    )
    reference = Hex8ReferenceOperator(labels, packed, **kwargs)
    parent = Hex8Operator(labels, c, **kwargs)
    u = rng.normal(size=(3, *shape)).astype(dtype)
    np.testing.assert_array_equal(reference.C, parent.C)
    np.testing.assert_array_equal(reference.reference, parent.reference)
    np.testing.assert_array_equal(reference.inverse, parent.inverse)
    assert reference.gamma_minus == parent.gamma_minus
    for audit in (False, True):
        tol = 3e-6 if dtype == "float32" and not audit else 2e-13
        expected = parent.action(u, audit=audit)
        np.testing.assert_allclose(
            reference.action(u, audit=audit),
            expected,
            atol=tol * np.max(np.abs(expected)),
            rtol=tol,
        )
    macro = rng.normal(size=6)
    np.testing.assert_array_equal(reference.rhs(macro), parent.rhs(macro))
    np.testing.assert_array_equal(reference.precondition(u), parent.precondition(u))
    np.testing.assert_array_equal(
        reference.mean_stress(u, macro), parent.mean_stress(u, macro)
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_independent_weighted_energy_and_adjoint(dtype):
    rng = np.random.default_rng(102)
    shape, lengths = (3, 4, 2), (0.6, 1.3, 2.1)
    labels = rng.integers(0, 5, shape)
    op = Hex8ReferenceOperator(
        labels, materials(), lengths=lengths, backend="numpy", dtype=dtype
    )
    u, v = rng.normal(size=(2, 3, *shape)).astype(dtype)
    eu, weights = independent_strains(u.astype(float), lengths)
    ev, _ = independent_strains(v.astype(float), lengths)
    ku, kv = op.action(u, audit=True), op.action(v, audit=True)
    expected = weighted_energy(eu, op.C[labels], ev, weights)
    np.testing.assert_allclose(np.vdot(u, kv), expected, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(np.vdot(v, ku), expected, rtol=2e-13, atol=1e-12)
    assert np.vdot(u, ku) > 0
    np.testing.assert_allclose(
        np.vdot(u, ku), weighted_energy(eu, op.C[labels], eu, weights), rtol=2e-14
    )
    np.testing.assert_allclose(ku.sum((1, 2, 3)), 0, atol=2e-13)

    # Test B.T against arbitrary Gauss stresses, independently of C and symmetry.
    stress = rng.normal(size=(8, 6, *shape))
    adjoint = np.zeros_like(ku)
    for g, bg in enumerate(op.B):
        op._numpy_adjoint(stress[g], bg, weights[g], adjoint)
    virtual_work = np.einsum("xyzgs,gsxyz,g->", eu, stress, weights)
    np.testing.assert_allclose(
        np.vdot(u, adjoint), virtual_work, rtol=2e-13, atol=2e-14
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_output_contract_and_fresh_fp64_audit(dtype):
    rng = np.random.default_rng(103)
    labels = rng.integers(0, 5, (3, 4, 2))
    op = Hex8ReferenceOperator(labels, materials(), backend="numpy", dtype=dtype)
    u = rng.normal(size=(3, *labels.shape)).astype(dtype)
    for audit in (False, True):
        expected = op.action(u, audit=audit)
        out = np.full(expected.shape, np.nan, dtype=expected.dtype)
        assert op.action(u, audit=audit, out=out) is out
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(op.action(u, audit=audit), expected)
        with pytest.raises(ValueError, match="output"):
            op.action(expected, audit=audit, out=expected)
        with pytest.raises(ValueError, match="output"):
            op.action(u, audit=audit, out=np.empty_like(out, dtype="int64"))
        with pytest.raises(ValueError, match="output"):
            op.action(u, audit=audit, out=np.empty_like(out)[..., ::-1])
        with pytest.raises(ValueError, match="output"):
            op.action(u, audit=audit, out=out[..., :1])
        np.testing.assert_allclose(
            op.action(u[..., ::-1], audit=audit),
            op.action(u[..., ::-1].copy(), audit=audit),
            atol=0,
            rtol=0,
        )
    with pytest.raises(ValueError, match="displacement"):
        op.action(u[0])
    snapshot = u.copy()
    expected = Hex8Operator(labels, op.C, backend="numpy", dtype="float64").action(
        u, audit=True
    )
    # Neither regular nor audit action may depend on parent's contracted tables.
    op.ke.fill(np.nan)
    op.ke64.fill(np.nan)
    np.testing.assert_allclose(
        op.action(u, audit=True), expected, rtol=2e-13, atol=2e-13
    )
    assert np.isfinite(op.action(u)).all()
    np.testing.assert_array_equal(u, snapshot)
    assert op.action(u, audit=True).dtype == np.float64


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_homogeneous_null_modes_and_reference_inverse(dtype):
    shape = (4, 3, 6)
    c = materials(phases=1)
    op = Hex8ReferenceOperator(
        np.zeros(shape, int),
        c,
        reference=c[0],
        lengths=(1.7, 0.6, 2.2),
        backend="numpy",
        dtype=dtype,
    )
    translation = np.broadcast_to(
        np.array([0.3, -1.7, 4.2])[:, None, None, None], (3, *shape)
    ).astype(dtype)
    for audit in (False, True):
        np.testing.assert_array_equal(op.action(translation, audit=audit), 0)
    checkerboard = np.broadcast_to(
        (-1.0) ** np.arange(shape[-1]), translation.shape
    ).astype(dtype)
    assert np.linalg.norm(op.action(checkerboard)) > 1
    rng = np.random.default_rng(104)
    u = rng.normal(size=translation.shape).astype(dtype)
    u -= u.mean((1, 2, 3), keepdims=True)
    np.testing.assert_allclose(
        op.precondition(op.action(u)),
        u,
        rtol=3e-6 if dtype == "float32" else 3e-13,
        atol=3e-6 if dtype == "float32" else 3e-13,
    )
    for macro in np.eye(6):
        np.testing.assert_array_equal(op.rhs(macro), 0)
        assert solve_hex8_load(op, macro)["iterations"] == 0


@pytest.mark.parametrize(
    "dtype,rtol,tol", [("float64", 1e-10, 3e-11), ("float32", 2e-5, 3e-5)]
)
def test_parent_solver_six_loads_and_independent_total_energy(dtype, rtol, tol):
    rng = np.random.default_rng(105)
    shape, lengths = (3, 3, 2), (0.8, 1.1, 1.9)
    labels, c = rng.integers(0, 5, shape), materials()
    kwargs = dict(
        lengths=lengths, reference=materials(72, 1)[0], backend="numpy", dtype=dtype
    )
    op = Hex8ReferenceOperator(labels, c, **kwargs)
    parent = Hex8Operator(labels, c, **kwargs)
    fields, residuals, totals, stress = [], [], [], np.empty((6, 6))
    for a, macro in enumerate(np.eye(6)):
        info = solve_hex8_load(op, macro, rtol=rtol)
        other = solve_hex8_load(parent, macro, rtol=rtol)
        assert info["accepted"] and info["residual_inf"] <= info["threshold"]
        np.testing.assert_array_equal(
            info["residual"], op.rhs(macro) - op.action(info["solution"], audit=True)
        )
        np.testing.assert_allclose(
            info["solution"], other["solution"], atol=tol, rtol=tol
        )
        fields.append(info["solution"])
        residuals.append(info["residual"])
        strain, weights = independent_strains(info["solution"].astype(float), lengths)
        totals.append(strain + macro)
        stress[:, a] = op.mean_stress(info["solution"], macro)
    correction = np.array([[op.dot(u, r) for r in residuals] for u in fields])
    raw = stress - correction
    effective = (raw + raw.T) / 2
    energy = np.array(
        [[weighted_energy(a, op.C[labels], b, weights) for b in totals] for a in totals]
    )
    np.testing.assert_allclose(effective, energy, atol=2e-12, rtol=2e-12)
    expected = homogenize_hex8(
        labels, c, **kwargs, rtol=rtol, storage="host", return_fields=True
    )
    np.testing.assert_allclose(effective, expected["C"], atol=tol, rtol=tol)


def test_constitutive_gather_crosses_chunk_boundary():
    shape = (21, 20, 20)  # More cells than a constitutive chunk.
    rng = np.random.default_rng(106)
    labels = rng.integers(0, 5, shape)
    op = Hex8ReferenceOperator(labels, materials(), backend="numpy")
    strain = rng.normal(size=(6, *shape))
    stress = np.full_like(strain, np.nan)
    op._numpy_constitutive(op.C, strain, stress)
    expected = sum(
        op.C[labels, :, j].transpose(3, 0, 1, 2) * strain[j] for j in range(6)
    )
    np.testing.assert_allclose(stress, expected, rtol=2e-14, atol=2e-14)
