"""Optional CPU FP64 re-evaluation of all six accepted fields and cross terms.

This is deliberately separate from the fast Krylov loops. It neither solves a
different discretization nor certifies floating-point rounding. GaNi fields are
projected onto the original retained space before re-evaluation; HEX8 uses the
original nodal fields. A failed original-operator check raises with its report.
"""

import time

import numpy as np

from .common import (
    LinearSolveError,
    checked_energy_matrix,
    energy_report,
    isotropic_reference,
    reference_bounds,
)


class TensorVerificationError(LinearSolveError):
    """Report a failed effective-tensor check with its diagnostic record."""

    def __init__(self, report):
        super().__init__(
            "Final tensor verification failed: " + report["status"],
            status=report["status"],
            diagnostics=report,
        )


def _geometry(shape, lengths):
    from cufhom.fft.operators import gani_wavenumbers

    q = np.array(np.meshgrid(*gani_wavenumbers(shape, lengths), indexing="ij"))
    norm = np.linalg.norm(q, axis=0)
    valid = norm > 0
    q /= np.where(norm > 0, norm, 1)
    q *= valid
    weights = np.full(q.shape[-1], 2.0)
    weights[0] = 1.0
    if shape[-1] % 2 == 0:
        weights[-1] = 1.0
    return q, weights


def _spectral_parts(field, q):
    # Removing an anchor constant leaves all retained modes unchanged and
    # makes a homogeneous load exactly zero before the FFT.
    h = np.fft.rfftn(field - field[:, :1, :1, :1], axes=(1, 2, 3))
    x, y, z = q
    s = np.sqrt(2.0)
    v = np.array(
        [
            x * h[0] + (y * h[5] + z * h[4]) / s,
            y * h[1] + (x * h[5] + z * h[3]) / s,
            z * h[2] + (x * h[4] + y * h[3]) / s,
        ]
    )
    parallel = np.sum(q * v, axis=0)
    d = 2 * v - q * parallel
    projected = np.array(
        [
            x * d[0],
            y * d[1],
            z * d[2],
            (y * d[2] + z * d[1]) / s,
            (x * d[2] + z * d[0]) / s,
            (x * d[1] + y * d[0]) / s,
        ]
    )
    return v, parallel, projected


def _gani(material, fields, lengths, reference):
    shape, n = material.shape, int(np.prod(material.shape))
    _, bulk, shear = isotropic_reference(material.stiffness, reference)
    transverse = np.sqrt(1 / (2 * shear))
    longitudinal = np.sqrt(1 / (bulk + 4 * shear / 3))
    q, weights = _geometry(shape, lengths)
    totals = []
    projection_changes = []
    for a, field in enumerate(fields):
        _, _, projected = _spectral_parts(field, q)
        admissible = np.fft.irfftn(projected, s=shape, axes=(1, 2, 3))
        projection_changes.append(
            float(np.linalg.norm(admissible - field) / max(1.0, np.linalg.norm(field)))
        )
        admissible[a] += 1.0
        totals.append(admissible)
    totals = np.asarray(totals).reshape(6, 6, n)
    energy, stress_mean = np.zeros((6, 6)), np.zeros((6, 6))
    residuals, norms, rhs_norms = [], [], []
    ids, table = material.index.ravel(), material.stiffness
    for a in range(6):
        stress = np.empty((6, n))
        load = np.empty_like(stress)
        for start in range(0, n, 8192):
            sl = slice(start, start + 8192)
            c = table[ids[sl]]
            stress[:, sl] = np.einsum("nij,jn->in", c, totals[a, :, sl])
            load[:, sl] = c[:, :, a].T
        stress_mean[:, a] = stress.mean(axis=1)
        energy[:, a] = np.einsum("bcn,cn->b", totals, stress) / n
        v, parallel, projected = _spectral_parts(stress.reshape(6, *shape), q)
        norms.append(float(np.sqrt(np.sum(weights * np.abs(projected) ** 2)) / n))
        _, _, projected_load = _spectral_parts(load.reshape(6, *shape), q)
        rhs_norms.append(
            float(np.sqrt(np.sum(weights * np.abs(projected_load) ** 2)) / n)
        )
        # r_z/sqrt(N), including the RFFT metric, so Re(V* V) is G0.
        residuals.append(
            (
                -1j
                * np.sqrt(weights)
                / n
                * (
                    np.sqrt(2.0) * transverse * v
                    + (longitudinal - np.sqrt(2.0) * transverse) * q * parallel
                )
            ).ravel()
        )
    residuals = np.asarray(residuals)
    g0 = (residuals.conj() @ residuals.T).real
    return (
        energy,
        stress_mean,
        g0,
        norms,
        rhs_norms,
        dict(
            metric="projected_stress_RMS",
            projection_change_relative=projection_changes,
            primary_array_bytes=int(totals.nbytes + residuals.nbytes),
        ),
    )


def _hex8(material, fields, lengths, reference):
    from .hex8_general import _NODES, Hex8Operator

    op = Hex8Operator(
        material.index,
        material.stiffness,
        lengths=lengths,
        reference=reference,
        backend="numpy",
        dtype="float64",
    )
    shape, n = material.shape, op.n
    u = np.asarray(fields, dtype=np.float64).reshape(6, 3, n)
    energy, stress_mean = np.zeros((6, 6)), np.zeros((6, 6))
    # Integrate the six fields together, including every cross product. No
    # FP32 retained solution or residual correction enters this assembly.
    b = op.B.reshape(8, 6, 24)
    for start in range(0, n, 1024):
        flat = np.arange(start, min(start + 1024, n))
        cells = np.array(np.unravel_index(flat, shape)).T
        nodes = (cells[:, None, :] + _NODES[None]) % shape
        ids = np.ravel_multi_index(nodes.transpose(2, 0, 1), shape)
        local = u[:, :, ids].transpose(0, 2, 3, 1).reshape(6, len(flat), 24)
        strain = np.einsum("gij,anj->angi", b, local)
        strain += np.eye(6)[:, None, None, :]
        stress = np.einsum(
            "nij,angj->angi", material.stiffness[material.index.ravel()[flat]], strain
        )
        energy += np.einsum("angi,bngi->ab", strain, stress) / (8 * n)
        stress_mean += stress.sum(axis=(1, 2)).T / (8 * n)
    residuals, norms, rhs_norms = [], [], []
    for a in range(6):
        rhs = op.rhs(np.eye(6)[a])
        r = rhs - op.action(u[a].reshape(3, *shape), audit=True)
        residuals.append(r.ravel())
        norms.append(float(np.max(np.abs(r))))
        rhs_norms.append(float(np.max(np.abs(rhs))))
    residuals = np.asarray(residuals)
    g0 = np.empty((6, 6))
    for a in range(6):
        z = op.precondition(residuals[a].reshape(3, *shape)).ravel()
        g0[:, a] = residuals @ z
    return (
        energy,
        stress_mean,
        g0,
        norms,
        rhs_norms,
        dict(
            metric="nodal_force_Linf",
            projection_change_relative=[0.0] * 6,
            primary_array_bytes=int(u.nbytes + residuals.nbytes),
        ),
    )


def verify_effective_tensor(
    material,
    fields,
    *,
    kind,
    lengths,
    reference,
    production,
    rtol,
    atol,
    tensor_rtol=None,
    energy_acceptance=False,
    coefficient_source="input_float64",
):
    """Return a reassembled matrix only after its fresh numerical check passes.

    With tensor_rtol, the diagonal energy target is checked numerically. Without
    it, only native residual acceptance is checked; no tensor-error tolerance is
    inferred from rtol. Reports always state that rounding is not certified.
    """
    started = time.perf_counter()
    expected = (6, 6 if kind == "gani" else 3, *material.shape)
    fields = np.asarray(fields, dtype=np.float64)
    if fields.shape != expected or not np.isfinite(fields).all():
        raise LinearSolveError(
            "Invalid fields for final tensor verification", status="nonfinite"
        )
    matrix, stress, g0, norms, rhs, detail = (_gani if kind == "gani" else _hex8)(
        material, fields, lengths, reference
    )
    matrix, defect = checked_energy_matrix(matrix)
    g0 = (g0 + g0.T) / 2
    bounds = reference_bounds(material.stiffness, reference)
    bracket = energy_report(matrix, bounds, np.diag(g0))
    bracket.update(
        full_G0_matrix_available=True,
        assembly_route="full_cross_products_float64",
        retained_solution_rounding_present=False,
        operator="FP64 re-evaluation on " + coefficient_source,
    )
    thresholds = np.maximum(atol, rtol * np.asarray(rhs))
    native_passed = bool(np.all(np.asarray(norms) <= thresholds))
    upper = np.asarray(bracket["upper"])
    energy_passed = (
        None
        if tensor_rtol is None
        else bool(np.all(upper <= tensor_rtol * (np.diag(matrix) - upper)))
    )
    passed = energy_passed if energy_acceptance else native_passed
    if energy_passed is False:
        passed = False
    report = dict(
        status="accepted"
        if passed
        else "energy_target_rejected"
        if energy_passed is False
        else "fresh_residual_rejected",
        accepted=bool(passed),
        arithmetic="float64",
        coefficient_source=coefficient_source,
        field_reconstruction="compatible_projection_float64"
        if kind == "gani"
        else "nodal_float64",
        six_loads=True,
        all_cross_products=True,
        C=matrix,
        C_stress=stress,
        G0=g0,
        energy_error_bound=bracket,
        spectral_bounds=bounds,
        residuals=norms,
        rhs_norms=rhs,
        thresholds=thresholds.tolist(),
        native_residual_passed=native_passed,
        requested_tensor_rtol=tensor_rtol,
        tensor_target_passed=energy_passed,
        certified=False,
        difference_from_production=float(
            np.linalg.norm(matrix - production) / np.linalg.norm(matrix)
        ),
        symmetry_defect=defect,
        **detail,
    )
    report["load_audits"] = [
        dict(
            load_id=a,
            metric=detail["metric"],
            residual=norms[a],
            rhs_norm=rhs[a],
            threshold=float(thresholds[a]),
            evaluation="float64",
            criterion="energy" if energy_acceptance else "fresh_residual",
            native_residual_passed=bool(norms[a] <= thresholds[a]),
            energy_target_passed=None
            if tensor_rtol is None
            else bool(upper[a] <= tensor_rtol * (matrix[a, a] - upper[a])),
        )
        for a in range(6)
    ]
    report["elapsed_s"] = time.perf_counter() - started
    if not passed:
        raise TensorVerificationError(report)
    return report
