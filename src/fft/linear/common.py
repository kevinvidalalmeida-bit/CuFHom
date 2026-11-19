"""Operator-independent contracts for periodic SPD elastic homogenization.

These host-side services run at preparation/acceptance boundaries, never inside
generated FFTs, CUDA graphs or specialized Krylov iterations. Residual metrics
remain explicitly operator dependent; a shared tolerance value is not an error
equivalence between discretizations.
"""

import operator
from dataclasses import dataclass

import numpy as np
from cufhom.fft.krylov import NonlinearConvergenceError

MANDEL_ORDER = ("11", "22", "33", "23", "13", "12")
SYM21_PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))


class LinearSolveError(NonlinearConvergenceError):
    """Explicit numerical termination; never an accepted load."""

    def __init__(self, message, *, status, iterations=0, diagnostics=None):
        super().__init__(message)
        self.status = status
        self.iterations = int(iterations)
        self.diagnostics = {} if diagnostics is None else diagnostics


class WorkingPrecisionError(ValueError):
    """Valid input that cannot be used safely by this working precision."""

    status = "invalid_working_operator"
    iterations = 0


class FreshProgress:
    """Bound repeated non-improving *fresh* inspections, independently of CG."""

    def __init__(self):
        self.best, self.stalled = np.inf, 0

    def check(self, value, iterations):
        """Update fresh-residual progress or raise on nonfinite values or stagnation."""
        if not np.isfinite(value) or value < 0:
            raise LinearSolveError(
                "Nonfinite fresh residual", status="nonfinite", iterations=iterations
            )
        self.stalled = self.stalled + 1 if value > 0.9 * self.best else 0
        self.best = min(self.best, value)
        if self.stalled >= 5:
            raise LinearSolveError(
                "Fresh residual stalled above tolerance (attainable accuracy)",
                status="precision_limit",
                iterations=iterations,
                diagnostics=dict(best=self.best, fresh=value, stalled=self.stalled),
            )


def energy_tolerance(value):
    """Validate before allocating or compiling a solver, including native calls."""
    if value is None:
        return None
    a = np.asarray(value)
    if (
        a.ndim
        or a.dtype.kind not in "iuf"
        or not np.isfinite(a)
        or not 0 < float(a) < 1
    ):
        raise ValueError("energy tolerance must be a finite scalar in (0, 1)")
    return float(a)


def working_material_check(stiffness, dtype):
    """Conservative material-rounding screen, not a global operator certificate.

    HEX8 subsequently rounds integrated element blocks, a distinct perturbation.
    Reject rows whose positivity is below a normwise rounding margin; users can
    explicitly select FP64. No coefficients are repaired or regularized.
    """
    dtype = np.dtype(dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.asarray(stiffness).astype(dtype).astype(np.float64)
    if not np.isfinite(rounded).all():
        raise WorkingPrecisionError(
            "Working material is nonfinite after rounding; use working=float64"
        )
    values = np.linalg.eigvalsh(rounded)
    margin = (
        12 * np.finfo(dtype).eps * np.linalg.norm(rounded, ord=np.inf, axis=(-2, -1))
    )
    if np.any(values[:, 0] <= np.maximum(margin, 32 * np.finfo(dtype).tiny)):
        raise WorkingPrecisionError(
            "Working material is not safely positive definite after rounding; use working=float64"
        )
    return dict(
        minimum_eigenvalue=float(values[:, 0].min()),
        maximum_condition=float(np.max(values[:, -1] / values[:, 0])),
        material_rounding_checked=True,
        global_operator_certified=False,
    )


def energy_report(matrix, bounds, indicators, *, fallback_upper=None):
    """Algebraic field-energy estimates, explicitly excluding output rounding.

    Missing indicators are JSON null, never NaN or a fabricated zero. GaNi may
    supply the independent projected-stress upper bound when G0 is unavailable.
    """
    g0, lower, upper, methods = [], [], [], []
    for a, value in enumerate(indicators):
        if value is not None:
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise LinearSolveError(
                    "Invalid fresh energy indicator", status="nonfinite"
                )
            g0.append(value)
            lower.append(value / bounds["gamma_plus"])
            upper.append(value / bounds["gamma_minus"])
            methods.append("fresh_G0")
        else:
            g0.append(None)
            limit = None if fallback_upper is None else float(fallback_upper[a])
            if limit is not None and (not np.isfinite(limit) or limit < 0):
                raise LinearSolveError(
                    "Invalid projected-stress energy estimate", status="nonfinite"
                )
            lower.append(0.0 if limit is not None else None)
            upper.append(limit)
            methods.append(
                "projected_stress_upper" if limit is not None else "unavailable"
            )
    complete = all(x is not None for x in upper)
    trace = float(sum(upper)) if complete else None
    return dict(
        G0=g0,
        lower=lower,
        upper=upper,
        available=complete,
        method=methods,
        relative_upper=[
            None if x is None else float(x / matrix[a, a]) for a, x in enumerate(upper)
        ],
        matrix_2norm_upper=trace,
        relative_matrix_upper=None
        if trace is None
        else float(trace / np.linalg.norm(matrix, 2)),
        scope="algebraic_error_of_exact_field_energy",
        operator="fresh residual of the accepted solution, this discretization",
        full_G0_matrix_available=False,
        material_rounding_included=False,
        evaluation_rounding_included=False,
        assembly_rounding_included=False,
        retained_solution_rounding_included=False,
        returned_matrix_certified=False,
        certified=False,
    )


def elastic_matrix(value, *, name="stiffness"):
    """Validate and symmetrize a finite positive-definite 6-by-6 Mandel matrix."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    c = np.asarray(value, dtype=np.float64)
    if c.shape != (6, 6) or not np.isfinite(c).all():
        raise ValueError(f"{name} must be a finite 6x6 Mandel matrix")
    scale = np.max(np.abs(c))
    if np.max(np.abs(c - c.T)) > 1e-10 * scale:
        raise ValueError(f"{name} must be symmetric")
    c = 0.5 * c + 0.5 * c.T
    if np.linalg.eigvalsh(c)[0] <= 0:
        raise ValueError(f"{name} must be positive definite")
    return c


@dataclass(frozen=True, init=False)
class IndexedMaterial:
    """Owned, immutable CPU image and SPD Mandel tables, shared by both plans.

    ``index`` is a three-dimensional integer array of phase labels;
    ``stiffness`` contains one SPD 6x6 Mandel matrix or packed 21-entry row
    per label. ``dtype`` selects FP32 or FP64 storage.

    Coefficients are rounded exactly once to dtype. Full and packed tables
    describe that same material; neither operator groups orientations or
    otherwise prepares a different coefficient image.
    """

    index: np.ndarray
    stiffness: np.ndarray
    packed: np.ndarray
    dtype: np.dtype

    def __init__(self, index, stiffness, *, dtype="float64"):
        dt = np.dtype(dtype)
        if dt not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        labels = np.asarray(index)
        if labels.ndim != 3 or min(labels.shape) < 2 or labels.dtype.kind not in "iu":
            raise ValueError(
                "index must be a 3D integer array, at least two cells per axis"
            )
        if np.iscomplexobj(stiffness):
            raise ValueError("stiffness must be real")
        c = np.asarray(stiffness, dtype=np.float64)
        if c.ndim == 2 and c.shape[1] == 21:
            full = np.zeros((len(c), 6, 6))
            for k, (i, j) in enumerate(SYM21_PAIRS):
                full[:, i, j] = full[:, j, i] = c[:, k]
            c = full
        if c.ndim != 3 or c.shape[1:] != (6, 6) or len(c) == 0:
            raise ValueError("stiffness must have shape (P,6,6) or (P,21)")
        if (
            labels.min() < 0
            or labels.max() >= len(c)
            or len(c) > np.iinfo(np.int32).max
        ):
            raise ValueError("index contains labels outside the stiffness table")
        with np.errstate(over="ignore", invalid="ignore"):
            c = np.stack([elastic_matrix(row) for row in c]).astype(dt)
        if not np.isfinite(c).all() or np.any(
            np.linalg.eigvalsh(c.astype(float))[:, 0] <= 0
        ):
            raise ValueError("stiffness must remain SPD in the requested precision")
        labels = np.array(labels, dtype=np.int32, order="C", copy=True)
        c = np.ascontiguousarray(c)
        packed = np.stack([c[:, i, j] for i, j in SYM21_PAIRS], axis=1)
        for value in (labels, c, packed):
            value.flags.writeable = False
        object.__setattr__(self, "index", labels)
        object.__setattr__(self, "stiffness", c)
        object.__setattr__(self, "packed", packed)
        object.__setattr__(self, "dtype", dt)

    @property
    def shape(self):
        """Return the three spatial dimensions of the owned material index array."""
        return self.index.shape


@dataclass(frozen=True)
class SolveOptions:
    """Shared input contract; absolute tolerances use the reported native norm."""

    backend: str = "cupy"
    rtol: float = 1e-6
    atol: float = 0.0
    maxiter: int = 3000
    storage: str = "auto"

    def __post_init__(self):
        """Validate solver tolerances, backend and iteration budget."""
        if self.backend not in ("numpy", "cupy"):
            raise ValueError("backend must be numpy or cupy")
        if (
            not np.isfinite([self.rtol, self.atol]).all()
            or min(self.rtol, self.atol) < 0
            or self.rtol + self.atol == 0
        ):
            raise ValueError("Tolerances must be finite, nonnegative and not both zero")
        try:
            budget = operator.index(self.maxiter)
        except TypeError as error:
            raise ValueError("maxiter must be a positive integer") from error
        if budget < 1 or isinstance(self.maxiter, (bool, np.bool_)):
            raise ValueError("maxiter must be a positive integer")
        if self.storage not in ("auto", "device", "host"):
            raise ValueError("storage must be auto, device or host")

    def threshold(self, rhs_norm):
        """Return the larger absolute or relative threshold for a validated RHS norm."""
        rhs_norm = float(rhs_norm)
        if not np.isfinite(rhs_norm) or rhs_norm < 0:
            raise ValueError("RHS norm must be finite and nonnegative")
        return max(float(self.atol), float(self.rtol) * rhs_norm)


def cell_lengths(lengths):
    """Return three positive finite physical cell lengths in X, Y, Z order."""
    if np.iscomplexobj(lengths):
        raise ValueError("lengths must be real")
    values = np.asarray(lengths, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("lengths must be three positive finite X,Y,Z values")
    return tuple(float(v) for v in values)


def isotropic_reference(stiffness, reference=None):
    """GaNi's supported reference: isotropic, without changing physical C."""
    c = (
        np.asarray(stiffness, float).mean(0)
        if reference is None
        else elastic_matrix(reference, name="reference")
    )
    bulk = float(c[:3, :3].sum() / 9)
    shear = float((np.trace(c) - 3 * bulk) / 10)
    iso = 2 * shear * np.eye(6)
    iso[:3, :3] += bulk - 2 * shear / 3
    if reference is not None and not np.allclose(
        iso, c, rtol=1e-10, atol=1e-12 * np.max(np.abs(c))
    ):
        raise ValueError(
            "GaNi currently requires an isotropic reference; physical material may be fully anisotropic"
        )
    if min(bulk, shear) <= 0:
        raise ValueError("Reference bulk and shear moduli must be positive")
    return iso, bulk, shear


def reference_bounds(stiffness, reference):
    """Numerical material-relative spectral bounds, not interval certificates.

    One reference Cholesky serves every phase; no grid-dependent matrices are
    assembled. Bounds apply to each discretization on its admissible space.
    """
    c0 = elastic_matrix(reference, name="reference")
    c = np.asarray(stiffness, dtype=float)
    root = np.linalg.cholesky(c0)
    inverse = np.linalg.solve(root, np.eye(6))
    relative = inverse @ c @ inverse.T
    values = np.linalg.eigvalsh((relative + relative.transpose(0, 2, 1)) / 2)
    lower, upper = float(values.min()), float(values.max())
    if not np.isfinite([lower, upper]).all() or lower <= 0:
        raise ValueError("Material-relative spectral bounds lost positive definiteness")
    return dict(
        gamma_minus=lower,
        gamma_plus=upper,
        condition_bound=upper / lower,
        rigorous_rounding_certificate=False,
    )


_LABEL_COUNTS = {}
_LABEL_COUNTS_SOURCE = r"""
extern "C" __global__ void label_counts(const ID* ids, unsigned long long* out, long long n, int rows){
  extern __shared__ unsigned int h[];
  for(int i=threadIdx.x;i<rows;i+=blockDim.x) h[i]=0;
  __syncthreads();
  for(long long i=blockIdx.x*(long long)blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
    atomicAdd(&h[(int)ids[i]],1u);
  __syncthreads();
  for(int i=threadIdx.x;i<rows;i+=blockDim.x) if(h[i]) atomicAdd(&out[i],(unsigned long long)h[i]);
}
"""


def device_label_counts(ids, rows):
    """Host int64 voxel count of each label of validated device ids in [0, rows).

    One shared-memory histogram per block (cp.bincount re-validates the ids
    and serializes on a few global bins: about 4 ms at 192^3 instead of 0.1).
    """
    from cufhom_core.general.cupy_capacity import cp

    flat = ids.ravel()
    names = {
        np.dtype(np.int32): "int",
        np.dtype(np.uint16): "unsigned short",
        np.dtype(np.uint8): "unsigned char",
    }
    if np.dtype(flat.dtype) not in names or rows > 12288 or not flat.flags.c_contiguous:
        return cp.asnumpy(cp.bincount(flat, minlength=rows)).astype(np.int64)
    key = np.dtype(flat.dtype)
    if key not in _LABEL_COUNTS:
        _LABEL_COUNTS[key] = cp.RawKernel(
            _LABEL_COUNTS_SOURCE.replace("ID", names[key]), "label_counts"
        )
    out = cp.zeros(rows, cp.uint64)
    blocks = int(
        min(
            4 * cp.cuda.Device().attributes["MultiProcessorCount"],
            max(1, -(-flat.size // 512)),
        )
    )
    _LABEL_COUNTS[key](
        (blocks,),
        (512,),
        (flat, out, np.int64(flat.size), np.int32(rows)),
        shared_mem=4 * rows,
    )
    return cp.asnumpy(out).astype(np.int64)


def device_free_bytes():
    """Free device memory plus the unused blocks cached by the current CuPy pool.

    Residency decisions must not depend on the cache left by an earlier call:
    the pool reuses those blocks and releases them before an allocation fails.
    """
    from cufhom_core.general.cupy_capacity import cp, current_memory_pool

    pool = current_memory_pool()
    return int(cp.cuda.runtime.memGetInfo()[0]) + int(
        pool.total_bytes() - pool.used_bytes()
    )


def audit_record(
    *, metric, residual, rhs_norm, threshold, iterations, accepted, native
):
    """Normalize metadata, retaining the native metric and acceptance decision."""
    residual, rhs_norm, threshold = map(float, (residual, rhs_norm, threshold))
    valid = (
        bool(accepted)
        and np.isfinite([residual, rhs_norm, threshold]).all()
        and min(residual, rhs_norm, threshold) >= 0
        and residual <= threshold
    )
    if not valid:
        raise RuntimeError(
            f"{metric} fresh residual rejected: {residual} > {threshold}"
        )
    return dict(
        metric=metric,
        residual=residual,
        rhs_norm=rhs_norm,
        threshold=threshold,
        relative=residual / rhs_norm if rhs_norm else None,
        iterations=int(iterations),
        accepted=True,
        native=native,
    )


def checked_energy_matrix(matrix, *, symmetry_rtol=1e-7):
    """Validate the finite, symmetric positive-definite effective energy matrix."""
    c = np.asarray(matrix, dtype=np.float64)
    if c.shape != (6, 6) or not np.isfinite(c).all():
        raise RuntimeError("Effective energy matrix must be finite and 6x6")
    scale = np.max(np.abs(c))
    if scale == 0:
        raise RuntimeError("Effective energy matrix must be positive definite")
    scaled = c / scale
    defect = float(np.linalg.norm(scaled - scaled.T) / np.linalg.norm(scaled))
    if defect > symmetry_rtol:
        raise RuntimeError(
            f"Effective energy matrix symmetry defect {defect} exceeds {symmetry_rtol}"
        )
    c = 0.5 * c + 0.5 * c.T
    if np.linalg.eigvalsh(c)[0] <= 0:
        raise RuntimeError("Effective energy matrix must be positive definite")
    return c, defect


def stored_planes(n2):
    """Return the number of k2 planes in the fused spectral layout.

    k2 planes of the fused layout: n2/2 on even grids, whose z-Nyquist plane
    is packed into the k2 = 0 plane, and all (n2+1)/2 on odd grids.
    """
    return n2 // 2 if n2 % 2 == 0 else (n2 + 1) // 2
