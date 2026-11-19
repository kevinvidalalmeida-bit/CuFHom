"""Nonsymmetric Krylov solvers in a real Hilbert space: BiCGStab and GMRES.

Complex coordinate arrays are treated as pairs of real values: coefficients,
Hessenberg matrices and Givens rotations are real. Operators may be real-linear
(including conjugation), and need not be symmetric or complex-linear.

nonsymmetric_solve is the Newton default: BiCGStab (short recurrence, five
work fields, no restart) and, only for a solve whose BiCGStab breaks down or
exhausts its budget, restarted GMRES. GMRES(12) loses most of its Krylov
information at each restart on the plastic tangents of the paraboloidal and
nonassociated Drucker-Prager laws; BiCGStab needs about half the operator
applications there, with a third of the storage and of the synchronizations.
"""

import operator

import numpy as np
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom_core.tensors.fft import cp

_BATCH_DOTS = {}
_BASIS_UPDATES = {}
_WIDE_NORMALIZE = None
_GRAM_KERNELS = {}
_BICGSTAB_KERNELS = {}


def _normalize(value, norm, out):
    """Normalize without narrowing the FP64 norm or widening a whole field.

    A normal, representable reciprocal takes one native field multiply.
    Extreme reciprocals instead use FP64 scalar division, with each result
    rounded directly into the field dtype. In particular, a subnormal FP32
    reciprocal must not quantize an otherwise well-scaled unit vector.
    """
    global _WIDE_NORMALIZE
    if not np.isfinite(norm) or norm <= 0:
        raise NonlinearConvergenceError("Invalid GMRES normalization norm")
    xp = _backend(value)
    real_dtype = value.real.dtype
    limits = np.finfo(real_dtype)
    with np.errstate(over="ignore", under="ignore"):
        inverse = np.float64(1.0) / norm
    if limits.tiny <= inverse <= limits.max:
        xp.multiply(value, real_dtype.type(inverse), out=out)
        return
    # Work componentwise for complex fields too: no complex division and no
    # full-field FP64 temporary. NumPy's widened ufunc uses bounded buffers.
    source, target = value.view(real_dtype), out.view(real_dtype)
    if xp is np:
        np.divide(source, np.float64(norm), out=target, dtype=np.float64)
        return
    if _WIDE_NORMALIZE is None:
        _WIDE_NORMALIZE = cp.ElementwiseKernel(
            "T value, float64 norm",
            "T result",
            "result = (T)((double)value / norm);",
            "gmres_normalize_wide",
        )
    _WIDE_NORMALIZE(source, np.float64(norm), target)


def _basis_update(basis, coefficients, target, subtract):
    """Fuse a whole Arnoldi projection/combination into one device pass.

    Separate rounded multiply and add/subtract preserve the scalar update
    order. No basis-sized product temporary and no launch per basis vector.
    """
    xp = _backend(target)
    scalar = (
        np.float32
        if target.dtype in (np.dtype("float32"), np.dtype("complex64"))
        else np.float64
    )
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise NonlinearConvergenceError("Nonfinite GMRES update coefficients")
    # Large least-squares/projection coefficients can exceed FP32 even when
    # every coefficient*basis component fits. Subnormal coefficients likewise
    # need their FP64 value retained until after the product is evaluated.
    magnitude = np.abs(coefficients)
    limits = np.finfo(scalar)
    wide = scalar is np.float32 and bool(
        np.any(
            (magnitude > limits.max) | ((magnitude != 0) & (magnitude < limits.tiny))
        )
    )
    if xp is np:
        product = np.empty_like(target) if wide else None
        for row, c in zip(basis, coefficients):
            if wide:
                np.multiply(
                    row.view(scalar), c, dtype=np.float64, out=product.view(scalar)
                )
                delta = product
            else:
                delta = scalar(c) * row
            if subtract:
                target -= delta
            else:
                target += delta
            del delta
        if wide and not bool(np.isfinite(target).all()):
            raise NonlinearConvergenceError("Unrepresentable GMRES basis update")
        return
    key = (np.dtype(scalar).str, wide)
    if key not in _BASIS_UPDATES:
        kind = "f" if scalar is np.float32 else "d"
        real = "float" if kind == "f" else "double"
        coefficient_type = "double" if wide else real
        product = (
            f"({real})__dmul_rn(coeff[i],(double)basis[(long long)i*n+k])"
            if wide
            else f"__{kind}mul_rn(coeff[i],basis[(long long)i*n+k])"
        )
        code = f"""
extern "C" __global__ void basis_update(const {real}* basis,const {coefficient_type}* coeff,
 {real}* value,long long n,int rows,int subtract){{
 const long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
 {real} v=value[k];
 for(int i=0;i<rows;i++){{
   const {real} delta={product};
   v=subtract?__{kind}sub_rn(v,delta):__{kind}add_rn(v,delta);
 }}
 value[k]=v;
}}
"""
        _BASIS_UPDATES[key] = cp.RawKernel(code, "basis_update")
    real_target = target.view(scalar).reshape(-1)
    c = cp.asarray(coefficients, dtype=np.float64 if wide else scalar)
    _BASIS_UPDATES[key](
        ((real_target.size + 255) // 256,),
        (256,),
        (
            basis,
            c,
            real_target,
            np.int64(real_target.size),
            np.int32(len(coefficients)),
            np.int32(subtract),
        ),
    )
    if wide and not bool(cp.isfinite(target).all()):
        raise NonlinearConvergenceError("Unrepresentable GMRES basis update")


def _backend(field):
    if isinstance(field, np.ndarray):
        return np
    if cp is not None and isinstance(field, cp.ndarray):
        return cp
    raise TypeError("GMRES requires a NumPy or CuPy array")


def real_inner_products(basis, value):
    """Batched Re(sum(conj(v_i)*value)), with FP64 products/accumulation.

    Returns a backend array of len(basis) scalars. The GPU reduction broadcasts
    value across basis rows and reduces without allocating a basis-sized FP64
    product matrix or invoking cuBLAS. No field is transferred to the host.
    """
    xp = _backend(value)
    if (
        _backend(basis) is not xp
        or basis.shape[1:] != value.shape
        or basis.dtype != value.dtype
    ):
        raise ValueError(
            "GMRES inner-product fields must have matching backend, shape and dtype"
        )
    if value.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("complex64"),
        np.dtype("complex128"),
    ):
        raise ValueError("GMRES requires float32/64 or complex64/128")
    real_dtype = value.real.dtype
    rows = xp.ascontiguousarray(basis).view(real_dtype).reshape(basis.shape[0], -1)
    vector = xp.ascontiguousarray(value).view(real_dtype).reshape(1, -1)
    if xp is np:
        return np.einsum(
            "ij,ij->i", rows, np.broadcast_to(vector, rows.shape), dtype=np.float64
        )
    n = vector.size
    if n == 0 or len(basis) == 0:
        return cp.zeros(len(basis), cp.float64)
    key = (value.device.id, np.dtype(real_dtype).str)
    if key not in _BATCH_DOTS:
        real = "float" if real_dtype == np.dtype("float32") else "double"
        code = f"""
extern "C" __global__ void gram(const {real}* v,const {real}* w,double* partial,long long n){{
 const int row=blockIdx.y;double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
   sum+=(double)v[(long long)row*n+i]*(double)w[i];
 for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
 __shared__ double warps[8];
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)warps[warp]=sum;
 __syncthreads();
 if(warp==0){{
   sum=lane<8?warps[lane]:0;
   for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
   if(lane==0)partial[(long long)row*gridDim.x+blockIdx.x]=sum;
 }}
}}
"""
        _BATCH_DOTS[key] = cp.RawKernel(code, "gram")
    # A long dot gets many blocks, even with only 1..12 Arnoldi rows.
    # The small second reduction avoids the low-occupancy full-row reduction.
    blocks = min(1024, (n + 255) // 256)
    partial = cp.empty((len(basis), blocks), cp.float64)
    _BATCH_DOTS[key]((blocks, len(basis)), (256,), (rows, vector, partial, np.int64(n)))
    return cp.sum(partial, axis=1, dtype=cp.float64)


def euclidean_dot(a, b):
    """Real Euclidean inner product, including both parts of complex arrays."""
    return float(real_inner_products(a[None, ...], b)[0])


# Optional protocol for custom dots: dot.gmres_batch(basis, value) must use
# EXACTLY the same inner product as dot(a,b). A custom/weighted dot without this
# protocol remains valid and uses scalar calls, rather than being bypassed.
euclidean_dot.gmres_batch = real_inner_products


def _norm_squared(dot, value, *, reject_underflow=False):
    result = float(dot(value, value))
    if not np.isfinite(result) or result < 0:
        raise NonlinearConvergenceError("Nonfinite or negative GMRES squared norm")
    if reject_underflow and result == 0 and _nonzero_storage(value):
        raise NonlinearConvergenceError("GMRES squared norm underflowed")
    return result


def _nonzero_storage(value):
    """Detect nonzero bits even when device arithmetic flushes subnormals.

    Called only after a zero norm. Floating comparisons/conversion can flush
    an FP32 subnormal on a device, so neither an arithmetic norm nor value!=0
    alone distinguishes stored subnormals from an exact signed-zero field.
    """
    xp = _backend(value)
    if xp is np:
        return bool(np.any(value != 0))
    bits = value.view(cp.uint32 if value.real.dtype == np.float32 else cp.uint64)
    mask = (
        np.uint32(0x7FFFFFFF)
        if value.real.dtype == np.float32
        else np.uint64(0x7FFFFFFFFFFFFFFF)
    )
    return bool(cp.any((bits & mask) != 0))


def _inner_products(dot, basis, value, xp):
    batch = getattr(dot, "gmres_batch", None)
    if batch is None:
        result = np.array([float(dot(v, value)) for v in basis], dtype=np.float64)
    else:
        result = batch(basis, value)
        # Only the small coefficient vector crosses the device/host boundary.
        if xp is not np and isinstance(result, cp.ndarray):
            if result.shape != (len(basis),) or result.dtype.kind == "c":
                raise ValueError(
                    "dot.gmres_batch must return one real scalar per basis vector"
                )
            result = cp.asnumpy(result)
        result = np.asarray(result)
        if result.shape != (len(basis),) or np.iscomplexobj(result):
            raise ValueError(
                "dot.gmres_batch must return one real scalar per basis vector"
            )
        result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise NonlinearConvergenceError(
            "Nonfinite GMRES orthogonalization coefficients"
        )
    return result


def _back_substitute(upper, right):
    """Small real triangular solve on the CPU; never starts BLAS/cuBLAS."""
    y = right.copy()
    for i in range(len(y) - 1, -1, -1):
        diagonal = upper[i, i]
        if not np.isfinite(diagonal) or diagonal == 0:
            raise NonlinearConvergenceError(
                "GMRES breakdown: singular Hessenberg triangular factor"
            )
        for j in range(i + 1, len(y)):
            y[i] -= upper[i, j] * y[j]
        y[i] /= diagonal
    if not np.isfinite(y).all():
        raise NonlinearConvergenceError("Nonfinite GMRES least-squares solution")
    return y


def gmres(action, rhs, dot, tolerance, maxiter, precondition=None, restart=12):
    """Solve A x = rhs with restarted, twice-reorthogonalized right GMRES.

    dot defines a positive real inner product. For complex fields it must use
    Re(<a,b>), not a complex coefficient; all combinations are real. Prefer
    euclidean_dot for GPU batched reductions, or supply dot.gmres_batch with
    the same metric. Other dots use the supplied scalar callback throughout.

    action and precondition must be fixed, linear over the reals, preserve the
    field's shape/dtype/backend, and never mutate their argument. Precondition
    applies P = M^-1 on the right. Only V is stored: P(sum(y_i V_i)) is
    recomputed at each restart/audit instead of retaining a second basis Z.
    A variable/nonlinear preconditioner requires flexible GMRES, not this API.

    restart bounds basis storage at restart+1 fields (12 is intended for
    displacement coordinates, not six-component strain fields). Work storage
    is a constant number of fields; Hessenberg/Givens data are FP64 host
    arrays. Count is Arnoldi operator applications, excluding true-residual
    audits. maxiter is a total budget across restarts, not per restart.

    Norms and extreme coefficients retain FP64 scalar range without storing
    an FP64 field. Ordinary normalization uses a native-precision reciprocal;
    extreme scales divide components by the FP64 norm directly. Nonzero
    fields whose squared norm underflows FP64 are rejected, including audits.

    Every successful nonzero solve passes ||rhs-Ax|| <= tolerance*||rhs||
    using dot. Estimated convergence or happy breakdown alone never accepts.
    Exact zero rhs returns zero without applying A. Failure raises
    NonlinearConvergenceError and does not return an unconverged correction.
    """
    tolerance = float(tolerance)
    maxiter, restart = operator.index(maxiter), operator.index(restart)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("GMRES tolerance must be finite and between zero and one")
    if maxiter < 1 or restart < 1:
        raise ValueError("GMRES maxiter and restart must be positive")
    xp = _backend(rhs)
    if rhs.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("complex64"),
        np.dtype("complex128"),
    ):
        raise ValueError("GMRES requires float32/64 or complex64/128")
    rhs = xp.ascontiguousarray(rhs)
    x = xp.zeros_like(rhs)
    rhs_squared = _norm_squared(dot, rhs, reject_underflow=True)
    if rhs_squared == 0:
        return x, 0
    target = tolerance * np.sqrt(rhs_squared)
    real_dtype = np.dtype(
        "float32"
        if rhs.dtype in (np.dtype("float32"), np.dtype("complex64"))
        else "float64"
    )
    eps = np.finfo(real_dtype).eps
    width = min(restart, maxiter)
    basis = xp.empty((width + 1,) + rhs.shape, dtype=rhs.dtype)
    residual = rhs.copy()
    residual_squared = rhs_squared
    count = 0

    def checked(value):
        if (
            _backend(value) is not xp
            or value.shape != rhs.shape
            or value.dtype != rhs.dtype
        ):
            raise ValueError(
                "GMRES operator/preconditioner must preserve backend, shape and dtype"
            )
        return value

    def right(value):
        return value if precondition is None else checked(precondition(value))

    while count < maxiter:
        beta = np.sqrt(residual_squared)
        _normalize(residual, beta, basis[0])
        # The true residual is no longer needed until the next audit.
        del residual
        columns = min(width, maxiter - count)
        h = np.zeros((columns + 1, columns), dtype=np.float64)
        cosine, sine = np.zeros(columns), np.zeros(columns)
        g = np.zeros(columns + 1)
        g[0] = beta
        for j in range(columns):
            z = right(basis[j])
            # Own w: orthogonalization must not modify an operator's output
            # aliasing rhs, x or a basis vector (e.g. the identity operator).
            w = xp.array(checked(action(z)), order="C", copy=True)
            del z
            count += 1
            before = np.sqrt(_norm_squared(dot, w))
            # Two complete classical Gram-Schmidt passes, each with one batch
            # of inner products, followed by field updates on the device.
            for _ in range(2):
                coefficients = _inner_products(dot, basis[: j + 1], w, xp)
                h[: j + 1, j] += coefficients
                _basis_update(basis[: j + 1], coefficients, w, True)
            next_norm = np.sqrt(_norm_squared(dot, w))
            h[j + 1, j] = next_norm
            breakdown = next_norm <= eps * before

            for i in range(j):
                a, b = h[i, j], h[i + 1, j]
                h[i, j] = cosine[i] * a + sine[i] * b
                h[i + 1, j] = -sine[i] * a + cosine[i] * b
            diagonal = np.hypot(h[j, j], h[j + 1, j])
            if not np.isfinite(diagonal) or diagonal == 0:
                # No solution of the current projected problem. In particular,
                # A=0 with rhs!=0 is breakdown, not happy convergence.
                raise NonlinearConvergenceError(
                    "GMRES breakdown: singular Hessenberg triangular factor"
                )
            cosine[j], sine[j] = h[j, j] / diagonal, h[j + 1, j] / diagonal
            h[j, j], h[j + 1, j] = diagonal, 0.0
            g[j + 1], g[j] = -sine[j] * g[j], cosine[j] * g[j]
            if abs(g[j + 1]) <= target or breakdown or j + 1 == columns:
                del w
                y = _back_substitute(h[: j + 1, : j + 1], g[: j + 1])
                combination = xp.zeros_like(rhs)
                _basis_update(basis[: j + 1], y, combination, False)
                update = right(combination)
                xp.add(x, update, out=x)
                del update, combination
                residual = rhs - checked(action(x))
                residual_squared = _norm_squared(dot, residual, reject_underflow=True)
                if np.sqrt(residual_squared) <= target:
                    return x, count
                if breakdown:
                    raise NonlinearConvergenceError(
                        f"GMRES breakdown failed true residual after {count} iterations"
                    )
                # Reliable restart with the audited residual when an estimate
                # was optimistic. Retain x and the original RHS norm/target.
                break
            # The next basis vector is consumed only within this cycle.
            # Skip its full-field normalization/write at restarts and audits.
            _normalize(w, next_norm, basis[j + 1])
            del w
    raise NonlinearConvergenceError(
        f"GMRES failed true residual after {count} iterations"
    )


_MAX_GRAM = 4


def _gram(dot, vectors):
    """Symmetric matrix of Euclidean (v_i, v_j) for up to four fields.

    One device pass reads each field once and accumulates every product in
    FP64 registers (the same products/accumulation as real_inner_products),
    then one transfer. A custom dot without the Euclidean batch protocol is
    called once per pair instead.
    """
    k = len(vectors)
    gram = np.empty((k, k))
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    first = vectors[0]
    xp = _backend(first)
    if getattr(dot, "gmres_batch", None) is not real_inner_products:
        values = [float(dot(vectors[i], vectors[j])) for i, j in pairs]
    elif xp is np:
        values = [
            float(real_inner_products(vectors[i][None], vectors[j])[0])
            for i, j in pairs
        ]
    else:
        if not 1 <= k <= _MAX_GRAM:
            raise ValueError(f"Gram reductions take one to {_MAX_GRAM} fields")
        real_dtype = first.real.dtype
        for v in vectors:
            if (
                v.dtype != first.dtype
                or v.size != first.size
                or not v.flags.c_contiguous
            ):
                raise ValueError(
                    "Gram fields must be contiguous with matching dtype and size"
                )
        n = first.size * (2 if first.dtype.kind == "c" else 1)
        key = (first.device.id, np.dtype(real_dtype).str, k)
        if key not in _GRAM_KERNELS:
            real = "float" if real_dtype == np.dtype("float32") else "double"
            P = len(pairs)
            args = ",".join(f"const {real}* v{i}" for i in range(k))
            load = "".join(f"const double x{i}=(double)v{i}[i];" for i in range(k))
            accumulate = "".join(
                f"sum[{p}]+=x{i}*x{j};" for p, (i, j) in enumerate(pairs)
            )
            _GRAM_KERNELS[key] = cp.RawKernel(
                f"""
extern "C" __global__ void gram_fields({args},double* partial,long long n){{
 double sum[{P}];for(int p=0;p<{P};p++)sum[p]=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){{
   {load}{accumulate}
 }}
 __shared__ double warps[{P}][8];
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int p=0;p<{P};p++){{
   double v=sum[p];
   for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
   if(lane==0)warps[p][warp]=v;
 }}
 __syncthreads();
 if(warp==0)for(int p=0;p<{P};p++){{
   double v=lane<8?warps[p][lane]:0;
   for(int s=16;s;s>>=1)v+=__shfl_down_sync(0xffffffff,v,s);
   if(lane==0)partial[(long long)p*gridDim.x+blockIdx.x]=v;
 }}
}}
""",
                "gram_fields",
            )
        blocks = min(1024, (n + 255) // 256)
        partial = cp.empty((len(pairs), blocks), cp.float64)
        _GRAM_KERNELS[key](
            (blocks,),
            (256,),
            tuple(v.view(real_dtype) for v in vectors) + (partial, np.int64(n)),
        )
        values = cp.asnumpy(cp.sum(partial, axis=1, dtype=cp.float64))
    for (i, j), value in zip(pairs, values):
        gram[i, j] = gram[j, i] = value
    return gram


def _bicgstab_kernels(xp):
    """Fused multi-row updates on real views: u = r - beta u and r -= alpha u."""
    if xp is np:

        def direction(r, u, beta):
            u *= -u.dtype.type(beta)
            u += r

        def descend(u, r, alpha):
            r -= r.dtype.type(alpha) * u

        return direction, descend
    if not _BICGSTAB_KERNELS:
        _BICGSTAB_KERNELS["direction"] = cp.ElementwiseKernel(
            "T r, float64 beta", "T u", "u = r - (T)beta*u;", "bicgstab_direction"
        )
        _BICGSTAB_KERNELS["descend"] = cp.ElementwiseKernel(
            "T u, float64 alpha", "T r", "r = r - (T)alpha*u;", "bicgstab_descend"
        )
    d, e = _BICGSTAB_KERNELS["direction"], _BICGSTAB_KERNELS["descend"]
    return (
        lambda r, u, beta: d(r, np.float64(beta), u),
        lambda u, r, alpha: e(u, np.float64(alpha), r),
    )


def bicgstab(
    action,
    rhs,
    dot,
    tolerance,
    maxiter,
    precondition=None,
    degree=2,
    stagnation=200,
    action_into=None,
):
    """Solve A x = rhs with right-preconditioned BiCGStab(l) (Sleijpen & Fokkema 1993).

    Each cycle takes l BiCG steps and then minimizes the residual over a
    degree-l polynomial (normal equations in FP64 on the host); l = 1 is
    BiCGStab. Degree 2 resolves the near-imaginary tangent spectra of the
    nonassociated laws on which BiCGStab's degree-1 step stagnates.

    Same contract as gmres: real inner product dot, fixed real-linear action
    and precondition (P = M^-1 on the right), count = operator applications
    (2l per cycle) excluding true-residual audits, and acceptance only after
    ||rhs - A x|| <= tolerance*||rhs||. The shadow residual is rhs itself.
    Storage: R and U with l+1 fields each and the iterate.
    action_into(z, out), when given, writes A z into a basis row directly.
    An audit that fails restarts from the true residual. Failure raises
    NonlinearConvergenceError with attributes iterations (operator
    applications spent) and best (the current iterate after M^-1 when its
    recursive residual is below ||rhs||, else None), on breakdown, a
    nonfinite scalar, no new smallest residual within stagnation
    applications, or an exhausted budget.
    """
    tolerance, maxiter = float(tolerance), operator.index(maxiter)
    degree, stagnation = operator.index(degree), operator.index(stagnation)
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("BiCGStab tolerance must be finite and between zero and one")
    if maxiter < 1 or degree < 1 or stagnation < 1:
        raise ValueError("BiCGStab maxiter, degree and stagnation must be positive")
    xp = _backend(rhs)
    if rhs.dtype not in (
        np.dtype("float32"),
        np.dtype("float64"),
        np.dtype("complex64"),
        np.dtype("complex128"),
    ):
        raise ValueError("BiCGStab requires float32/64 or complex64/128")
    rhs = xp.ascontiguousarray(rhs)
    rhs_squared = _norm_squared(dot, rhs, reject_underflow=True)
    if rhs_squared == 0:
        return xp.zeros_like(rhs), 0
    limit = tolerance**2 * rhs_squared
    real = rhs.real.dtype
    view = (lambda a: a.view(real)) if rhs.dtype.kind == "c" else (lambda a: a)
    direction, descend = _bicgstab_kernels(xp)
    cycle_degree = degree
    R = xp.zeros((cycle_degree + 1,) + rhs.shape, dtype=rhs.dtype)
    U = xp.zeros_like(R)
    y = xp.zeros_like(rhs)
    best_squared, best_count, current = rhs_squared, 0, rhs_squared
    count = 0

    def solution(value):
        return value if precondition is None else precondition(value)

    def apply(value, out):
        if action_into is not None:
            action_into(solution(value), out)
            return
        result = action(solution(value))
        if (
            _backend(result) is not xp
            or result.shape != rhs.shape
            or result.dtype != rhs.dtype
        ):
            raise ValueError(
                "BiCGStab operator/preconditioner must preserve backend, shape and dtype"
            )
        out[...] = result

    def fail(message):
        error = NonlinearConvergenceError(message)
        error.iterations = count
        error.best = solution(y) if count and current < rhs_squared else None
        return error

    def audited():
        """Return the current iterate if it passes the true-residual test, else None."""
        x = solution(y)
        true = rhs - action(x)
        return (x if _norm_squared(dot, true) <= limit else None), true

    class Breakdown(Exception):
        pass

    R[0] = rhs
    rho0, alpha, omega = 1.0, 0.0, 1.0
    try:
        while count + 2 * cycle_degree <= maxiter:
            rho0 = -omega * rho0
            for j in range(cycle_degree):
                # ||R_0||^2 rides in the same reduction: a residual that
                # converges within a cycle is accepted before its scalars,
                # now rounding noise, can pollute the iterate.
                gram = _gram(dot, [R[0], rhs] if j == 0 else [R[0], R[j], rhs])
                rho1, rr = gram[-2, -1], gram[0, 0]
                if not np.isfinite(rr) or rr < 0:
                    raise fail("Nonfinite BiCGStab residual")
                if rr <= limit:
                    x, true = audited()
                    if x is not None:
                        return x, count
                    if j == 0:
                        # Restart the recurrence from the audited residual.
                        R[0], U[0] = true, 0
                        rho0, alpha, omega = -1.0, 0.0, 1.0
                        gram = _gram(dot, [R[0], rhs])
                        rho1, rr = gram[0, 1], gram[0, 0]
                current = rr
                if j == 0 and count:
                    if rr < best_squared:
                        best_squared, best_count = rr, count
                    elif count - best_count >= stagnation:
                        raise fail(f"BiCGStab stagnated after {count} iterations")
                if not np.isfinite(rho1) or rho0 == 0 or rho1 == 0:
                    raise Breakdown("vanishing rho")
                beta = alpha * rho1 / rho0
                rho0 = rho1
                direction(view(R[: j + 1]), view(U[: j + 1]), beta)
                apply(U[j], U[j + 1])
                count += 1
                gamma = _gram(dot, [U[j + 1], rhs])[0, 1]
                if not np.isfinite(gamma) or abs(gamma) <= np.finfo(np.float64).tiny:
                    raise Breakdown("vanishing (rhat, A u)")
                alpha = rho0 / gamma
                descend(view(U[1 : j + 2]), view(R[: j + 1]), alpha)
                apply(R[j], R[j + 1])
                count += 1
                _axpy(xp, view(U[0]), alpha, view(y))
            # Minimal residual over span(R_1..R_l): Gram matrix and right side
            # in one pass over R when it fits, else one batch per row.
            if cycle_degree < _MAX_GRAM:
                gram = _gram(dot, list(R))[1:]
            else:
                gram = np.array(
                    [
                        _inner_products(dot, R, R[k], xp)
                        for k in range(1, cycle_degree + 1)
                    ]
                )
            try:
                g = np.linalg.solve(gram[:, 1:], gram[:, 0])
            except np.linalg.LinAlgError:
                raise Breakdown("singular minimal-residual system") from None
            if not np.isfinite(g).all() or g[-1] == 0:
                raise Breakdown("vanishing stabilization step")
            omega = g[-1]
            _basis_update(R[:cycle_degree], g, y, False)
            _basis_update(R[1:], g, R[0], True)
            _basis_update(U[1:], g, U[0], True)
    except Breakdown as exc:
        # A residual that vanished within a cycle also breaks the recurrence.
        x, _ = audited()
        if x is not None:
            return x, count
        raise fail(f"BiCGStab breakdown: {exc}") from None
    except NonlinearConvergenceError as exc:
        if hasattr(exc, "iterations"):
            raise
        raise fail(str(exc)) from exc
    if count:
        current = _gram(dot, [R[0]])[0, 0]
    raise fail(f"BiCGStab failed true residual after {count} iterations")


def _axpy(xp, x, a, y):
    """Update y += a x, rounding the FP64 scalar once to field precision."""
    if xp is np:
        y += y.dtype.type(a) * x
        return
    if "axpy" not in _BICGSTAB_KERNELS:
        _BICGSTAB_KERNELS["axpy"] = cp.ElementwiseKernel(
            "T x, float64 a", "T y", "y = y + (T)a*x;", "bicgstab_axpy"
        )
    _BICGSTAB_KERNELS["axpy"](x, np.float64(a), y)


def nonsymmetric_solve(
    action,
    rhs,
    dot,
    tolerance,
    maxiter,
    precondition=None,
    restart=12,
    action_into=None,
):
    """BiCGStab(2), completed by restarted GMRES within the same budget.

    GMRES minimizes the residual monotonically and reaches the loose
    Eisenstat-Walker targets on which BiCG-type recurrences can stagnate.
    It continues from BiCGStab's last iterate (a correction solve) with the
    remaining operator applications, so a hybrid solve never costs more than
    one budget. Returns (x, count, fallback).
    """
    try:
        x, count = bicgstab(
            action, rhs, dot, tolerance, maxiter, precondition, action_into=action_into
        )
        return x, count, False
    except NonlinearConvergenceError as exc:
        spent, start = exc.iterations, exc.best
    remaining = maxiter - spent
    if remaining < 1:
        raise NonlinearConvergenceError(
            f"Hybrid BiCGStab/GMRES failed true residual after {spent} iterations"
        )
    rhs_norm = np.sqrt(_norm_squared(dot, rhs))
    residual = rhs if start is None else rhs - action(start)
    residual_norm = np.sqrt(_norm_squared(dot, residual))
    if start is not None and residual_norm >= rhs_norm:
        start, residual, residual_norm = None, rhs, rhs_norm
    target = tolerance * rhs_norm / residual_norm
    if target >= 1:
        return start, spent, True
    correction, count = gmres(
        action,
        residual,
        dot,
        target,
        remaining,
        precondition=precondition,
        restart=restart,
    )
    if start is not None:
        correction += start
    return correction, spent + count, True
