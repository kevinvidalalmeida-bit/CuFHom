"""Shared conjugate gradients with an audit of the true residual."""

import numpy as np
from cufhom_core.tensors.fft import cp


class NonlinearConvergenceError(RuntimeError):
    """No unconverged increment is committed or exported as a curve point."""


_CUPY_CG_UPDATE_KERNELS = None


def _cg_updates():
    global _CUPY_CG_UPDATE_KERNELS
    if _CUPY_CG_UPDATE_KERNELS is None:
        _CUPY_CG_UPDATE_KERNELS = (
            cp.ElementwiseKernel(
                "T p, T ap, float64 alpha",
                "T x, T r",
                "const T a=(T)alpha; x += a*p; r -= a*ap;",
                "nonlinear_cg_update_xr",
            ),
            cp.ElementwiseKernel(
                "T r, float64 beta",
                "T p",
                "p = r + (T)beta*p;",
                "nonlinear_cg_update_p",
            ),
        )
    return _CUPY_CG_UPDATE_KERNELS


def _cg(
    action,
    rhs,
    dot,
    tolerance,
    maxiter,
    xp,
    project=None,
    fused=False,
    precondition=None,
    initial=None,
):
    """CG in range(P); action may be C or P C for a self-adjoint C.

    The Galerkin variant uses p:C:p and r <- P(r-alpha C:p), rather than
    p:P(C:p) and r <- P(r-alpha P(C:p)). Since r,p belong to range(P), both
    solve the SAME constrained problem, but the former saves one FFT pair.
    The true residual is always P(rhs-C:x), never the unprojected stress.
    """
    x = xp.zeros_like(rhs) if initial is None else initial.copy()
    r = rhs.copy() if initial is None else rhs - action(x)
    if project is not None:
        r = project(r)
    rr = dot(r, r)
    limit = tolerance * tolerance * (rr if initial is None else dot(rhs, rhs))
    if rr == 0:
        return x, 0
    z = precondition(r) if precondition is not None else r
    rz = dot(r, z) if precondition is not None else rr
    p = z.copy()
    updates = _cg_updates() if fused and xp is not np else None
    for i in range(maxiter):
        # p is a combination of projected residuals. Projecting r each iteration
        # prevents null-space drift without a second redundant FFT pair for p.
        ap = action(p)
        pap = dot(p, ap)
        if not np.isfinite(pap) or pap <= 0:
            raise NonlinearConvergenceError(
                f"CG lost positive curvature at {i}: pAp={pap}, rr={rr}"
            )
        if not np.isfinite(rz) or rz <= 0:
            raise NonlinearConvergenceError("CG preconditioner lost positivity")
        alpha = rz / pap
        if updates is not None:
            updates[0](p, ap, np.float64(alpha), x, r)
        else:
            x += alpha * p
            r -= alpha * ap
        if project is not None:
            r = project(r)
        new = dot(r, r)
        if not np.isfinite(new):
            raise NonlinearConvergenceError("Nonfinite CG residual")
        if new <= limit:
            if project is not None:
                x = project(x)
            r = rhs - action(x)  # Audit the true rather than recursive residual.
            if project is not None:
                r = project(r)
            new = dot(r, r)
            if new <= limit:
                return x, i + 1
            z = precondition(r) if precondition is not None else r
            next_rz = dot(r, z) if precondition is not None else new
            p = z.copy()  # Reliable restart, retaining the current solution.
        else:
            z = precondition(r) if precondition is not None else r
            next_rz = dot(r, z) if precondition is not None else new
            if updates is not None:
                updates[1](z, np.float64(next_rz / rz), p)
            else:
                p *= next_rz / rz
                p += z
        rr = new
        rz = next_rz
    raise NonlinearConvergenceError(
        f"CG failed true residual after {maxiter} iterations"
    )
