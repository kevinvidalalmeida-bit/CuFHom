"""Dual-metric CG acceptance: coordinate stopping plus true physical norm.
Continue in the preconditioned coordinates if the physical norm fails.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import fused_update,_finish_smoothing
from cufhom.pipeline.nonlinear_solver import _cg_updates,NonlinearConvergenceError
from cufhom_physical_metric import Metric
def cg(action, rhs, dot, tolerance, maxiter, xp, project=None, fused=False,
             precondition=None, initial=None, action_and_dot=None, residual_smoothing=False):
    """CG/PCG with true-residual restart and optional late residual smoothing.

    Smoothing uses Euclidean orthogonality of ordinary CG residuals, so it
    requires coordinates that already contain the symmetric preconditioner.
    Its harmonic residual estimate is NEVER an acceptance test in float32.
    """
    if xp is not cp or rhs.dtype not in (cp.complex64, cp.float32) or project is not None or initial is not None:
        raise ValueError('Fused linear CG requires zero-start complex64/float32 GPU vectors')
    if residual_smoothing and precondition is not None:
        raise ValueError('Residual smoothing requires symmetrically preconditioned coordinates')
    captures = dict(zip(action.__code__.co_freevars, (c.cell_contents for c in action.__closure__)))
    metric = Metric(captures['Q'])
    physical_limit = tolerance*tolerance*metric.norm(rhs)[1]
    x = cp.zeros_like(rhs); r = rhs.copy()
    rr = dot(r, r); limit = tolerance*tolerance*rr
    if rr == 0:
        return x, 0
    partial = cp.empty(min(16384, (rhs.nbytes//4+255)//256), dtype=cp.float64)
    z = precondition(r) if precondition else r
    rz = dot(r, z) if precondition else rr
    p = z.copy()
    direction = None
    y, estimate, weight = None, rr, 0.
    for i in range(maxiter):
        if action_and_dot is None:
            ap = action(p); pap = dot(p, ap)
        else:
            # The optional operator consumes the deferred p update before
            # applying A. Audits and reliable restarts use the plain action.
            ap, pap = action_and_dot(p, direction=direction)
            direction = None
        if not np.isfinite([pap, rz]).all() or min(pap, rz) <= 0:
            raise NonlinearConvergenceError('Fused linear CG lost positivity')
        update_args = () if y is None else ((y, weight),)
        new = fused_update(p, ap, rz/pap, x, r, partial, *update_args)
        if not np.isfinite(new) or new < 0:
            raise NonlinearConvergenceError('Nonfinite fused linear residual')
        estimate = new if y is None else estimate
        if residual_smoothing:
            if y is None:
                # Spend an extra vector pass only in the final decade of
                # residual norms. The factor is independent of the material,
                # mesh, geometry and requested tolerance.
                if limit < new <= 100*limit:
                    y, weight = x.copy(), 0.
            else:
                weight = estimate/(estimate+new)
                estimate = weight*new
        if estimate <= limit:
            if y is not None:
                _finish_smoothing(x, y, weight)
                x = y
            r = rhs-action(x)
            new = dot(r, r)
            if new <= limit and metric.norm(r)[1] <= physical_limit:
                return x, i+1
            # Reset both sequences after replacing the recursive residual.
            # x owns the old y buffer; a later smoothing start makes a copy.
            y, estimate, weight = None, new, 0.
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            p = z.copy()
        else:
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            if action_and_dot is None:
                _cg_updates()[1](z, np.float64(next_rz/rz), p)
            else:
                direction = (z, next_rz/rz)
        rz = next_rz
    raise NonlinearConvergenceError('Fused linear CG failed true residual')

@contextmanager
def trial():
    with patch.object(lc,'fused_cg',cg):yield
