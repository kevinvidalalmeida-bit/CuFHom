"""Archived fixed-SPD PCG and late smoothing in the r*B*r metric.

In exact arithmetic PCG residuals are B-orthogonal. Their minimum-norm convex
combination therefore uses the harmonic sum of r*B*r, rather than r*r.
Every apparent convergence still checks the original coordinate residual.
"""
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_krylov as lk
from cufhom.pipeline.nonlinear_solver import _cg_updates, NonlinearConvergenceError


def metric_pcg(action,rhs,dot,tolerance,maxiter,xp,project=None,fused=False,
               precondition=None,initial=None,action_and_dot=None,residual_smoothing=True):
    if xp is not cp or precondition is None or project is not None or initial is not None:
        raise ValueError('Metric PCG needs a fixed SPD GPU preconditioner and zero start')
    x=cp.zeros_like(rhs);r=rhs.copy();rr=dot(r,r);limit=tolerance*tolerance*rr
    if rr==0:return x,0
    z=precondition(r);rho=dot(r,z);metric_limit=tolerance*tolerance*rho
    p=z.copy();direction=None;y=None;estimate=rho;weight=0.
    partial=cp.empty(min(16384,(rhs.nbytes//4+255)//256),cp.float64)
    for iteration in range(maxiter):
        if action_and_dot is None:ap=action(p);pap=dot(p,ap)
        else:ap,pap=action_and_dot(p,direction=direction);direction=None
        if not np.isfinite([rho,pap]).all() or min(rho,pap)<=0:
            raise NonlinearConvergenceError('Metric PCG lost positivity')
        rr=lk.fused_update(p,ap,rho/pap,x,r,partial,None if y is None else (y,weight))
        z=precondition(r);next_rho=dot(r,z)
        if not np.isfinite([rr,next_rho]).all() or min(rr,next_rho)<0:
            raise NonlinearConvergenceError('Metric PCG has a nonfinite residual')
        estimate=next_rho if y is None else estimate
        if residual_smoothing:
            if y is None:
                if metric_limit<next_rho<=100*metric_limit:y=x.copy();weight=0.
            else:
                weight=estimate/(estimate+next_rho) if estimate+next_rho else 0.
                estimate=weight*next_rho
        if estimate<=metric_limit or (y is None and rr<=limit):
            if y is not None:lk._finish_smoothing(x,y,weight);x=y
            r=rhs-action(x);rr=dot(r,r)
            if rr<=limit:return x,iteration+1
            z=precondition(r);next_rho=dot(r,z)
            y=None;estimate=next_rho;weight=0.;p=z.copy();direction=None
        else:
            if action_and_dot is None:_cg_updates()[1](z,np.float64(next_rho/rho),p)
            else:direction=(z,next_rho/rho)
        rho=next_rho
    raise NonlinearConvergenceError('Metric PCG failed the true residual')
