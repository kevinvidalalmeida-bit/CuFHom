"""Research residual smoothing of existing symmetric CG; original audit retained."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import fused_update
from cufhom.pipeline.nonlinear_solver import _cg_updates
_BLEND=None

def smoothed(action,rhs,dot,tolerance,maxiter,xp,action_and_dot=None,precondition=None,**kwargs):
    global _BLEND
    if precondition is not None:raise ValueError('Already whitened coordinates expected')
    x=xp.zeros_like(rhs);r=rhs.copy();rr=dot(r,r);limit=tolerance*tolerance*rr
    if rr==0:return x,0
    y=x.copy();estimate=rr;p=r.copy();direction=None
    if xp is cp:
        partial=cp.empty(min(16384,(rhs.nbytes//4+255)//256),cp.float64)
        if _BLEND is None:_BLEND=cp.ElementwiseKernel('T x, float32 weight','T y',
            'y = y + weight * (x-y);','cufhom_cg_smooth')
    for i in range(maxiter):
        if action_and_dot:
            ap,pap=action_and_dot(p,direction=direction);direction=None
        else:
            ap=action(p);pap=dot(p,ap)
        if min(rr,pap)<=0 or not np.isfinite([rr,pap]).all():raise RuntimeError('CG positivity lost')
        if xp is cp:new=fused_update(p,ap,rr/pap,x,r,partial)
        else:
            x+=rr/pap*p;r-=rr/pap*ap;new=dot(r,r)
        weight=estimate/(estimate+new)
        estimate=weight*new
        if xp is cp:_BLEND(x,np.float32(weight),y)
        else:y+=weight*(x-y)
        if estimate<=limit:
            true=rhs-action(y);true_norm=dot(true,true)
            if true_norm<=limit:return y,i+1
            # Do not trust orthogonality after residual replacement. Restart
            # CG and the smoothing sequence from the audited approximation.
            x=y.copy();r=true;p=r.copy();rr=estimate=true_norm;direction=None
            continue
        if action_and_dot:direction=(r,new/rr)
        elif xp is cp:_cg_updates()[1](r,np.float64(new/rr),p)
        else:p=r+new/rr*p
        rr=new
    raise RuntimeError('Smoothed CG failed true residual')

@contextmanager
def trial():
    with patch.object(lc,'fused_cg',smoothed):yield
