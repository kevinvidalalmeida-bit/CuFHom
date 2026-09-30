"""Research residual smoothing of existing symmetric CG; original audit retained."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import _UPDATE, fused_update as plain_update
from cufhom.pipeline.linear_fusions import replace_once
from cufhom.pipeline.nonlinear_solver import _cg_updates
_BLEND=None
_FUSED=None

def fused_update(p,ap,alpha,x,r,partial,y,weight):
    global _FUSED
    if _FUSED is None:
        code=replace_once(_UPDATE, 'double* partial, long long count)',
            'double* partial, long long count, float* y, float weight)')
        code=replace_once(code,'x[i]+=alpha*p[i];',
            'float xi=x[i]; y[i]+=weight*(xi-y[i]); x[i]=xi+alpha*p[i];')
        _FUSED=cp.RawKernel(code,'update_rr')
    _FUSED((partial.size,), (256,), (p,ap,np.float32(alpha),x,r,partial,
        np.int64(r.nbytes//4),y,np.float32(weight)))
    return float(cp.sum(partial,dtype=cp.float64))


def smoothed(action,rhs,dot,tolerance,maxiter,xp,action_and_dot=None,precondition=None,**kwargs):
    global _BLEND
    if precondition is not None:raise ValueError('Already whitened coordinates expected')
    x=xp.zeros_like(rhs);r=rhs.copy();rr=dot(r,r);limit=tolerance*tolerance*rr
    if rr==0:return x,0
    y=None;estimate=rr;p=r.copy();direction=None;weight=0.
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
        if xp is cp:
            new=(fused_update(p,ap,rr/pap,x,r,partial,y,weight) if y is not None
                 else plain_update(p,ap,rr/pap,x,r,partial))
        else:
            if y is not None:y+=weight*(x-y)
            x+=rr/pap*p;r-=rr/pap*ap;new=dot(r,r)
        if y is None:
            estimate=new;weight=0.
            if new<=100*limit:y=x.copy()
        else:
            weight=estimate/(estimate+new)
            estimate=weight*new
        if estimate<=limit:
            if xp is cp:_BLEND(x,np.float32(weight),y)
            else:y+=weight*(x-y)
            true=rhs-action(y);true_norm=dot(true,true)
            if true_norm<=limit:return y,i+1
            # Do not trust orthogonality after residual replacement. Restart
            # CG and the smoothing sequence from the audited approximation.
            x=y.copy();r=true;p=r.copy();rr=estimate=true_norm;direction=None;weight=0.
            continue
        if action_and_dot:direction=(r,new/rr)
        elif xp is cp:_cg_updates()[1](r,np.float64(new/rr),p)
        else:p=r+new/rr*p
        rr=new
    raise RuntimeError('Smoothed CG failed true residual')

@contextmanager
def trial():
    with patch.object(lc,'fused_cg',smoothed):yield
