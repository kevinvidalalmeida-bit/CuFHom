"""Chronopoulos/Gear CG pilot: one host reduction, same physical acceptance."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import _finish_smoothing

_CODE=r'''
extern "C" __global__ void pair(const float* r,const float* ar,double* out,long long n){
 double rr=0,ra=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
 rr+=(double)r[i]*r[i];ra+=(double)r[i]*ar[i];}
 for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);ra+=__shfl_down_sync(0xffffffff,ra,s);}
 __shared__ double w[16];int l=threadIdx.x&31,t=threadIdx.x>>5;
 if(!l){w[t]=rr;w[8+t]=ra;}__syncthreads();if(!t){rr=l<8?w[l]:0;ra=l<8?w[8+l]:0;
 for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);ra+=__shfl_down_sync(0xffffffff,ra,s);}
 if(!l){out[blockIdx.x]=rr;out[gridDim.x+blockIdx.x]=ra;}}}
extern "C" __global__ void update(const float* ar,float* p,float* ap,float* x,float* r,
 float alpha,float beta,long long n,float* y,float weight,int smooth){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
 float pi=r[i]+beta*p[i],api=ar[i]+beta*ap[i],xi=x[i];
 if(smooth)y[i]+=weight*(xi-y[i]);
 p[i]=pi;ap[i]=api;x[i]=xi+alpha*pi;r[i]-=alpha*api;}}
'''
_MODULE=None

def cg(action,rhs,dot,tolerance,maxiter,xp,precondition=None,action_and_dot=None,
       residual_smoothing=True,**kwargs):
    global _MODULE
    if precondition is not None:raise ValueError('Pilot requires symmetrically whitened coordinates')
    x=xp.zeros_like(rhs);r=rhs.copy();rr=dot(r,r);limit=rr*tolerance*tolerance
    if rr==0:return x,0
    p=x.copy();ap=x.copy();oldrr=oldpap=None;steps=0
    y=None;estimate=rr;weight=0.
    if xp is cp:
        if _MODULE is None:_MODULE=cp.RawModule(code=_CODE)
        blocks=min(16384,(rhs.nbytes//4+255)//256);partial=cp.empty((2,blocks),cp.float64)
        pair=_MODULE.get_function('pair');update=_MODULE.get_function('update')
    while steps<maxiter:
        ar=action(r)
        if xp is cp:
            pair((blocks,),(256,),(r,ar,partial,np.int64(rhs.nbytes//4)))
            rr,delta=cp.asnumpy(cp.sum(partial,axis=1,dtype=cp.float64))
        else:rr,delta=dot(r,r),dot(r,ar)
        if y is None:
            estimate=rr;weight=0.
            if residual_smoothing and limit<rr<=100*limit:y=x.copy()
        else:
            weight=estimate/(estimate+rr);estimate=weight*rr
        if estimate<=limit:
            if y is not None:
                if xp is cp:_finish_smoothing(x,y,weight)
                else:y+=weight*(x-y)
                x=y
            r=rhs-action(x);true=dot(r,r)
            if true<=limit:return x,steps
            y=None;oldrr=oldpap=None;p=xp.zeros_like(rhs);ap=p.copy()
            continue
        beta=0. if oldrr is None else rr/oldrr
        pap=delta if oldpap is None else delta-beta*beta*oldpap
        if not np.isfinite([rr,delta,pap]).all() or min(rr,delta,pap)<=0:
            raise RuntimeError('Single-reduction CG lost positivity')
        alpha=rr/pap
        if xp is cp:
            update((blocks,),(256,),(ar,p,ap,x,r,np.float32(alpha),np.float32(beta),
                np.int64(rhs.nbytes//4),x if y is None else y,np.float32(weight),np.int32(y is not None)))
        else:
            if y is not None:y+=weight*(x-y)
            p=r+beta*p;ap=ar+beta*ap;x+=alpha*p;r-=alpha*ap
        oldrr,oldpap=rr,pap;steps+=1
    true=rhs-action(x)
    if dot(true,true)<=limit:return x,steps
    raise RuntimeError('Single-reduction CG failed true residual')

@contextmanager
def trial():
    with patch.object(lc,'fused_cg',cg):yield
