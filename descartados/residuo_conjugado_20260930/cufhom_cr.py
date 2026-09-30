"""Research CR in the same symmetric coordinates; exact physical audit retained."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import fused_update

_CODE=r'''
extern "C" __global__ void directions(const float* r,const float* ar,float* p,float* ap,
 float beta,double* partial,long long n){
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)blockDim.x*gridDim.x){
 p[i]=r[i]+beta*p[i];float v=ar[i]+beta*ap[i];ap[i]=v;sum+=(double)v*v;}
 for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
 __shared__ double w[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(!lane)w[warp]=sum;__syncthreads();if(!warp){sum=lane<8?w[lane]:0;
 for(int s=16;s;s>>=1)sum+=__shfl_down_sync(0xffffffff,sum,s);
 if(!lane)partial[blockIdx.x]=sum;}}
'''
_KERNEL=None

def cr(action,rhs,dot,tolerance,maxiter,xp,action_and_dot=None,precondition=None,**kwargs):
    global _KERNEL
    if precondition is not None:raise ValueError('Use already symmetrically whitened coordinates')
    x=xp.zeros_like(rhs);r=rhs.copy();rr=dot(r,r);limit=tolerance*tolerance*rr
    if rr==0:return x,0
    if xp is cp:
        if rhs.dtype!=cp.complex64:raise ValueError('GPU CR pilot expects complex64')
        if _KERNEL is None:_KERNEL=cp.RawKernel(_CODE,'directions')
        partial=cp.empty(min(16384,(rhs.nbytes//4+255)//256),cp.float64)
    def apply(z):
        if action_and_dot:return action_and_dot(z)
        az=action(z);return az,dot(z,az)
    ar,rar=apply(r);p=r.copy();ap=ar.copy();apap=dot(ap,ap)
    for i in range(maxiter):
        if min(rar,apap)<=0 or not np.isfinite([rar,apap]).all():raise RuntimeError('CR positivity lost')
        alpha=rar/apap
        if xp is cp:new=fused_update(p,ap,alpha,x,r,partial)
        else:
            x+=alpha*p;r-=alpha*ap;new=dot(r,r)
        if new<=limit:
            r=rhs-action(x);new=dot(r,r)
            if new<=limit:return x,i+1
            ar,rar=apply(r);p=r.copy();ap=ar.copy();apap=dot(ap,ap)
            continue
        ar,next_rar=apply(r);beta=next_rar/rar
        if xp is cp:
            _KERNEL((partial.size,),(256,),
                (r,ar,p,ap,np.float32(beta),partial,np.int64(rhs.nbytes//4)))
            apap=float(cp.sum(partial,dtype=cp.float64))
        else:
            p=r+beta*p;ap=ar+beta*ap;apap=dot(ap,ap)
        rar=next_rar
    raise RuntimeError('CR failed true residual')

@contextmanager
def trial():
    with patch.object(lc,'fused_cg',cr):yield
