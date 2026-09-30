"""Research block conjugate directions in real coordinates, rank-revealing blocks."""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom_core.tensors import Tensor, Operator

_CODE = r'''
extern "C" __global__ void gram(const float* a,const float* b,double* out,
 long long n,int na,int nb) {
 int pair=blockIdx.y,ia=pair/nb,ib=pair%nb;
 double sum=0;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<n;s+=(long long)blockDim.x*gridDim.x)
   sum+=(double)a[ia*n+s]*(double)b[ib*n+s];
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double w[8];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(!lane)w[warp]=sum;__syncthreads();
 if(!warp){sum=lane<8?w[lane]:0;for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 if(!lane)out[(long long)pair*gridDim.x+blockIdx.x]=sum;}
}
extern "C" __global__ void combine(const float* a,const double* c,float* out,
 long long n,int na,int nb) {
 int row=blockIdx.y;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<n;s+=(long long)blockDim.x*gridDim.x){
 double sum=0;for(int i=0;i<na;i++)sum+=(double)a[i*n+s]*c[row*na+i];
 out[row*n+s]=sum;}
}
'''
_KERNELS={}

class Ops:
    def __init__(self,xp):self.xp=xp
    def gram(self,a,b):
        if self.xp is np:return (a.conj()@b.T).real
        if 'gram' not in _KERNELS:_KERNELS['gram']=cp.RawKernel(_CODE,'gram')
        na,nb,n=len(a),len(b),a[0].nbytes//4
        partial=cp.empty((na*nb,min(4096,(n+255)//256)),cp.float64)
        _KERNELS['gram']((partial.shape[1],na*nb),(256,),
            (a,b,partial,np.int64(n),np.int32(na),np.int32(nb)))
        return cp.asnumpy(cp.sum(partial,axis=1)).reshape(na,nb)
    def combine(self,a,c):
        if self.xp is np:return c@a
        if 'combine' not in _KERNELS:_KERNELS['combine']=cp.RawKernel(_CODE,'combine')
        out=cp.empty((len(c),a.shape[1]),a.dtype);n=a[0].nbytes//4
        _KERNELS['combine']((min(16384,(n+255)//256),len(c)),(256,),
            (a,cp.asarray(np.ascontiguousarray(c),dtype=cp.float64),out,np.int64(n),np.int32(len(a)),np.int32(len(c))))
        return out
    def orthonormal(self,r):
        gram=self.gram(r,r);gram=(gram+gram.T)/2
        values,vectors=np.linalg.eigh(gram)
        threshold=(1e-10 if self.xp is cp else 1e-24)*max(values[-1],np.finfo(float).tiny)
        keep=values>threshold
        if not keep.any():return r[:0]
        return self.combine(r,(vectors[:,keep]/np.sqrt(values[keep])).T)


def block_cg(action,rhs,tolerance,maxiter,xp):
    ops=Ops(xp);X=xp.zeros_like(rhs);R=rhs.copy()
    norms=np.diag(ops.gram(rhs,rhs)).copy();limit=tolerance*tolerance*norms
    calls=0;ranks=[];restarts=0;Pold=APold=Gold=None
    def apply(rows):
        nonlocal calls
        result=xp.empty_like(rows)
        for j in range(len(rows)):result[j]=action(rows[j])
        calls+=len(rows)
        return result
    for iteration in range(maxiter):
        W=ops.orthonormal(R)
        if len(W)==0:
            exact=rhs-apply(X)
            if not np.all(np.diag(ops.gram(exact,exact))<=limit):
                raise RuntimeError('Block rank exhausted before convergence')
            return X,iteration,dict(operator_calls=calls,ranks=ranks,restarts=restarts)
        if Pold is not None:
            coeff=np.linalg.solve(Gold,ops.gram(APold,W))
            P=W-ops.combine(Pold,coeff.T)
        else:P=W
        # Normalize direction block independently: avoids ill-conditioned small solves.
        P=ops.orthonormal(P)
        AP=apply(P);G=ops.gram(P,AP);G=(G+G.T)/2
        np.linalg.cholesky(G)
        alpha=np.linalg.solve(G,ops.gram(P,R))
        X+=ops.combine(P,alpha.T)
        R-=ops.combine(AP,alpha.T)
        ranks.append(len(P))
        rr=np.diag(ops.gram(R,R))
        if np.all(rr<=limit):
            R=rhs-apply(X)
            rr=np.diag(ops.gram(R,R))
            if np.all(rr<=limit):return X,iteration+1,dict(operator_calls=calls,ranks=ranks,restarts=restarts)
            Pold=APold=Gold=None;restarts+=1
        else:Pold,APold,Gold=P,AP,G
    raise RuntimeError('Block CG failed true coordinate residual')


@contextmanager
def trial(records):
    original_factory=lc.make_linear_solver
    cached=None;expected=None;next_load=0
    def factory(*,A,GN,pb,**kwargs):
        delegate=original_factory(A=A,GN=GN,pb=pb,**kwargs)
        def inner(action,rhs,dot,tolerance,maxiter,xp,**kw):
            nonlocal cached,expected,next_load
            if cached is None:
                shape=tuple(pb.solve['N'])
                Q=lc.SymmetricCoordinates(lc.MixedProjector(shape,tuple(pb.Y),[True]*6,cp),kwargs['bulk'],kwargs['shear'])
                tensor=Tensor(name='E',N=shape,shape=(6,),Fourier=False,fft_form='r',dtype=np.float32)
                expected=cp.empty((6,rhs.size),rhs.dtype)
                for j in range(6):
                    tensor.val.fill(0);tensor.val[j]=1
                    b=Operator(name='FiGFA',mat=[[GN,A]])(-tensor)
                    expected[j]=Q.adjoint(b.val)/np.float32(np.prod(shape))
                cached,it,info=block_cg(action,expected,tolerance,min(maxiter,200),cp)
                records.append(dict(block_iterations=it,**info))
            delta=rhs-expected[next_load]
            if np.sqrt(dot(delta,delta)/max(dot(rhs,rhs),np.finfo(float).tiny))>1e-6:
                raise RuntimeError('Block right-hand side differs from production')
            result=cached[next_load].copy();it=records[-1]['block_iterations']
            next_load+=1
            if next_load==6:cached=expected=None
            return result,it
        def solve(*a,**kw):
            with patch.object(lc,'fused_cg',inner):return delegate(*a,**kw)
        return solve
    with patch.object(lc,'make_linear_solver',factory):yield
