"""Local three-component acoustic-symbol approximate inverse, research only.

D_j = mean_n (S(n) R(n))^T C_j (S(n) R(n)), B = U* D^-1 U.
U is the unitary real Fourier map on admissible coordinates. D is SPD by
construction. No six-component compliance average or coarse constant blocks.
This is a principal-symbol approximation, NOT the exact diagonal of H.
"""
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import time
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import fused_cg
from cufhom_core.tensors import fft

_CODE=r'''
extern "C" __global__ void scale(const float2* x,float2* y,long long n0,long long n1,
 long long n2,long long nz,float norm,int adjoint){
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 bool valid=s!=0&&!(n0%2==0&&i0==n0/2)&&!(n1%2==0&&i1==n1/2)&&!(n2%2==0&&i2==n2/2);
 float weight=(i2==0||(n2%2==0&&i2==n2/2))?1:1.4142135623730951f;
 float a=valid?(adjoint?norm*weight:norm/weight):0;
 for(int c=0;c<3;c++){float2 z=x[c*count+s];y[c*count+s]=make_float2(a*z.x,a*z.y);}}
extern "C" __global__ void local(const int* ids,const float* D,const float* x,float* y,long long n){
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(s>=n)return;const float* d=D+9*ids[s];
 float v[3]={x[s],x[n+s],x[2*n+s]};for(int c=0;c<3;c++)y[c*n+s]=d[c*3]*v[0]+d[c*3+1]*v[1]+d[c*3+2]*v[2];}
'''
_MODULE=None

def acoustic_matrices(tensors,bulk,shear,samples=96):
    t=(np.arange(samples)+.5)/samples
    z=1-2*t;phi=np.arange(samples)*np.pi*(3-np.sqrt(5))
    n=np.stack([np.sqrt(1-z*z)*np.cos(phi),np.sqrt(1-z*z)*np.sin(phi),z],axis=1)
    root=(np.eye(3)[None]*np.sqrt(1/(2*shear))+
        (np.sqrt(1/(bulk+4*shear/3))-np.sqrt(1/(2*shear)))*n[:,:,None]*n[:,None,:])
    v=np.sqrt(2)*np.eye(3)[None]+(1-np.sqrt(2))*n[:,:,None]*n[:,None,:]
    v=v@root
    S=np.empty((samples,6,3))
    S[:,:3]=n[:,:,None]*v
    for c,i,j in ((3,1,2),(4,0,2),(5,0,1)):S[:,c]=(n[:,i,None]*v[:,j]+n[:,j,None]*v[:,i])/np.sqrt(2)
    D=np.einsum('nai,jab,nbk->jik',S,tensors,S)/samples
    return (D+D.swapaxes(1,2))*.5

class AcousticLocal:
    def __init__(self,A,Q,bulk,shear,records,weight=1.):
        global _MODULE
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        if _MODULE is None:_MODULE=cp.RawModule(code=_CODE)
        self.Q=Q;self.shape=Q.shape;self.hshape=Q.hshape;self.ids=A.val
        table=cp.asnumpy(A.material_table).astype(np.float64)
        C=np.zeros((len(table),6,6));i,j=np.triu_indices(6);C[:,i,j]=table;C[:,j,i]=table
        D=acoustic_matrices(C,bulk,shear)
        eig=np.linalg.eigvalsh(D)
        if eig.min()<=0:raise ValueError('Non-SPD acoustic approximation')
        self.inverse=cp.asarray(np.linalg.inv(D),dtype=cp.float32)
        self.weight=weight;self.scale=_MODULE.get_function('scale');self.local=_MODULE.get_function('local')
        cp.cuda.get_current_stream().synchronize()
        records.append(dict(setup_s=time.perf_counter()-start,tensor_count=len(C),
            D_min=float(eig.min()),D_max=float(eig.max()),precondition_calls=0,weight=weight))
        self.record=records[-1]
    def map(self,v,adjoint):
        out=cp.empty((3,)+self.hshape,dtype=cp.complex64)
        self.scale(((self.Q.count+255)//256,),(256,),
            (v,out,*(np.int64(n) for n in self.shape),np.int64(self.hshape[-1]),np.float32(self.Q.norm),np.int32(adjoint)))
        return out
    def apply(self,v):
        self.record['precondition_calls']+=1
        h=self.map(v,False)
        field=fft.irfftn_unscaled_cached(h,self.shape,(1,2,3))
        out=cp.empty_like(field);N=int(np.prod(self.shape))
        self.local(((N+255)//256,),(256,),(self.ids,self.inverse,field,out,np.int64(N)))
        z=self.map(fft.rfftn_cached(out,self.shape,(1,2,3)),True).ravel()
        return z if self.weight==1. else (1-self.weight)*v+self.weight*z

@contextmanager
def trial(records,weight=1.):
    original=lc.make_linear_solver
    def factory(**kw):
        cache={};solve=original(**kw)
        def coord(P,*args):
            Q=lc.SymmetricCoordinates_original(P,*args)
            if 'precondition' not in cache:cache['precondition']=AcousticLocal(kw['A'],Q,kw['bulk'],kw['shear'],records,weight)
            return Q
        def cg(*a,**k):
            k['precondition']=cache['precondition'].apply;k['residual_smoothing']=False
            return fused_cg(*a,**k)
        def call(**p):
            with patch.object(lc,'SymmetricCoordinates',coord),patch.object(lc,'fused_cg',cg):return solve(**p)
        return call
    with ExitStack() as stack:
        stack.enter_context(patch.object(lc,'SymmetricCoordinates_original',lc.SymmetricCoordinates,create=True))
        stack.enter_context(patch.object(lc,'make_linear_solver',factory))
        yield
