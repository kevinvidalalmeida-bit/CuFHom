"""Isolated GPU pilot: H = I + QR* (C-C0) QR on admissible coordinates.

Keeps the full production action for true-residual restarts and final audits.
The search-direction norm is fused into QR; background stress is zero.
"""
from contextlib import contextmanager
import inspect
from unittest.mock import patch
import numpy as np
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.linear_krylov import coordinate_source
from cufhom.pipeline.linear_energy import _MATERIAL,_REDUCE,adjoint_update_source
from cufhom.pipeline.linear_fusions import replace_once
from cufhom.pipeline.nonlinear_solver import cp
from cufhom_core.tensors import fft

_KERNELS={}
LAST_RUN=[]
_STRAIN=r'''
extern "C" __global__ void strain_norm(float2* p,const float2* r,float beta,int update,
 float2* h,double* partial,long long n0,long long n1,long long n2,long long nz,
 float iy0,float iy1,float iy2,float norm,float transverse,float longitudinal) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 double sum=0;
 if(s<count) {
   long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
   bool valid=s!=0 && !(n0%2==0&&i0==n0/2) && !(n1%2==0&&i1==n1/2) && !(n2%2==0&&i2==n2/2);
   if(valid)for(int c=0;c<3;c++) {
     long long j=c*count+s;float2 pi=p[j];
     if(update){float2 ri=r[j];pi.x=ri.x+beta*pi.x;pi.y=ri.y+beta*pi.y;p[j]=pi;}
     sum+=(double)pi.x*pi.x+(double)pi.y*pi.y;
   }
   coordinate_local(p,h,n0,n1,n2,nz,iy0,iy1,iy2,norm,0,0,transverse,longitudinal);
 }
 REDUCE
}
'''


def strain_source():
    source=replace_once(coordinate_source(),'extern "C" __global__ void coordinates',
                        '__device__ __forceinline__ void coordinate_local')
    source=source.replace('COMPLEX','float2').replace('REAL','float').replace('MAKE','make_float2')
    return source+_STRAIN.replace('REDUCE',_REDUCE)


def update_source(smoothing):
    source=adjoint_update_source(smoothing)
    source=replace_once(source,'ri.x-=alpha*ap[c].x;ri.y-=alpha*ap[c].y;',
                        'ri.x-=alpha*(pi.x+ap[c].x);ri.y-=alpha*(pi.y+ap[c].y);')
    return source


class SplitHP:
    def __init__(self,Q,spectrum,state=None):self.Q,self.spectrum,self.state=Q,spectrum,state
    def update(self,p,alpha,x,r,smoothing):
        mode='update' if smoothing is None else 'update_smooth'
        if mode not in _KERNELS:_KERNELS[mode]=cp.RawKernel(update_source(smoothing is not None),'adjoint_update')
        Q=self.Q;blocks=(Q.count+255)//256;partial=cp.empty(blocks,cp.float64)
        y,blend=(x,0.) if smoothing is None else smoothing
        args=(self.spectrum,p,x,r,partial,*(np.int64(n) for n in Q.shape),np.int64(Q.hshape[-1]),
              *(np.float32(1/L) for L in Q.P.lengths),np.float32(Q.norm),np.float32(Q.transverse),
              np.float32(Q.longitudinal),np.float32(alpha),y,np.float32(blend))
        _KERNELS[mode]((blocks,),(256,),args)
        norm=float(cp.sum(partial,dtype=cp.float64));self.spectrum=None
        if self.state is not None:self.state['rr']=norm
        return norm


def prepare(Q,descriptor):
    table=descriptor.material_table
    G=.5/Q.transverse**2;K=1/Q.longitudinal**2-4*G/3
    reference=np.zeros((6,6));reference[:3,:3]=K-2*G/3
    np.fill_diagonal(reference[:3,:3],K+4*G/3);reference[3:,3:]=2*G*np.eye(3)
    indices=np.triu_indices(6)
    observed=cp.asnumpy(table[0])
    if not np.allclose(observed,reference[indices],rtol=3e-6,atol=3e-6*np.max(abs(reference))):
        raise ValueError('Background is not the isotropic reference whitened by QR')
    return cp.ascontiguousarray(table-table[:1])


def action(Q,descriptor,delta,p,direction=None,state=None):
    if 'strain' not in _KERNELS:_KERNELS['strain']=cp.RawKernel(strain_source(),'strain_norm')
    if 'material' not in _KERNELS:
        source=replace_once(_MATERIAL,'if(s<count) {','if(s<count && ids[s]!=0) {')
        source=replace_once(source,' }\n REDUCE',
                            ' } else if(s<count) {for(int c=0;c<6;c++)y[c*count+s]=0;}\n REDUCE')
        _KERNELS['material']=cp.RawKernel(source.replace('REDUCE',_REDUCE),'material_energy')
    bq=(Q.count+255)//256;V=int(np.prod(Q.shape));bm=(V+255)//256
    if state is None:
        partial=cp.empty(bq+bm,cp.float64)
        h=cp.empty((6,)+Q.hshape,cp.complex64)
        residual,beta=(p,0.) if direction is None else direction
        args=(p,residual,np.float32(beta),np.int32(direction is not None),h,partial[:bq],
              *(np.int64(n) for n in Q.shape),np.int64(Q.hshape[-1]),*(np.float32(1/L) for L in Q.P.lengths),
              np.float32(Q.norm),np.float32(Q.transverse),np.float32(Q.longitudinal))
        _KERNELS['strain']((bq,),(256,),args)
        strain=fft.irfftn_unscaled_cached(h,Q.shape,(1,2,3));del h
        material_partial=partial[bq:]
    else:
        # Exact in CG arithmetic: r_k is orthogonal to p_{k-1}. FP32 can
        # violate it, so this isolated variant retains all true audits.
        state['pnorm']=(state['rr'] if direction is None else
                        state['rr']+float(np.float32(direction[1]))**2*state['pnorm'])
        strain=Q.strain(p) if direction is None else Q.strain_updated_direction(p,*direction)
        partial=cp.empty(bm,cp.float64);material_partial=partial
    stress=cp.empty_like(strain)
    _KERNELS['material']((bm,),(256,),
                         (descriptor.val,delta,strain,stress,np.int64(V),material_partial))
    pap=float(cp.sum(partial,dtype=cp.float64))
    if state is not None:pap+=state['pnorm']
    return SplitHP(Q,fft.rfftn_cached(stress,Q.shape,(1,2,3)),state),pap


def update(p,step,alpha,x,r,partial,smoothing=None):return step.update(p,alpha,x,r,smoothing)


@contextmanager
def split_context(mode='direct'):
    original=lc.fused_cg;cache={};LAST_RUN.clear()
    def solve(plain,rhs,dot,*args,**kwargs):
        captured=inspect.getclosurevars(plain).nonlocals
        Q=captured['Q'];descriptor=getattr(captured['A'],'op',captured['A'])
        if rhs.dtype!=cp.complex64 or kwargs.get('precondition') is not None:
            raise ValueError('The split pilot requires the symmetric FP32 coordinate route')
        key=id(descriptor)
        if key not in cache:cache[key]=prepare(Q,descriptor)
        state={} if mode=='recurrence' else None
        def observed_dot(a,b):
            result=dot(a,b)
            if state is not None:state['rr']=result
            return result
        calls=[0]
        def observed(p,direction=None):
            calls[0]+=1
            return action(Q,descriptor,cache[key],p,direction,state)
        kwargs['action_and_dot']=observed;kwargs['update_callback']=update
        result,iterations=original(plain,rhs,observed_dot,*args,**kwargs)
        LAST_RUN.append(dict(iterations=iterations,split_calls=calls[0]))
        return result,iterations
    with patch.object(lc,'fused_cg',solve):yield


@contextmanager
def recurrence_context():
    with split_context('recurrence'):yield
