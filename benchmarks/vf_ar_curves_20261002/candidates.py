"""Isolated code-review pilots; production imports none of these patches."""
from contextlib import contextmanager,ExitStack
import inspect
from unittest.mock import patch

import numpy as np
from cufhom.pipeline import fft_solver as fs
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom_core.tensors import fft,operators

cp=fs.cp
LAST=dict(spectrum_reuses=0,rotation_groups=0)
_KERNEL=None
_ROTATION=r'''
__device__ double bond(const double* R,int row,int column) {
 const int p[6]={0,1,2,1,0,0},q[6]={0,1,2,2,2,1};
 int i=p[row],j=q[row],a=p[column],b=q[column];
 double factor=row<3?1.:1.4142135623730950488;
 double value=R[3*i+a]*R[3*j+b];
 if(column>=3)value=(value+R[3*i+b]*R[3*j+a])*.7071067811865475244;
 return factor*value;
}
extern "C" __global__ void rotate_mandel(const double* R,const double* C,REAL* output,
 long long groups,int width) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(s>=groups*width)return;
 long long group=s/width;int index=s%width,i,j;
 if(width==36){i=index/6;j=index%6;}
 else {i=0;int first=0;while(index>=first+6-i){first+=6-i;i++;}j=i+index-first;}
 double mi[6],mj[6];
 #pragma unroll
 for(int a=0;a<6;a++){mi[a]=bond(R+9*group,i,a);mj[a]=bond(R+9*group,j,a);}
 double sum=0;
 #pragma unroll
 for(int a=0;a<6;a++) {
   double inner=0;
   #pragma unroll
   for(int b=0;b<6;b++)inner+=C[6*a+b]*mj[b];
   sum+=mi[a]*inner;
 }
 output[s]=(REAL)sum;
}
'''


def rotate_batch(C,R,dtype,packed,chunk):
    global _KERNEL
    dtype=np.dtype(dtype)
    if _KERNEL is None:_KERNEL={}
    if dtype.str not in _KERNEL:
        _KERNEL[dtype.str]=cp.RawKernel(_ROTATION.replace('REAL','float' if dtype==np.float32 else 'double'),'rotate_mandel')
    width=21 if packed else 36
    result=cp.empty((len(R),width),dtype=dtype)
    tensor=cp.asarray(C,cp.float64);chunk=int(chunk) if chunk else len(R)
    chunk=max(1,min(chunk,len(R)))
    for first in range(0,len(R),chunk):
        rotations=cp.asarray(R[first:first+chunk],cp.float64);n=len(rotations)
        _KERNEL[dtype.str](((n*width+127)//128,),(128,),
                         (rotations,tensor,result[first:first+n],np.int64(n),np.int32(width)))
    LAST['rotation_groups']+=len(R)
    return result if packed else result.reshape((len(R),6,6))


@contextmanager
def rotation_context():
    original=inspect.getsource(fs._build_cfield_gpu)
    first=original.index('    Cf_local_f64 = ');last=original.index('    inv_idx_gpu = ')
    source=original[:first]+'    Cm_rot = _rotate_batch(Cf_local, R_batch, gpu_dtype, packed_sym21, rotation_batch_size)\n\n'+original[last:]
    namespace=dict(vars(fs),_rotate_batch=rotate_batch)
    exec(compile(source,'<isolated Mandel rotation>','exec'),namespace)
    LAST['rotation_groups']=0
    with patch.object(fs,'_build_cfield_gpu',namespace['_build_cfield_gpu']):yield


@contextmanager
def spectrum_context():
    original_adjoint=SymmetricCoordinates.adjoint
    original_strain=SymmetricCoordinates.strain
    original_direction=SymmetricCoordinates.strain_updated_direction
    original_project=operators.Operator.apply_material_result
    original_forward=operators.rfftn_cached
    cache=[];LAST['spectrum_reuses']=0
    def adjoint(Q,field):
        cache.clear()
        if Q.xp is not cp:return original_adjoint(Q,field)
        with Q.P.timers.measure('coordinates_adjoint'):
            h=fft.rfftn_cached(field,Q.shape,(1,2,3))
            result=Q._map(h,True);cache.extend((field,Q.shape,h))
            return result
    def strain(Q,z):
        cache.clear();return original_strain(Q,z)
    def direction(Q,p,r,beta):
        cache.clear();return original_direction(Q,p,r,beta)
    def project(operator,stress,material,projection):
        if not cache or cache[0] is not stress.val:return original_project(operator,stress,material,projection)
        field,shape,h=cache
        def forward(value,n,axes):
            if value is field and tuple(n)==shape and tuple(a%value.ndim for a in axes)==(1,2,3):
                LAST['spectrum_reuses']+=1;return h
            return original_forward(value,n,axes)
        with patch.object(operators,'rfftn_cached',forward):
            result=original_project(operator,stress,material,projection)
        cache.clear()
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(SymmetricCoordinates,'adjoint',adjoint))
        stack.enter_context(patch.object(SymmetricCoordinates,'strain',strain))
        stack.enter_context(patch.object(SymmetricCoordinates,'strain_updated_direction',direction))
        stack.enter_context(patch.object(operators.Operator,'apply_material_result',project))
        yield


@contextmanager
def combined_context():
    with rotation_context(),spectrum_context():yield
