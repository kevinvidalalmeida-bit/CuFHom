"""Rotate an anisotropic Mandel matrix with B(R) C B(R)^T on the GPU.

Mandel ordering is 00, 11, 22, 12, 02, 01. The basis tensors are orthonormal,
so B is orthogonal for an orthogonal R. Rotation algebra uses FP64 before
one final cast to the requested table precision; no fourth-order fields
or separate coefficient-packing kernels are needed.
"""
import numpy as np
import cupy as cp

_KERNELS={}
_SOURCE=r'''
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


def rotate_mandel_batch_gpu(C,rotations,dtype,packed_sym21,chunk_size=0):
    """Return one rotated 6×6 or upper-triangle table per supplied rotation.

    The caller supplies orthogonal 3×3 rotations and a symmetric 6×6 matrix,
    already validated by the material builder. Chunking bounds rotation
    temporaries without changing group order or coefficient packing.
    """
    dtype=np.dtype(dtype)
    if dtype not in (np.dtype('float32'),np.dtype('float64')):
        raise ValueError('Mandel table requires float32 or float64')
    if dtype.str not in _KERNELS:
        _KERNELS[dtype.str]=cp.RawKernel(
            _SOURCE.replace('REAL','float' if dtype==np.float32 else 'double'),'rotate_mandel')
    width=21 if packed_sym21 else 36
    result=cp.empty((len(rotations),width),dtype=dtype)
    if not len(rotations):return result if packed_sym21 else result.reshape((0,6,6))
    tensor=cp.ascontiguousarray(cp.asarray(C,dtype=cp.float64))
    chunk=int(chunk_size) if chunk_size else len(rotations)
    chunk=max(1,min(chunk,len(rotations)))
    for first in range(0,len(rotations),chunk):
        rotation=cp.ascontiguousarray(cp.asarray(rotations[first:first+chunk],dtype=cp.float64))
        n=len(rotation)
        _KERNELS[dtype.str](((n*width+127)//128,),(128,),
            (rotation,tensor,result[first:first+n],np.int64(n),np.int32(width)))
    return result if packed_sym21 else result.reshape((len(rotations),6,6))
