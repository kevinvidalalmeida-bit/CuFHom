"""Energy denominator and adjoint/update, without an Hp buffer.

H itself and the true-residual audit retain the production implementation.
The CG recurrence is unchanged; only the order of scalar summation changes.
"""
import numpy as np
import time
from cufhom_core.tensors.fft import cp
from cufhom_core.tensors import fft

_KERNELS = {}
_REDUCE = r'''
 for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 __shared__ double sums[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)sums[warp]=sum;
 __syncthreads();
 if(warp==0) {
   sum=lane<8?sums[lane]:0;
   for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
   if(lane==0)partial[blockIdx.x]=sum;
 }
'''
_MATERIAL = r'''
extern "C" __global__ void material_energy(const int* ids,const float* table,
 const float* x,float* y,long long count,double* partial) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 double sum=0;
 if(s<count) {
   const float* a=table+21*(long long)ids[s];
   float u[6],v[6];
   for(int c=0;c<6;c++)u[c]=x[c*count+s];
   v[0]=a[0]*u[0]+a[1]*u[1]+a[2]*u[2]+a[3]*u[3]+a[4]*u[4]+a[5]*u[5];
   v[1]=a[1]*u[0]+a[6]*u[1]+a[7]*u[2]+a[8]*u[3]+a[9]*u[4]+a[10]*u[5];
   v[2]=a[2]*u[0]+a[7]*u[1]+a[11]*u[2]+a[12]*u[3]+a[13]*u[4]+a[14]*u[5];
   v[3]=a[3]*u[0]+a[8]*u[1]+a[12]*u[2]+a[15]*u[3]+a[16]*u[4]+a[17]*u[5];
   v[4]=a[4]*u[0]+a[9]*u[1]+a[13]*u[2]+a[16]*u[3]+a[18]*u[4]+a[19]*u[5];
   v[5]=a[5]*u[0]+a[10]*u[1]+a[14]*u[2]+a[17]*u[3]+a[19]*u[4]+a[20]*u[5];
   for(int c=0;c<6;c++){y[c*count+s]=v[c];sum+=(double)u[c]*(double)v[c];}
 }
 REDUCE
}
'''
_ADJOINT = r'''
extern "C" __global__ void adjoint_update(const float2* h,const float2* p,
 float2* x,float2* r,double* partial,
 long long n0,long long n1,long long n2,long long nz,
 float iy0,float iy1,float iy2,float norm,float transverse,float longitudinal,
 float alpha,float2* y,float blend) {
 const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 double sum=0;
 if(s<count) {
   float2 ap[6];
   coordinate_local(h,ap,n0,n1,n2,nz,iy0,iy1,iy2,norm,0,1,transverse,longitudinal);
   for(int c=0;c<3;c++) {
     long long j=c*count+s;
     float2 xi=x[j],pi=p[j],ri=r[j];
     SMOOTH
     xi.x+=alpha*pi.x;xi.y+=alpha*pi.y;x[j]=xi;
     ri.x-=alpha*ap[c].x;ri.y-=alpha*ap[c].y;r[j]=ri;
     sum+=(double)ri.x*ri.x+(double)ri.y*ri.y;
   }
 }
 REDUCE
}
'''


def adjoint_update_source(smoothing):
    # Reuse the established coordinate formula. Its output is thread-local;
    # no Hp vector is stored. Every wrapper thread reaches the reduction.
    from cufhom.fft.linear.krylov import coordinate_source
    from cufhom.fft.linear.krylov import replace_once
    code = coordinate_source()
    code = replace_once(code, 'extern "C" __global__ void coordinates',
                        '__device__ __forceinline__ void coordinate_local')
    code = replace_once(code, 'y[3*count+j++]', 'y[j++]')
    if code.count('y[c*count+s]') != 2 or code.count('y[c*count]') != 2:
        raise RuntimeError('Coordinate output changed: review energy/update fusion')
    code = code.replace('y[c*count+s]', 'y[c]')
    code = code.replace('y[c*count]', 'y[c]')
    code = code.replace('COMPLEX','float2').replace('REAL','float').replace('MAKE','make_float2')
    smooth = ('' if not smoothing else
        'float2 yi=y[j];yi.x+=blend*(xi.x-yi.x);yi.y+=blend*(xi.y-yi.y);y[j]=yi;')
    return code+_ADJOINT.replace('SMOOTH',smooth).replace('REDUCE',_REDUCE)


class DeferredHP:
    def __init__(self, Q, spectrum):
        self.Q, self.spectrum = Q, spectrum

    def update(self, p, alpha, x, r, smoothing):
        mode = 'adjoint' if smoothing is None else 'adjoint_smooth'
        if mode not in _KERNELS:
            _KERNELS[mode] = cp.RawKernel(adjoint_update_source(smoothing is not None), 'adjoint_update')
        Q = self.Q
        blocks = (Q.count+255)//256
        partial = cp.empty(blocks, cp.float64)
        y, blend = (x, 0.) if smoothing is None else smoothing
        args = (self.spectrum, p, x, r, partial, *(np.int64(n) for n in Q.shape), np.int64(Q.hshape[-1]),
                *(np.float32(1/L) for L in Q.P.lengths), np.float32(Q.norm),
                np.float32(Q.transverse), np.float32(Q.longitudinal), np.float32(alpha), y, np.float32(blend))
        _KERNELS[mode]((blocks,), (256,), args)
        norm = float(cp.sum(partial, dtype=cp.float64))
        # The scalar reduction synchronizes this stream. Hp is consumed;
        # release its six-component spectrum before building the next H.
        self.spectrum = None
        return norm


def material_energy(descriptor, strain):
    if 'material' not in _KERNELS:
        _KERNELS['material'] = cp.RawKernel(_MATERIAL.replace('REDUCE', _REDUCE), 'material_energy')
    count = strain.size//6
    out = cp.empty_like(strain); partial = cp.empty((count+255)//256, cp.float64)
    _KERNELS['material']((partial.size,), (256,),
        (descriptor.val, descriptor.material_table, strain, out, np.int64(count), partial))
    return out, float(cp.sum(partial, dtype=cp.float64))


def energy_action(Q, descriptor, z, direction=None, material_timing=None):
    strain = Q.strain(z) if direction is None else Q.strain_updated_direction(z, *direction)
    if material_timing is not None:
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
    stress, pap = material_energy(descriptor, strain)
    if material_timing is not None:
        cp.cuda.get_current_stream().synchronize()
        stats, label = material_timing
        stats[label]['seconds'] += time.perf_counter()-started
        stats[label]['calls'] += 1
    h = fft.rfftn_cached(stress, Q.shape, (1, 2, 3))
    return DeferredHP(Q, h), pap

def energy_update(p, step, alpha, x, r, partial, smoothing=None):
    return step.update(p, alpha, x, r, smoothing)


def supports_energy(Q, descriptor):
    """Raw CUDA contract; all other material layouts use the existing path."""
    ids = getattr(descriptor, 'val', None)
    table = getattr(descriptor, 'material_table', None)
    return (Q.xp is cp and isinstance(ids, cp.ndarray) and isinstance(table, cp.ndarray)
        and ids.dtype == cp.int32 and table.dtype == cp.float32
        and ids.shape in (Q.shape, (1,)+Q.shape) and table.ndim == 2 and table.shape[1] == 21
        and ids.flags.c_contiguous and table.flags.c_contiguous)
