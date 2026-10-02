"""Archived prototype of the integrated energy denominator and update.

H itself and the true-residual audit retain the production implementation.
The CG recurrence is unchanged; only the order of scalar summation changes.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc, linear_krylov as lk
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
   long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
   float2 ap[3]; for(int c=0;c<3;c++)ap[c]=make_float2(0,0);
   if(s!=0&&!((n0%2==0&&i0==n0/2)||(n1%2==0&&i1==n1/2)||(n2%2==0&&i2==n2/2))) {
     const float sq2=1.4142135623730950488f;
     float k[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
     float inv=1/sqrtf(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
     for(int c=0;c<3;c++)k[c]*=inv;
     float scale=norm*((i2==0||(n2%2==0&&i2==n2/2))?1:sq2);
     for(int part=0;part<2;part++) {
       float z[6],v[3];for(int c=0;c<6;c++)z[c]=part?h[c*count+s].y:h[c*count+s].x;
       v[0]=k[0]*z[0]+(k[1]*z[5]+k[2]*z[4])/sq2;
       v[1]=k[1]*z[1]+(k[0]*z[5]+k[2]*z[3])/sq2;
       v[2]=k[2]*z[2]+(k[0]*z[4]+k[1]*z[3])/sq2;
       float dot=k[0]*v[0]+k[1]*v[1]+k[2]*v[2];
       for(int c=0;c<3;c++) {
         float a=scale*(sq2*transverse*v[c]+(longitudinal-sq2*transverse)*k[c]*dot);
         if(part)ap[c].x=a;else ap[c].y=-a;
       }
     }
   }
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


class DeferredHP:
    def __init__(self, Q, spectrum):
        self.Q, self.spectrum = Q, spectrum

    def update(self, p, alpha, x, r, smoothing):
        mode = 'adjoint' if smoothing is None else 'adjoint_smooth'
        if mode not in _KERNELS:
            smooth = ('' if smoothing is None else
                'float2 yi=y[j];yi.x+=blend*(xi.x-yi.x);yi.y+=blend*(xi.y-yi.y);y[j]=yi;')
            _KERNELS[mode] = cp.RawKernel(_ADJOINT.replace('SMOOTH', smooth).replace('REDUCE', _REDUCE), 'adjoint_update')
        Q = self.Q
        blocks = (Q.count+255)//256
        partial = cp.empty(blocks, cp.float64)
        y, blend = (x, 0.) if smoothing is None else smoothing
        args = (self.spectrum, p, x, r, partial, *(np.int64(n) for n in Q.shape), np.int64(Q.hshape[-1]),
                *(np.float32(1/L) for L in Q.P.lengths), np.float32(Q.norm),
                np.float32(Q.transverse), np.float32(Q.longitudinal), np.float32(alpha), y, np.float32(blend))
        _KERNELS[mode]((blocks,), (256,), args)
        return float(cp.sum(partial, dtype=cp.float64))


def material_energy(descriptor, strain):
    if 'material' not in _KERNELS:
        _KERNELS['material'] = cp.RawKernel(_MATERIAL.replace('REDUCE', _REDUCE), 'material_energy')
    count = strain.size//6
    out = cp.empty_like(strain); partial = cp.empty((count+255)//256, cp.float64)
    _KERNELS['material']((partial.size,), (256,),
        (descriptor.val, descriptor.material_table, strain, out, np.int64(count), partial))
    return out, float(cp.sum(partial, dtype=cp.float64))


def energy_action(Q, descriptor, z, direction=None):
    strain = Q.strain(z) if direction is None else Q.strain_updated_direction(z, *direction)
    stress, pap = material_energy(descriptor, strain)
    h = fft.rfftn_cached(stress, Q.shape, (1, 2, 3))
    return DeferredHP(Q, h), pap


@contextmanager
def energy_context():
    original_cg, original_update = lc.fused_cg, lk.fused_update
    def cg(action, rhs, dot, *args, **kw):
        captures = dict(zip(action.__code__.co_freevars, (c.cell_contents for c in action.__closure__)))
        Q, descriptor = captures['Q'], getattr(captures['A'], 'op', captures['A'])
        kw['action_and_dot'] = lambda z, direction=None: energy_action(Q, descriptor, z, direction)
        return original_cg(action, rhs, dot, *args, **kw)
    def update(p, ap, alpha, x, r, partial, smoothing=None):
        return (ap.update(p, alpha, x, r, smoothing) if isinstance(ap, DeferredHP)
                else original_update(p, ap, alpha, x, r, partial, smoothing))
    with patch.object(lc, 'fused_cg', cg), patch.object(lk, 'fused_update', update):
        yield
