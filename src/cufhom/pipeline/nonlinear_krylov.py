"""Fused coordinate PCG, preserving the physical/isometric residual norm.

The isotropic reference is unchanged. Its action is evaluated while updating
the direction, and r.r / r.M^-1.r are reduced together during the x/r update.
No extra preconditioned vector is streamed to global memory each iteration.
"""
import numpy as np
from .fft_solver import cp

_MODULE = None
_CODE = r'''
__device__ __forceinline__ void reference3(const float2* r, float2* z,
 long long s,long long n0,long long n1,long long nz,float iy0,float iy1,float iy2,
 float a,float b) {
 if(s==0){for(int c=0;c<3;c++)z[c]=make_float2(0,0);return;}
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 float k[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
 float inv=1/(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]),re=0,im=0;
 for(int c=0;c<3;c++){re+=k[c]*r[c].x;im+=k[c]*r[c].y;}
 for(int c=0;c<3;c++)z[c]=make_float2(a*r[c].x+(b-a)*k[c]*re*inv,
                                   a*r[c].y+(b-a)*k[c]*im*inv);
}
__device__ __forceinline__ void finish2(double rr,double rz,double* out) {
 for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);rz+=__shfl_down_sync(0xffffffff,rz,s);}
 __shared__ double rr_w[8],rz_w[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0){rr_w[warp]=rr;rz_w[warp]=rz;}
 __syncthreads();
 if(warp==0){
   rr=lane<8?rr_w[lane]:0;rz=lane<8?rz_w[lane]:0;
   for(int s=16;s;s>>=1){rr+=__shfl_down_sync(0xffffffff,rr,s);rz+=__shfl_down_sync(0xffffffff,rz,s);}
   if(lane==0){out[blockIdx.x]=rr;out[gridDim.x+blockIdx.x]=rz;}
 }
}
extern "C" __global__ void update_energy(const float2* p,const float2* ap,
 float alpha,float2* x,float2* r,double* partial,long long n0,long long n1,long long nz,
 float iy0,float iy1,float iy2,float a,float b,const float* mean,int free_mask) {
 long long count=n0*n1*nz;
 double rr=0,rz=0;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;
     s+=(long long)gridDim.x*blockDim.x){
   float2 rc[3],z[3];
   for(int c=0;c<3;c++){
     long long j=c*count+s;
     float2 xi=x[j],ri=r[j],pi=p[j],api=ap[j];
     xi.x+=alpha*pi.x;xi.y+=alpha*pi.y;
     ri.x-=alpha*api.x;ri.y-=alpha*api.y;
     x[j]=xi;r[j]=rc[c]=ri;
   }
   reference3(rc,z,s,n0,n1,nz,iy0,iy1,iy2,a,b);
   for(int c=0;c<3;c++){
     rr+=(double)rc[c].x*rc[c].x+(double)rc[c].y*rc[c].y;
     rz+=(double)rc[c].x*z[c].x+(double)rc[c].y*z[c].y;
   }
   if(s==0){
     float rm[6];int i=0;
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       long long j=3*count+i++;
       float2 xi=x[j],ri=r[j],pi=p[j],api=ap[j];
       xi.x+=alpha*pi.x;xi.y+=alpha*pi.y;
       ri.x-=alpha*api.x;ri.y-=alpha*api.y;
       x[j]=xi;r[j]=ri;rm[c]=ri.x;
       rr+=(double)ri.x*ri.x+(double)ri.y*ri.y;
     }
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       float v=0;
       for(int d=0;d<6;d++)if(free_mask&(1<<d))v+=mean[6*c+d]*rm[d];
       rz+=(double)rm[c]*v;
     }
   }
 }
 finish2(rr,rz,partial);
}
extern "C" __global__ void direction(const float2* r,float beta,float2* p,
 long long n0,long long n1,long long nz,float iy0,float iy1,float iy2,
 float a,float b,const float* mean,int free_mask) {
 long long count=n0*n1*nz;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;
     s+=(long long)gridDim.x*blockDim.x){
   float2 rc[3],z[3];
   for(int c=0;c<3;c++)rc[c]=r[c*count+s];
   reference3(rc,z,s,n0,n1,nz,iy0,iy1,iy2,a,b);
   for(int c=0;c<3;c++){
     long long j=c*count+s;float2 v=p[j];
     p[j]=make_float2(z[c].x+beta*v.x,z[c].y+beta*v.y);
   }
   if(s==0){
     int i=0;
     for(int c=0;c<6;c++)if(free_mask&(1<<c)){
       float v=0;int j=0;
       for(int d=0;d<6;d++)if(free_mask&(1<<d))v+=mean[6*c+d]*r[3*count+j++].x;
       long long index=3*count+i++;float2 old=p[index];
       p[index]=make_float2(v+beta*old.x,beta*old.y);
     }
   }
 }
}
'''


class FusedCoordinatePCG:
    """Fixed reference per solve; supports the free mean modes and PTC shift."""
    def __init__(self, coordinates):
        global _MODULE
        if cp is None or coordinates.xp is not cp:
            raise ValueError('Fused coordinate PCG requires CuPy')
        self.Q = coordinates
        self.blocks = min(16384, (coordinates.count+255)//256)
        self.partial = cp.empty((2, self.blocks), dtype=cp.float64)
        self.geometry = (*(np.int64(n) for n in coordinates.hshape),
                         *(np.float32(1/L) for L in coordinates.P.lengths))
        self.mask = np.int32(sum(1<<i for i in coordinates.P.free))
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_CODE)
        self.update_kernel = _MODULE.get_function('update_energy')
        self.direction_kernel = _MODULE.get_function('direction')

    def _reference_args(self, reference):
        if reference.P is not self.Q.P or reference.mean.dtype != cp.float32:
            raise ValueError('Reference must share the projector and use float32')
        return (np.float32(reference.a), np.float32(reference.b), reference.mean, self.mask)

    def update(self, p, ap, alpha, x, r, reference):
        self.update_kernel((self.blocks,), (256,),
            (p, ap, np.float32(alpha), x, r, self.partial,
             *self.geometry, *self._reference_args(reference)))
        return cp.asnumpy(cp.sum(self.partial, axis=1, dtype=cp.float64))

    def direction(self, r, beta, p, reference):
        self.direction_kernel((self.blocks,), (256,),
            (r, np.float32(beta), p, *self.geometry, *self._reference_args(reference)))

    def solve(self, action, rhs, dot, tolerance, maxiter, reference):
        from .nonlinear_solver import NonlinearConvergenceError
        if (rhs.dtype != cp.complex64 or rhs.ndim != 1 or rhs.size != self.Q.size
                or not rhs.flags.c_contiguous):
            raise ValueError('Expected contiguous complex64 coordinates on the configured grid')
        x, r = cp.zeros_like(rhs), rhs.copy()
        rr = dot(r, r)
        if not np.isfinite(rr):
            raise NonlinearConvergenceError('Nonfinite PCG RHS')
        if rr == 0:
            return x, 0
        limit = tolerance*tolerance*rr
        p = self.Q.precondition(r, reference)
        rz = dot(r, p)
        for iteration in range(maxiter):
            ap = action(p)
            pap = dot(p, ap)
            if not np.isfinite([pap,rz]).all() or min(pap,rz) <= 0:
                raise NonlinearConvergenceError('Fused coordinate PCG lost positivity')
            rr, new_rz = self.update(p, ap, rz/pap, x, r, reference)
            if not np.isfinite([rr,new_rz]).all():
                raise NonlinearConvergenceError('Nonfinite fused coordinate residual')
            if rr <= limit:
                r = rhs-action(x)
                rr = dot(r, r)
                if not np.isfinite(rr):
                    raise NonlinearConvergenceError('Nonfinite true coordinate residual')
                if rr <= limit:
                    return x, iteration+1
                p = self.Q.precondition(r, reference)
                new_rz = dot(r, p)
            else:
                self.direction(r, new_rz/rz, p, reference)
            rz = new_rz
        raise NonlinearConvergenceError(f'Fused coordinate PCG failed true residual after {maxiter} iterations')
