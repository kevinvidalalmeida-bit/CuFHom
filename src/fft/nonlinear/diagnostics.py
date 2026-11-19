"""Bounded, FP64-accumulated diagnostics of an accepted plasticity trial.

The fused GPU path reads fields without materializing deviatoric stresses,
yield violations or hardening sums. It never updates a constitutive state.
"""

import numpy as np
from cufhom.fft.krylov import NonlinearConvergenceError

_COUNT = 22
_MODULES = {}
_CUDA = r"""
#define NSTAT 22
__device__ __forceinline__ bool maximum_stat(int k){return k==17||k==18||k==21;}
__device__ __forceinline__ double merge(double a,double b,int k){return maximum_stat(k)?fmax(a,b):a+b;}
extern "C" __global__ void trial_diagnostics(const REAL* e,const REAL* sig,
 const REAL* old_alpha,const REAL* dp,const REAL* ep,const REAL* direction,
 const int* ids,long long n,double sy,double hardening,double friction,double beta,
 double rtol,int pressure,double* partial,double kinematic,double saturation,double exponent,
 double viscous,int general){
 double a[NSTAT]={0};
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  double s[6],v[6],work=0,norm=0;bool bad=false;
  for(int c=0;c<6;c++){
   s[c]=(double)sig[(long long)c*n+i];v[c]=(double)e[(long long)c*n+i];
   bad|=!isfinite(s[c])||!isfinite(v[c]);a[c]+=v[c];a[6+c]+=s[c];
   work+=s[c]*v[c];norm+=v[c]*v[c];
  }
  const double alpha=(double)old_alpha[i],delta=(double)dp[i],total=alpha+delta;
  bad|=!isfinite(alpha)||!isfinite(delta)||alpha<0||delta<0;
  const double mean=(s[0]+s[1]+s[2])/3.;double f=0;
  if(!ids[i]){
   double q2=0,x[6]={0},xt=0;
   // J2 backstress at n+1, X = 2/3 C dev(ep + sqrt(1.5) u dp).
   if(kinematic!=0){for(int c=0;c<6;c++){x[c]=(double)ep[(long long)c*n+i]+sqrt(1.5)*(double)direction[(long long)c*n+i]*delta;
     if(c<3)xt+=x[c];}
     for(int c=0;c<6;c++)x[c]=(2*kinematic/3)*(x[c]-(c<3?xt/3:0.));}
   for(int c=0;c<6;c++){double d=s[c]-(c<3?mean:0.);if(kinematic!=0)d-=x[c];q2+=d*d;}
   if(general)f=sqrt(1.5*q2)-sy-hardening*total-saturation*(1-exp(-exponent*total))-viscous*delta;
   else f=sqrt(1.5*q2)+friction*mean-sy-hardening*total;
  }
  double irreversible=pressure?(ids[i]?0.:delta*(sy+(beta-friction)*mean)):sy*delta;
  if(viscous!=0)irreversible+=viscous*delta*delta;
  double volume=0;
  if(pressure)for(int c=0;c<3;c++){
   const double p=(double)ep[(long long)c*n+i],u=(double)direction[(long long)c*n+i];
   bad|=!isfinite(p)||!isfinite(u);volume+=p+sqrt(1.5)*u*delta;
  }
  bad|=!isfinite(work)||!isfinite(norm)||!isfinite(f)||!isfinite(irreversible)||!isfinite(volume);
  a[12]+=work;a[13]+=norm;a[14]+=delta;a[15]+=total;a[16]+=(total>1.e-10);
  if(!ids[i])a[17]=fmax(a[17],f);
  a[18]=fmax(a[18],pressure&&irreversible < -rtol*sy*delta?1.:0.);
  a[19]+=irreversible;a[20]+=volume;a[21]=fmax(a[21],bad?1.:0.);
 }
 __shared__ double sums[NSTAT][8];
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int k=0;k<NSTAT;k++){
  double v=a[k];for(int d=16;d;d/=2)v=merge(v,__shfl_down_sync(0xffffffff,v,d),k);
  if(lane==0)sums[k][warp]=v;
 }
 __syncthreads();
 if(threadIdx.x<NSTAT){int k=threadIdx.x;double v=0;for(int w=0;w<8;w++)v=merge(v,sums[k][w],k);
  partial[(long long)k*gridDim.x+blockIdx.x]=v;}
}
extern "C" __global__ void finish_diagnostics(const double* partial,int count,double* out){
 const int k=blockIdx.x;double v=0;
 for(int i=threadIdx.x;i<count;i+=blockDim.x)v=merge(v,partial[(long long)k*count+i],k);
 for(int d=16;d;d/=2)v=merge(v,__shfl_down_sync(0xffffffff,v,d),k);
 __shared__ double sums[8];const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 if(lane==0)sums[warp]=v;__syncthreads();
 if(threadIdx.x==0){v=0;for(int w=0;w<8;w++)v=merge(v,sums[w],k);out[k]=v;}
}
"""


class FusedDiagnostics:
    """Two kernels and one 22-scalar transfer; at most 0.7 MiB workspace."""

    def __init__(self, cp, material, ids, dtype, *, pressure=False, rtol=1e-5):
        self.cp, self.material, self.ids = cp, material, ids
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("Diagnostics require float32 or float64 fields")
        if (
            not isinstance(ids, cp.ndarray)
            or ids.dtype != np.dtype("int32")
            or not ids.flags.c_contiguous
            or ids.ndim != 3
            or not ids.size
        ):
            raise ValueError(
                "Diagnostics require a nonempty contiguous int32 material image"
            )
        if not np.isfinite(rtol) or rtol < 0:
            raise ValueError("Diagnostics require a finite nonnegative tolerance")
        self.shape, self.n = ids.shape, int(ids.size)
        self.pressure, self.rtol = bool(pressure), float(rtol)
        self.blocks = min(4096, (self.n + 255) // 256)
        key = (int(cp.cuda.runtime.getDevice()), self.dtype.str)
        module = _MODULES.get(key)
        if module is None:
            module = _MODULES[key] = cp.RawModule(
                code=_CUDA.replace(
                    "REAL", "float" if self.dtype == np.float32 else "double"
                )
            )
        self.first = module.get_function("trial_diagnostics")
        self.finish = module.get_function("finish_diagnostics")
        self.partial = cp.empty((_COUNT, self.blocks), cp.float64)
        self.result = cp.empty(_COUNT, cp.float64)
        self.calls = 0

    @property
    def workspace_bytes(self):
        """Return the allocated device bytes used by the audit reduction workspace."""
        return int(self.partial.nbytes + self.result.nbytes)

    def evaluate(self, strain, response, state, viscous=0.0):
        """viscous: overstress modulus of the accepted J2 step (yield audit q - R = viscous*dp)."""
        cp, m = self.cp, self.material
        kinematic = 0.0 if self.pressure else m.kinematic_hardening
        fields = self.pressure or kinematic != 0
        general = not self.pressure and (m.nonlinear_hardening or viscous != 0)
        arrays = [strain, response.stress, state.equivalent, response.dp]
        if fields:
            arrays += [state.plastic_strain, response.direction]
        for field in arrays:
            if (
                not isinstance(field, cp.ndarray)
                or field.dtype != self.dtype
                or not field.flags.c_contiguous
            ):
                raise ValueError(
                    "Diagnostics need contiguous fields in the configured precision"
                )
        if (
            strain.shape != (6, *self.shape)
            or response.stress.shape != strain.shape
            or state.equivalent.shape != self.shape
            or response.dp.shape != self.shape
            or fields
            and (
                state.plastic_strain.shape != strain.shape
                or response.direction.shape != strain.shape
            )
        ):
            raise ValueError("Diagnostics field shape mismatch")
        self.first(
            (self.blocks,),
            (256,),
            (
                strain,
                response.stress,
                state.equivalent,
                response.dp,
                state.plastic_strain if fields else np.uint64(0),
                response.direction if fields else np.uint64(0),
                self.ids,
                np.int64(self.n),
                np.float64(m.yield_stress),
                np.float64(m.hardening),
                np.float64(m.friction if self.pressure else 0.0),
                np.float64(m.beta if self.pressure else 0.0),
                np.float64(self.rtol),
                np.int32(self.pressure),
                self.partial,
                np.float64(kinematic),
                np.float64(0.0 if self.pressure else m.saturation),
                np.float64(m.saturation_exponent),
                np.float64(viscous),
                np.int32(general),
            ),
        )
        self.finish(
            (_COUNT,), (256,), (self.partial, np.int32(self.blocks), self.result)
        )
        values = cp.asnumpy(self.result)
        self.calls += 1
        if not np.isfinite(values).all() or values[21] != 0:
            raise NonlinearConvergenceError(
                "Nonfinite or inadmissible plasticity diagnostic field"
            )
        if values[18] != 0:
            raise NonlinearConvergenceError("Negative local Drucker-Prager dissipation")
        violation = float(values[17] / m.yield_stress)
        values /= self.n
        return dict(
            mean_strain=values[:6],
            mean_stress=values[6:12],
            stress_work=float(values[12]),
            strain_norm=float(np.sqrt(values[13])),
            equivalent_mean=float(values[15]),
            plastic_fraction=float(values[16]),
            yield_violation=violation,
            dissipation=float(values[19]),
            plastic_volumetric_mean=float(values[20]),
        )
