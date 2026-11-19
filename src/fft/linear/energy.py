"""Energy denominator and adjoint/update, without an Hp buffer.

H itself and the true-residual audit retain the production implementation.
The CG recurrence is unchanged; only the order of scalar summation changes.
"""

import numpy as np
from cufhom_core.tensors import fft
from cufhom_core.tensors.fft import cp

from .precision import cuda_source, real_dtype

_KERNELS = {}
_STEP = r"""
extern "C" __global__ void energy_step(double numerator,
 const double* denominator,float* step) {
 const double pap=denominator[0];
 // Compute the FP64 division once, rather than once per voxel/thread.
 const float a=(float)(numerator/pap);
 step[0]=(isfinite(pap)&&pap>0&&isfinite(numerator)&&numerator>0&&isfinite(a)&&a>0)
     ? a : nanf("");
}
"""
_REDUCE = r"""
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
"""
_MATERIAL = r"""
extern "C" __global__ void material_energy(const int* ids,const float* table,
 const float* x,float* y,long long count,long long materials,double* partial) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 double sum=0;
 if(s<count) {
   const int id=ids[s];
   if(id<0||id>=materials) {
     for(int c=0;c<6;c++)y[c*count+s]=nanf("");
     sum=nan("");
   } else {
   const float* a=table+21*(long long)id;
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
 }
 REDUCE
}
"""
_ADJOINT = r"""
extern "C" __global__ void adjoint_update(const float2* h,const float2* p,
 float2* x,float2* r,double* partial,
 long long n0,long long n1,long long n2,long long nz,
 float iy0,float iy1,float iy2,float norm,float transverse,float longitudinal,
 const float* step,float2* y,float blend) {
 const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 // A failed energy check must poison the reduced norm, including masked
 // frequencies, so the host cannot accept a nonpositive/invalid CG step.
 const float alpha=step[0];
 double sum=isfinite(alpha)?0:nan("");
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
"""


def adjoint_update_source(smoothing, dtype=np.float32):
    # Reuse the established coordinate formula. Its output is thread-local;
    # no Hp vector is stored. Every wrapper thread reaches the reduction.
    """Generate the fused coordinate-adjoint and energy-update CUDA source."""
    from cufhom.fft.linear.krylov import coordinate_source, replace_once

    code = coordinate_source()
    code = replace_once(
        code,
        'extern "C" __global__ void coordinates',
        "__device__ __forceinline__ void coordinate_local",
    )
    code = replace_once(code, "y[3*count+j++]", "y[j++]")
    if code.count("y[c*count+s]") != 2 or code.count("y[c*count]") != 2:
        raise RuntimeError("Coordinate output changed: review energy/update fusion")
    code = code.replace("y[c*count+s]", "y[c]")
    code = code.replace("y[c*count]", "y[c]")
    code = (
        code.replace("COMPLEX", "float2")
        .replace("REAL", "float")
        .replace("MAKE", "make_float2")
    )
    smooth = (
        ""
        if not smoothing
        else "float2 yi=y[j];yi.x+=blend*(xi.x-yi.x);yi.y+=blend*(xi.y-yi.y);y[j]=yi;"
    )
    update = _ADJOINT.replace("SMOOTH", smooth).replace("REDUCE", _REDUCE)
    return cuda_source(code + update, dtype)


class DeferredHP:
    """Stress spectrum of H p, consumed by the fused adjoint/update kernel."""

    def __init__(self, Q, spectrum):
        self.Q, self.spectrum = Q, spectrum

    def update(self, p, numerator, x, r, smoothing, *, denominator):
        """Update the iterate and residual using a device FP64 energy quotient.

        x += alpha p, r -= alpha RQ* spectrum with alpha = numerator/denominator
        divided once in FP64 on the device; returns the new |r|^2.
        """
        if (
            not isinstance(denominator, cp.ndarray)
            or denominator.shape != ()
            or denominator.dtype != cp.float64
        ):
            raise ValueError("Expected a device float64 energy scalar")
        dtype = p.real.dtype
        scalar = dtype.type
        mode = (
            "adjoint_device" if smoothing is None else "adjoint_smooth_device",
            dtype.str,
        )
        if mode not in _KERNELS:
            _KERNELS[mode] = cp.RawKernel(
                adjoint_update_source(smoothing is not None, dtype), "adjoint_update"
            )
        step_key = ("step", dtype.str)
        if step_key not in _KERNELS:
            _KERNELS[step_key] = cp.RawKernel(cuda_source(_STEP, dtype), "energy_step")
        Q = self.Q
        blocks = (Q.count + 255) // 256
        partial = cp.empty(blocks, cp.float64)
        step = cp.empty((), dtype)
        _KERNELS[step_key]((1,), (1,), (np.float64(numerator), denominator, step))
        y, blend = (x, 0.0) if smoothing is None else smoothing
        _KERNELS[mode](
            (blocks,),
            (256,),
            (
                self.spectrum,
                p,
                x,
                r,
                partial,
                *(np.int64(n) for n in Q.shape),
                np.int64(Q.hshape[-1]),
                *(scalar(1 / L) for L in Q.P.lengths),
                scalar(Q.norm),
                scalar(Q.transverse),
                scalar(Q.longitudinal),
                step,
                y,
                scalar(blend),
            ),
        )
        norm = float(cp.sum(partial, dtype=cp.float64))
        # The scalar reduction synchronizes this stream. Hp is consumed;
        # release its six-component spectrum before building the next H.
        self.spectrum = None
        return norm


def material_energy(descriptor, strain):
    """Stress C:strain and the FP64 device energy strain:C:strain."""
    dtype = real_dtype(strain.dtype)
    if descriptor.material_table.dtype != dtype:
        raise ValueError("Material and strain must have the same precision")
    key = ("material", dtype.str)
    if key not in _KERNELS:
        _KERNELS[key] = cp.RawKernel(
            cuda_source(_MATERIAL.replace("REDUCE", _REDUCE), dtype), "material_energy"
        )
    count = strain.size // 6
    out = cp.empty_like(strain)
    partial = cp.empty((count + 255) // 256, cp.float64)
    _KERNELS[key](
        (partial.size,),
        (256,),
        (
            descriptor.val,
            descriptor.material_table,
            strain,
            out,
            np.int64(count),
            np.int64(descriptor.material_table.shape[0]),
            partial,
        ),
    )
    return out, cp.sum(partial, dtype=cp.float64)


def energy_action(Q, descriptor, z, direction=None):
    """H z as a deferred spectrum, and the device energy z:Hz."""
    strain = (
        Q.strain(z) if direction is None else Q.strain_updated_direction(z, *direction)
    )
    stress, pap = material_energy(descriptor, strain)
    del strain
    h = fft.rfftn_cached(stress, Q.shape, (1, 2, 3))
    return DeferredHP(Q, h), pap


def energy_update_device(p, step, numerator, denominator, x, r, smoothing=None):
    """Divide FP64 scalars on the GPU; read only the new norm back.

    The material-energy sum stays on the current stream while cuFFT computes
    the stress spectrum. Invalid denominators produce a nonfinite norm and
    are rejected by CG. Field arithmetic and true-residual audits are intact.
    """
    return step.update(p, numerator, x, r, smoothing, denominator=denominator)


def supports_energy(Q, descriptor):
    """Raw CUDA contract; all other material layouts use the existing path."""
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom_core.tensors import Tensor

    if (
        cp is None
        or type(Q) is not SymmetricCoordinates
        or type(descriptor) is not Tensor
        or descriptor.Fourier
        or descriptor.multype not in ("sym21_indexed", "indexed_sym21")
    ):
        return False
    ids = getattr(descriptor, "val", None)
    table = getattr(descriptor, "material_table", None)
    return (
        Q.xp is cp
        and isinstance(ids, cp.ndarray)
        and isinstance(table, cp.ndarray)
        and ids.dtype == cp.int32
        and table.dtype in (cp.float32, cp.float64)
        and ids.shape in (Q.shape, (1,) + Q.shape)
        and table.ndim == 2
        and table.shape[0] > 0
        and table.shape[1] == 21
        and np.array_equal(descriptor.N, Q.shape)
        and descriptor.fft_form == "r"
        and ids.flags.c_contiguous
        and table.flags.c_contiguous
    )
