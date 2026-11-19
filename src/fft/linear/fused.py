"""Fused spectral pipeline for displacement CG (FP32/FP64 GPU).

Used by the linear solver and, with the J2 consistent tangent as material
and free mean strains, by the Newton solves of the nonlinear solver.

A staged H application materializes intermediate six-component fields
between coordinate maps, transforms and the material action. This pipeline
reduces those writes and separate vector passes. Its data-flow counts are
logical traffic estimates; DRAM traffic and a bandwidth limit require
profiling rather than inference from the number of passes alone.

Here the 3-D transform is split by axis and fused with its neighbours:

  KA_inv  deferred x += alpha p, p <- r+beta*p, strain map QR p,
          inverse FFT along axis 0
  KB_inv  inverse FFT along axis 1 (in place)
  KC      C2R along axis 2, material law, FP64 energy, R2C along axis 2
  KB_fwd  forward FFT along axis 1 (in place)
  KA_fwd  forward FFT along axis 0, adjoint RQ*, alpha, r update, FP64 |r|^2

Hybrid spectra use one internal layout, (n0, n2/2 in tiles of tb, n1, tile,
components) complex64/complex128: KB reads and writes each (x0, tile) as one contiguous
block, KC gathers a line from n2/(2 tb) short segments in L2. On even grids
the z-Nyquist plane is packed into the k2 = 0 plane (Q = z0 + i zM, same
wavevector); odd grids store (n2+1)/2 planes and pair lines in KC. Line FFTs are mixed-radix Stockham
transforms in shared memory with FP64-generated twiddles. The discretization,
frequencies, coordinates, material table, CG recurrence and FP64 reduction
accumulators are unchanged. Local energy arithmetic uses working precision;
FFT rounding and the order of arithmetic differ from the staged action.
Production coordinate maps can read/write tile-major vectors directly for
the linear RHS, audits and final strain. Explicit conversions remain for
reference comparisons and the nonlinear interface.
"""

import contextvars
import hashlib
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from math import gcd

import numpy as np
from cufhom.fft.linear.energy import supports_energy
from cufhom.fft.linear.kernel_cache import (
    cache_path,
    device_properties,
    read_records,
    save_record,
)
from cufhom.fft.linear.krylov import _finish_smoothing
from cufhom_core.tensors.fft import cp

from .common import FreshProgress, LinearSolveError, stored_planes
from .precision import cuda_source, real_dtype

_MAX_PRIME = 13
_KERNELS = {}
_PIPELINE_PARTS = {}
_SOURCE_REAL = contextvars.ContextVar("linear_cuda_real", default=np.float32)
_FREQUENCY_NORM_MAX_BYTES = 64 * 2**20
_GEOMETRY_CACHE = OrderedDict()


def _geometry_for_pipeline(key, parts, shape, lengths):
    """Bound the reusable geometry across grids; active pipelines own theirs."""
    norm = parts.get("frequency_norm")
    if norm is None:
        norm = _frequency_inverse_norm(shape, lengths)
        if norm is not None:
            used = sum(
                item["frequency_norm"].nbytes
                for item in _GEOMETRY_CACHE.values()
                if item.get("frequency_norm") is not None
            )
            while _GEOMETRY_CACHE and used + norm.nbytes > _FREQUENCY_NORM_MAX_BYTES:
                _, old = _GEOMETRY_CACHE.popitem(last=False)
                previous = old.pop("frequency_norm", None)
                if previous is not None:
                    used -= previous.nbytes
            parts["frequency_norm"] = norm
    if norm is not None:
        _GEOMETRY_CACHE[key] = parts
        _GEOMETRY_CACHE.move_to_end(key)
    return norm


def _frequency_inverse_norm(shape, lengths):
    """Bounded FP64 geometry cache, shared by loads/materials of one grid.

    Larger grids retain on-the-fly normalization so this optimization does
    not raise their working-set requirement. Excluded modes are zero.
    """
    n0, n1, n2 = shape
    m = stored_planes(n2)
    count = n0 * n1 * m
    if count * 8 > _FREQUENCY_NORM_MAX_BYTES:
        return None
    key = "frequency_inverse_norm"
    if key not in _KERNELS:
        _KERNELS[key] = cp.RawKernel(
            r"""
extern "C" __global__ void frequency_inverse_norm(double* out,long long n0,long long n1,
 long long m,double iy0,double iy1,double iy2) {
 const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(s>=n0*n1*m)return;
 const long long i0=s/(n1*m),i1=(s/m)%n1,i2=s%m;
 const bool z0=i0==0||(n0%2==0&&i0==n0/2),z1=i1==0||(n1%2==0&&i1==n1/2);
 const double k0=(z0?0:(i0<=n0/2?i0:i0-n0))*iy0,k1=(z1?0:(i1<=n1/2?i1:i1-n1))*iy1,k2=i2*iy2;
 out[s]=(z0&&z1&&i2==0)?0.:1./sqrt(k0*k0+k1*k1+k2*k2);
}
""",
            key,
        )
    out = cp.empty(count, cp.float64)
    _KERNELS[key](
        ((count + 255) // 256,),
        (256,),
        (
            out,
            np.int64(n0),
            np.int64(n1),
            np.int64(m),
            *(np.float64(1 / L) for L in lengths),
        ),
    )
    return out


def _cached_geometry_source(source):
    """Add the bounded lookup to FP64 axis-0 kernels only."""
    n2 = int(re.search(r"#define N2 (\d+)", source).group(1))
    planes = "(N2/2)" if n2 % 2 == 0 else str(stored_planes(n2))
    source = source.replace(
        "float k[3]) {", "float k[3],const float* frequency_norm) {"
    )
    original = "const float inv=1/sqrtf(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);"
    if source.count(original) != 1:
        raise RuntimeError(
            "Review changed frequency normalization before specializing FP64"
        )
    source = source.replace(
        original,
        f"const float inv=frequency_norm?frequency_norm[((long long)i0*N1+i1)*{planes}+i2]"
        ":1/sqrtf(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);",
    )
    return (
        source.replace(
            "float longitudinal) {", "float longitudinal,const float* frequency_norm) {"
        )
        .replace("unit_k(i0,i1,i2,k)", "unit_k(i0,i1,i2,k,frequency_norm)")
        .replace(
            "transverse,longitudinal);", "transverse,longitudinal,frequency_norm);"
        )
    )


def kc_shared_complex(n2, lpb):
    """Complex words of KC shared memory: the line buffer and its twiddles."""
    return n2 // 2 * 6 * lpb + 2 * (n2 // 2) if n2 % 2 == 0 else n2 * 3 * lpb + n2


def radices(n):
    """Radix plan: powers of two as 8/4/2, then odd prime factors."""
    out, m, e = [], int(n), 0
    while m % 2 == 0:
        m //= 2
        e += 1
    a, b = divmod(e, 3)
    if b == 1 and a >= 1:
        out += [8] * (a - 1) + [4, 4]
    else:
        out += [8] * a + ([4] if b == 2 else [2] if b == 1 else [])
    p = 3
    while m > 1:
        while m % p == 0:
            out.append(p)
            m //= p
        p += 2
    return out


def _f(x):
    scalar = _SOURCE_REAL.get()
    literal = getattr(scalar, "literal", None)  # double-float pairs (evaluation.py)
    if literal is not None:
        return literal(x)
    return repr(float(scalar(x))) + ("f" if scalar is np.float32 else "")


def _dft(R, sign, v):
    """In-place DFT of float2 registers v[0..R-1]; sign=-1 is forward."""
    if R == 2:
        return (
            f"{{float2 a={v}[0],b={v}[1];{v}[0]=make_float2(a.x+b.x,a.y+b.y);"
            f"{v}[1]=make_float2(a.x-b.x,a.y-b.y);}}"
        )
    if R == 4:
        t3 = "-d.y,d.x" if sign > 0 else "d.y,-d.x"
        return f"""{{float2 a0={v}[0],a1={v}[1],a2={v}[2],a3={v}[3];
 float2 t0=make_float2(a0.x+a2.x,a0.y+a2.y),t1=make_float2(a0.x-a2.x,a0.y-a2.y);
 float2 t2=make_float2(a1.x+a3.x,a1.y+a3.y),d=make_float2(a1.x-a3.x,a1.y-a3.y);
 float2 t3=make_float2({t3});
 {v}[0]=make_float2(t0.x+t2.x,t0.y+t2.y);{v}[2]=make_float2(t0.x-t2.x,t0.y-t2.y);
 {v}[1]=make_float2(t1.x+t3.x,t1.y+t3.y);{v}[3]=make_float2(t1.x-t3.x,t1.y-t3.y);}}"""
    if R == 8:
        code = [
            "{float2 e[4],o[4];",
            f"e[0]={v}[0];e[1]={v}[2];e[2]={v}[4];e[3]={v}[6];"
            f"o[0]={v}[1];o[1]={v}[3];o[2]={v}[5];o[3]={v}[7];",
            _dft(4, sign, "e"),
            _dft(4, sign, "o"),
        ]
        for k in range(4):
            wr, wi = np.cos(np.pi * k / 4), sign * np.sin(np.pi * k / 4)
            if k == 0:
                code.append("{float2 t=o[0];")
            elif _SOURCE_REAL.get() is np.float64:
                # Exact eighth roots: quarter-turns need no multiplications;
                # diagonal roots need two instead of four. Keep FP32's
                # established rounding path and generated source unchanged.
                if k == 2:
                    real, imag = (
                        ("-o[2].y", "o[2].x") if sign > 0 else ("o[2].y", "-o[2].x")
                    )
                else:
                    root = _f(np.sqrt(0.5))
                    if (k, sign) == (1, 1):
                        real, imag = "(o[1].x-o[1].y)", "(o[1].x+o[1].y)"
                    elif k == 1:
                        real, imag = "(o[1].x+o[1].y)", "(o[1].y-o[1].x)"
                    elif sign > 0:
                        real, imag = "(-o[3].x-o[3].y)", "(o[3].x-o[3].y)"
                    else:
                        real, imag = "(o[3].y-o[3].x)", "(-o[3].x-o[3].y)"
                    real, imag = root + "*" + real, root + "*" + imag
                code.append(f"{{float2 t=make_float2({real},{imag});")
            else:
                code.append(
                    f"{{float2 t=make_float2({_f(wr)}*o[{k}].x-({_f(wi)})*o[{k}].y,"
                    f"{_f(wr)}*o[{k}].y+({_f(wi)})*o[{k}].x);"
                )
            code.append(
                f"{v}[{k}]=make_float2(e[{k}].x+t.x,e[{k}].y+t.y);"
                f"{v}[{k + 4}]=make_float2(e[{k}].x-t.x,e[{k}].y-t.y);}}"
            )
        return "\n".join(code + ["}"])
    # Odd prime: symmetric real/imaginary pairing halves the multiplications.
    h = (R - 1) // 2
    code = ["{float2 a0=" + v + "[0];"]
    for r in range(1, h + 1):
        code.append(
            f"float2 s{r}=make_float2({v}[{r}].x+{v}[{R - r}].x,{v}[{r}].y+{v}[{R - r}].y);"
            f"float2 d{r}=make_float2({v}[{r}].x-{v}[{R - r}].x,{v}[{r}].y-{v}[{R - r}].y);"
        )
    code.append(
        f"{v}[0]=make_float2(a0.x"
        + "".join(f"+s{r}.x" for r in range(1, h + 1))
        + ",a0.y"
        + "".join(f"+s{r}.y" for r in range(1, h + 1))
        + ");"
    )
    for k in range(1, h + 1):
        cs = [_f(np.cos(2 * np.pi * r * k / R)) for r in range(1, h + 1)]
        sn = [_f(np.sin(2 * np.pi * r * k / R)) for r in range(1, h + 1)]
        A = "+".join(["a0.x"] + [f"{c}*s{r}.x" for r, c in enumerate(cs, 1)])
        Ai = "+".join(["a0.y"] + [f"{c}*s{r}.y" for r, c in enumerate(cs, 1)])
        B = "+".join(f"{s}*d{r}.x" for r, s in enumerate(sn, 1))
        Bi = "+".join(f"{s}*d{r}.y" for r, s in enumerate(sn, 1))
        code.append(f"{{float2 A=make_float2({A},{Ai});float2 B=make_float2({B},{Bi});")
        if sign > 0:
            code.append(
                f"{v}[{k}]=make_float2(A.x-B.y,A.y+B.x);{v}[{R - k}]=make_float2(A.x+B.y,A.y-B.x);}}"
            )
        else:
            code.append(
                f"{v}[{k}]=make_float2(A.x+B.y,A.y-B.x);{v}[{R - k}]=make_float2(A.x-B.y,A.y+B.x);}}"
            )
    return "\n".join(code + ["}"])


def _fft(L, W, NT, sign):
    """Unnormalized in-place FFT of W interleaved length-L transforms.

    buf[pos*W+w] in shared memory, tw[k]=exp(-2 pi i k/L). Stockham stages
    stage their inputs in registers, so one buffer suffices. The caller
    synchronizes before; the code ends synchronized.
    """
    code, Ns = [], 1
    for R in radices(L):
        NB = W * (L // R)
        K = -(-NB // NT)
        code.append(f"{{float2 v[{K}][{R}];")
        code.append(
            f"#pragma unroll\nfor(int q=0;q<{K};q++){{const int b=tid+q*{NT};if(b<{NB}){{"
            f"const int w=b%{W},j=b/{W};"
        )
        code.append(
            f"#pragma unroll\nfor(int r=0;r<{R};r++)v[q][r]=buf[(j+r*{L // R})*{W}+w];}}}}"
        )
        code.append("__syncthreads();")
        code.append(
            f"#pragma unroll\nfor(int q=0;q<{K};q++){{const int b=tid+q*{NT};if(b<{NB}){{"
            f"const int w=b%{W},j=b/{W},m=j%{Ns};"
        )
        if Ns > 1:
            conj = "" if sign < 0 else "-"
            code.append(
                f"#pragma unroll\nfor(int r=1;r<{R};r++){{float2 t=tw[r*m*{L // (Ns * R)}];t.y={conj}t.y;"
                f"const float2 a=v[q][r];v[q][r]=make_float2(a.x*t.x-a.y*t.y,a.x*t.y+a.y*t.x);}}"
            )
        code.append(_dft(R, sign, "v[q]"))
        code.append(f"const int d=(j/{Ns})*{Ns * R}+m;")
        code.append(
            f"#pragma unroll\nfor(int r=0;r<{R};r++)buf[(d+r*{Ns})*{W}+w]=v[q][r];}}}}"
        )
        code.append("__syncthreads();}")
        Ns *= R
    return "\n".join(code)


# Block partials, then the last block sums them in a fixed partition and tree:
# the device scalar is independent of block scheduling (deterministic).
_REDUCE = r"""
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double sums[32];
 __shared__ bool last;
 if((threadIdx.x&31)==0)sums[threadIdx.x>>5]=sum;
 __syncthreads();
 if(threadIdx.x<32) {
   sum=threadIdx.x<(blockDim.x>>5)?sums[threadIdx.x]:0;
   for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
   if(threadIdx.x==0) {
     partial[RB]=sum;
     __threadfence();
     last=atomicAdd(counter,1u)==RT-1;
   }
 }
 __syncthreads();
 if(last) {
   sum=0;
   // L2 loads (__ldcg) see the other blocks' fenced partials; eight fixed
   // accumulators keep loads in flight while the order stays deterministic.
   double acc[8]={0,0,0,0,0,0,0,0};
   unsigned int b=threadIdx.x;
   for(;b+7*blockDim.x<RT;b+=8*blockDim.x)
     for(int u=0;u<8;u++)acc[u]+=__ldcg(partial+b+u*blockDim.x);
   for(;b<RT;b+=blockDim.x)acc[0]+=__ldcg(partial+b);
   sum=((acc[0]+acc[1])+(acc[2]+acc[3]))+((acc[4]+acc[5])+(acc[6]+acc[7]));
   for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
   if((threadIdx.x&31)==0)sums[threadIdx.x>>5]=sum;
   __syncthreads();
   if(threadIdx.x<32) {
     sum=threadIdx.x<(blockDim.x>>5)?sums[threadIdx.x]:0;
     for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
     if(threadIdx.x==0){result[0]=sum;*counter=0u;}
   }
 }
"""


def _coordinate_code(shape, lengths, norm, means=False):
    """Generate the QR / RQ* formulas used by the production coordinate kernel.

    With free mean strains (mixed loading) the six floats of the k=0 slot
    carry them: E = norm*L m, with L (6x6, root) the Cholesky factor of the
    reference compliance on the free components; the adjoint applies L^T.
    """
    n0, n1, n2 = shape
    iy = [_SOURCE_REAL.get()(1 / L) for L in lengths]
    param, strain_mean, adjoint_mean = "", "", ""
    if means:
        param = ",const float* root"
        strain_mean = f""" if(i0==0&&i1==0&&i2==0) {{
   const float m[6]={{x[0].x,x[0].y,x[1].x,x[1].y,x[2].x,x[2].y}};
   for(int c=0;c<6;c++){{float v=0;for(int j=0;j<6;j++)v+=root[6*c+j]*m[j];y[c]=make_float2({_f(norm)}*v,0.f);}}
   return;
 }}
"""
        adjoint_mean = f""" if(i0==0&&i1==0&&i2==0) {{
   float m[6];
   for(int j=0;j<6;j++){{float v=0;for(int c=0;c<6;c++)v+=root[6*c+j]*h[c].x;m[j]={_f(norm)}*v;}}
   for(int c=0;c<3;c++)y[c]=make_float2(m[2*c],m[2*c+1]);
   return;
 }}
"""
    return f"""
#define N0 {n0}
#define N1 {n1}
#define N2 {n2}
__device__ __forceinline__ bool unit_k(int i0,int i1,int i2,float k[3]) {{
 // Even-grid Nyquist indices have a zero wavenumber on their axis; only a
 // vanishing wavevector (each index 0 or Nyquist) is excluded.
 const bool z0=i0==0||(N0%2==0&&i0==N0/2),z1=i1==0||(N1%2==0&&i1==N1/2),z2=i2==0||(N2%2==0&&i2==N2/2);
 if(z0&&z1&&z2)return false;
 k[0]=(z0?0:(i0<=N0/2?i0:i0-N0))*{_f(iy[0])};k[1]=(z1?0:(i1<=N1/2?i1:i1-N1))*{_f(iy[1])};k[2]=(z2?0:i2)*{_f(iy[2])};
 const float inv=1/sqrtf(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
 for(int c=0;c<3;c++)k[c]*=inv;
 return true;
}}
__device__ __forceinline__ void strain_map(const float2 x[3],float2 y[6],int i0,int i1,int i2,
 const float transverse,const float longitudinal{param}) {{
{strain_mean} float k[3];
 if(!unit_k(i0,i1,i2,k)){{for(int c=0;c<6;c++)y[c]=make_float2(0.f,0.f);return;}}
 const float r2=(float)1.4142135623730950488;
 const float weight=(i2==0)?1:r2;
 const float scale={_f(norm)}/weight;
 for(int part=0;part<2;part++) {{
   float v[3],p[6];
   for(int c=0;c<3;c++)v[c]=part?x[c].y:x[c].x;
   const float dot=k[0]*v[0]+k[1]*v[1]+k[2]*v[2];
   for(int c=0;c<3;c++)v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;
   for(int c=0;c<3;c++)p[c]=k[c]*v[c];
   p[3]=(k[1]*v[2]+k[2]*v[1])/r2;
   p[4]=(k[0]*v[2]+k[2]*v[0])/r2;
   p[5]=(k[0]*v[1]+k[1]*v[0])/r2;
   for(int c=0;c<6;c++){{if(part)y[c].x=-scale*p[c];else y[c].y=scale*p[c];}}
 }}
}}
__device__ __forceinline__ void adjoint_map(const float2 h[6],float2 y[3],int i0,int i1,int i2,
 const float transverse,const float longitudinal{param}) {{
{adjoint_mean} float k[3];
 if(!unit_k(i0,i1,i2,k)){{for(int c=0;c<3;c++)y[c]=make_float2(0.f,0.f);return;}}
 const float r2=(float)1.4142135623730950488;
 const float weight=(i2==0)?1:r2;
 const float scale={_f(norm)}*weight;
 for(int part=0;part<2;part++) {{
   float g[6],v[3];
   for(int c=0;c<6;c++)g[c]=part?h[c].y:h[c].x;
   v[0]=k[0]*g[0]+(k[1]*g[5]+k[2]*g[4])/r2;
   v[1]=k[1]*g[1]+(k[0]*g[5]+k[2]*g[3])/r2;
   v[2]=k[2]*g[2]+(k[0]*g[4]+k[1]*g[3])/r2;
   const float dot=k[0]*v[0]+k[1]*v[1]+k[2]*v[2];
   for(int c=0;c<3;c++)v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;
   for(int c=0;c<3;c++){{if(part)y[c].x=scale*v[c];else y[c].y=-scale*v[c];}}
 }}
}}
"""


def _spectrum_index(shape, Tb, x0, k1, k2, c):
    """Offset of hybrid coefficient (x0, k1, k2, c): layout (x0, k2//Tb, k1, k2%Tb, c)."""
    n0, n1, n2 = shape
    return f"((((long long)({x0})*{stored_planes(n2) // Tb}+({k2})/{Tb})*{n1}+({k1}))*{Tb}+({k2})%{Tb})*6+({c})"


def _staged_load(count, NT, load, store, value="float2"):
    """Generate a copy stage that overlaps batches of loads and stores.

    Copy count elements, NT threads: every load of a batch is issued before
    its stores, so the global latency overlaps instead of serializing.
    """
    K = -(-count // NT)
    B = min(K, 16)
    guard = "" if count % NT == 0 else f"if(i<{count})"
    return f"""
 #pragma unroll 1
 for(int q0=0;q0<{K};q0+={B}) {{
   {value} staged[{B}];
   #pragma unroll
   for(int q=0;q<{B};q++){{const int i=tid+(q0+q)*{NT};if(q0+q<{K}){guard}staged[q]={load};}}
   #pragma unroll
   for(int q=0;q<{B};q++){{const int i=tid+(q0+q)*{NT};if(q0+q<{K}){guard}{store}=staged[q];}}
 }}"""


def _axis0_inverse(shape, T, NT, coord, Tb, device=False, means=False):
    """KA_inv: deferred x update, direction update, strain map, inverse FFT along axis 0.

    pending=1 applies x += alpha p with the previous p before it is replaced;
    pending=2 first moves the smoothing sequence y towards that x. The values
    and rounding are those of the former update kernel; only the time differs.
    """
    n0, n1, n2 = shape
    M, W = stored_planes(n2), 6 * T
    root = ",const float* root" if means else ""
    if device:
        # Scalars come from the device CG state (see _CONTROL); FP64 beta is
        # rounded to FP32 exactly as the host loop did.
        head = (
            "ka_inv_dev(float2* p,const float2* r,float2* S,const float2* tw_g,float2* x,float2* y,"
            "const float* step,const double* st,const int* ist,float transverse,float longitudinal"
            + root
            + ") {\n"
            " if(ist[0])return;\n"
            " const int pending=ist[3],update=!ist[4];\n"
            " const float beta=update?(float)(st[0]/st[1]):0.f,blend=(float)st[5];"
        )
    else:
        head = (
            "ka_inv(float2* p,const float2* r,float beta,int update,float2* S,const float2* tw_g,"
            "float2* x,float2* y,const float* step,float blend,int pending,float transverse,"
            "float longitudinal" + root + ") {"
        )
    return (
        coord
        + f"""
extern "C" __global__ void __launch_bounds__({NT}) {head}
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n0 * W};
 const int tid=threadIdx.x;
 // Launch order runs k1 fastest (adjacent in the spectrum); blk is the
 // logical tile index: vector chunk and reduction slot, as before.
 const int k1=blockIdx.x%{n1}, tile=blockIdx.x/{n1}, blk=k1*{M // T}+tile, k2=tile*{T};
 const float alpha=pending?*step:0.f;
 for(int i=tid;i<{n0};i+={NT})tw[i]=tw_g[i];
 for(int i=tid;i<{n0 * T};i+={NT}) {{
   const int k0=i/{T},t=i%{T};
   const long long o=((long long)blk*{n0 * T}+i)*3;
   float2 q[3],e[6];
   for(int c=0;c<3;c++)q[c]=p[o+c];
   if(pending) {{
     for(int c=0;c<3;c++) {{
       float2 xi=x[o+c];
       if(pending==2) {{const float2 yi=y[o+c];y[o+c]=make_float2(yi.x+blend*(xi.x-yi.x),yi.y+blend*(xi.y-yi.y));}}
       xi.x+=alpha*q[c].x;xi.y+=alpha*q[c].y;x[o+c]=xi;
       if(pending==3)y[o+c]=xi;  // residual smoothing starts from the updated x
     }}
   }}
   if(update) {{
     for(int c=0;c<3;c++) {{const float2 rc=r[o+c];q[c]=make_float2(rc.x+beta*q[c].x,rc.y+beta*q[c].y);p[o+c]=q[c];}}
   }}
   strain_map(q,e,k0,k1,k2+t,transverse,longitudinal{root and ",root"});
   for(int c=0;c<6;c++)buf[k0*{W}+t*6+c]=e[c];
 }}
 __syncthreads();
 {_fft(n0, W, NT, +1)}
 for(int i=tid;i<{n0 * W};i+={NT}) {{
   const int x0=i/{W},w=i%{W};
   S[{_spectrum_index(shape, Tb, "x0", "k1", "k2+w/6", "w%6")}]=buf[i];
 }}
}}
"""
    )


def _axis1(shape, T, NT, sign):
    """KB: in-place FFT along axis 1 of the six-component hybrid spectrum."""
    n0, n1, n2 = shape
    M, W = stored_planes(n2), 6 * T
    name = "kb_inv" if sign > 0 else "kb_fwd"
    return f"""
extern "C" __global__ void __launch_bounds__({NT}) {name}(float2* S,const float2* tw_g,int slab0,const int* done) {{
 if(*done)return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n1 * W};
 const int tid=threadIdx.x;
 float2* base=S+((long long)(slab0+blockIdx.y)*{M // T}+blockIdx.x)*{n1 * W};
 for(int i=tid;i<{n1};i+={NT})tw[i]=tw_g[i];{_staged_load(n1 * W, NT, "base[i]", "buf[i]")}
 __syncthreads();
 {_fft(n1, W, NT, sign)}
 for(int i=tid;i<{n1 * W};i+={NT})base[i]=buf[i];
}}
"""


_ISOTROPIC_KC = os.environ.get("CUFHOM_ISOTROPIC_KC", "1") != "0"


def _isotropic_sym21(table, rtol=None):
    """Check whether every sym21 row is isotropic in Mandel form.

    Test the upper triangle in row-major order against lambda (1x1) + 2 mu I.
    """
    t = np.asarray(table.get() if hasattr(table, "get") else table, dtype=np.float64)
    if rtol is None:  # an FP32 table rounds lambda+2mu and lambda, 2mu separately
        rtol = 1e-12 if getattr(table, "dtype", np.float64) == np.float64 else 4e-7
    if t.ndim != 2 or t.shape[1] < 21:
        return False
    rows = t[:, :21]
    lam, tmu = rows[:, 1], rows[:, 15]
    expected = np.zeros_like(rows)
    diag = (0, 6, 11, 15, 18, 20)
    for j, i in enumerate(diag):
        expected[:, i] = tmu + (lam if j < 3 else 0.0)
    for i in (1, 2, 7):
        expected[:, i] = lam
    scale = np.abs(rows).max()
    return bool(np.all(np.abs(rows - expected) <= rtol * scale))


def _axis2_material(shape, LPB, NT, Tb, id_type="int", tangent=False, iso=False):
    """KC: C2R, material energy and R2C along the contiguous axis.

    tangent: id 0 is the J2 matrix, with the consistent tangent of the
    nonlinear solver, a v + b u (u.v) + (K - a/3) tr(v) I, from per-voxel
    a, b and u (6 planes); u is read only where b != 0 (plastic voxels).
    """
    n0, n1, n2 = shape
    id_pair = "ushort2" if id_type == "unsigned short" else "int2"
    M = stored_planes(n2)
    lines = n0 * n1
    if n2 % 2 == 0:
        W, K2 = 6 * LPB, -(-LPB * M // NT)
    else:
        # Odd lines are transformed in pairs: one complex FFT of length n2
        # per two lines, lpb/2 pairs per block (see _axis2_material_odd).
        if LPB % 2:
            raise ValueError(
                "Odd contiguous axes need an even number of lines per KC block"
            )
        W, K2 = 3 * LPB, -(-(LPB // 2) * n2 // NT)
    # Material data (ids, and the tangent) is prefetched into registers at the
    # start, so its latency overlaps the line loads and the first FFT.
    args, declare, load, law = "", "", "", ""
    if tangent:
        args = "const float* ta,const float* tb,const float* tu,float bulk,"
        declare = f"float2 pa[{K2}],pb[{K2}],pu[{K2}][6];"
        load = f"""
     const long long voxel=(line0+l)*{n2}+2*m;
     pa[q]=*reinterpret_cast<const float2*>(ta+voxel);pb[q]=*reinterpret_cast<const float2*>(tb+voxel);
     for(int c=0;c<6;c++)pu[q][c]=(pb[q].x!=0.f||pb[q].y!=0.f)?*reinterpret_cast<const float2*>(tu+{n0 * n1 * n2}LL*c+voxel)
                                                             :make_float2(0.f,0.f);"""
        law = """
     if(id==0) {
       const float am=e?pa[q].y:pa[q].x,bm=e?pb[q].y:pb[q].x,tr=x[0]+x[1]+x[2];
       float d=0;
       for(int c=0;c<6;c++)d+=(e?pu[q][c].y:pu[q][c].x)*x[c];
       for(int c=0;c<6;c++)v[e][c]=am*x[c]+bm*(e?pu[q][c].y:pu[q][c].x)*d+(c<3?(bulk-am/3)*tr:0.f);
       float ev=0.f;
       for(int c=0;c<6;c++)ev=fmaf(x[c],v[e][c],ev);
       sum+=ev;
       continue;
     }"""
    if tangent == "pressure":
        args = "const float* ta,const float* tb,const float* tu,const float* tk,const float* td,const float* te,float flow_beta,"
        declare = f"float2 pa[{K2}],pb[{K2}],pk[{K2}],pd[{K2}],pe[{K2}],pu[{K2}][6];"
        load = f"""
     const long long voxel=(line0+l)*{n2}+2*m;
     pa[q]=*reinterpret_cast<const float2*>(ta+voxel);pb[q]=*reinterpret_cast<const float2*>(tb+voxel);
     pk[q]=*reinterpret_cast<const float2*>(tk+voxel);pd[q]=*reinterpret_cast<const float2*>(td+voxel);
     pe[q]=*reinterpret_cast<const float2*>(te+voxel);
     const bool needed=pb[q].x!=0.f||pb[q].y!=0.f||pd[q].x!=0.f||pd[q].y!=0.f||pe[q].x!=0.f||pe[q].y!=0.f;
     for(int c=0;c<6;c++)pu[q][c]=needed?*reinterpret_cast<const float2*>(tu+{n0 * n1 * n2}LL*c+voxel):make_float2(0.f,0.f);"""
        law = """
     if(id==0) {
       const float am=e?pa[q].y:pa[q].x,bm=e?pb[q].y:pb[q].x,km=e?pk[q].y:pk[q].x;
       const float dm=e?pd[q].y:pd[q].x,em=e?pe[q].y:pe[q].x,tr=x[0]+x[1]+x[2];
       float u[6],norm2=0.f,d=0.f;
       if(bm!=0.f||dm!=0.f||em!=0.f) {
         for(int c=0;c<6;c++){u[c]=(e?pu[q][c].y:pu[q][c].x)-(c<3?flow_beta/(3*sqrtf(1.5f)):0.f);norm2+=u[c]*u[c];}
         const float inv=norm2>0.f?1/sqrtf(norm2):0.f;
         for(int c=0;c<6;c++){u[c]*=inv;d+=u[c]*x[c];}
       } else {for(int c=0;c<6;c++)u[c]=0.f;}
       for(int c=0;c<6;c++)v[e][c]=am*x[c]+bm*u[c]*d+dm*u[c]*tr+(c<3?(km-am/3)*tr+em*d:0.f);
       float ev=0.f;
       for(int c=0;c<6;c++)ev=fmaf(x[c],v[e][c],ev);
       sum+=ev;
       continue;
     }"""
    # Isotropic tables (every row lambda I(x)I + 2 mu I in Mandel form, checked by the
    # pipeline): sigma = 2 mu eps + lambda tr(eps) delta, 3 FMA-equivalents per row
    # instead of the 36 of the general symmetric 6x6 product.
    matvec = (
        """
     const float lam_=a[1],tmu_=a[15],tr_=lam_*(x[0]+x[1]+x[2]);
     v[e][0]=tmu_*x[0]+tr_;v[e][1]=tmu_*x[1]+tr_;v[e][2]=tmu_*x[2]+tr_;
     v[e][3]=tmu_*x[3];v[e][4]=tmu_*x[4];v[e][5]=tmu_*x[5];"""
        if iso and not tangent
        else """
     v[e][0]=a[0]*x[0]+a[1]*x[1]+a[2]*x[2]+a[3]*x[3]+a[4]*x[4]+a[5]*x[5];
     v[e][1]=a[1]*x[0]+a[6]*x[1]+a[7]*x[2]+a[8]*x[3]+a[9]*x[4]+a[10]*x[5];
     v[e][2]=a[2]*x[0]+a[7]*x[1]+a[11]*x[2]+a[12]*x[3]+a[13]*x[4]+a[14]*x[5];
     v[e][3]=a[3]*x[0]+a[8]*x[1]+a[12]*x[2]+a[15]*x[3]+a[16]*x[4]+a[17]*x[5];
     v[e][4]=a[4]*x[0]+a[9]*x[1]+a[13]*x[2]+a[16]*x[3]+a[18]*x[4]+a[19]*x[5];
     v[e][5]=a[5]*x[0]+a[10]*x[1]+a[14]*x[2]+a[17]*x[3]+a[19]*x[4]+a[20]*x[5];"""
    )
    pair = f"""   const {id_pair} id2=pid[q];
   float u[2][6],v[2][6];
   for(int c=0;c<6;c++){{const float2 z=buf[m*{W}+l*6+c];u[0][c]=z.x;u[1][c]=z.y;}}
   for(int e=0;e<2;e++) {{
     const long long id=e?id2.y:id2.x;
     if(id<0||id>=materials) {{for(int c=0;c<6;c++)v[e][c]=nanf("");sum=nan("");continue;}}
     // Rows padded to 24 floats: six 16-byte loads instead of 21 scalar ones.
     const float4* row=reinterpret_cast<const float4*>(table+24*(long long)id);
     float a[24];
     #pragma unroll
     for(int t=0;t<6;t++){{const float4 r4=row[t];a[4*t]=r4.x;a[4*t+1]=r4.y;a[4*t+2]=r4.z;a[4*t+3]=r4.w;}}
     const float* x=u[e];{law}{matvec}
     // Each voxel's energy in float32 (v already carries float32 rounding),
     // accumulated in FP64: twelve dependent FP64 operations per pair cost
     // ~30 % of this kernel on a GPU with 1/64-rate FP64.
     float ev=0.f;
     for(int c=0;c<6;c++)ev=fmaf(x[c],v[e][c],ev);
     sum+=ev;
   }}
   for(int c=0;c<6;c++)buf[m*{W}+l*6+c]=make_float2(v[0][c],v[1][c]);
"""
    if n2 % 2:
        return _axis2_material_odd(
            shape, LPB, NT, Tb, id_type, tangent, args, law, pair
        )
    return f"""
extern "C" __global__ void __launch_bounds__({NT}) kc(float2* S,const {
        id_type
    }* ids,const float* table,
 long long materials,{args}double* partial,const float2* tw_g,const float2* twn_g,
 unsigned int* counter,double* result,long long group0,unsigned int RT,const int* done) {{
 if(*done)return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{M * W}; float2* twn=tw+{M};
 const unsigned int RB=group0+blockIdx.x;
 const int tid=threadIdx.x;
 for(int i=tid;i<{M};i+={NT}){{tw[i]=tw_g[i];twn[i]=twn_g[i];}}
 double sum=0;
 const long long line0=(long long)RB*{LPB};
 {id_pair} pid[{K2}];{declare}
 #pragma unroll
 for(int q=0;q<{K2};q++) {{
   const int i=tid+q*{NT},l=i/{M},m=i%{M};
   if(i<{LPB * M}&&line0+l<{lines}) {{
     pid[q]=*reinterpret_cast<const {id_pair}*>(ids+(line0+l)*{n2}+2*m);{load}
   }}
 }}{
        _staged_load(
            LPB * M * 6,
            NT,
            f"(line0+i/{M * 6}<{lines})?S[{_spectrum_index(shape, Tb, f'(line0+i/{M * 6})/{n1}', f'(line0+i/{M * 6})%{n1}', f'(i%{M * 6})/6', 'i%6')}]:make_float2(0.f,0.f)",
            f"buf[((i%{M * 6})/6)*{W}+(i/{M * 6})*6+i%6]",
        )
    }
 __syncthreads();
 // Hermitian C2R preprocessing. The k2 = 0 slot holds X0 + i XM, the real
 // k2 = 0 and z-Nyquist coefficients of the line (see FusedPipeline.to_fused).
 for(int i=tid;i<{W * (M // 2 + 1)};i+={NT}) {{
   const int w=i%{W},k=i/{W};
   if(k==0) {{const float2 q=buf[w];buf[w]=make_float2(q.x+q.y,q.x-q.y);}}
   else if(2*k=={M}) {{const float2 X=buf[k*{W}+w];buf[k*{
        W
    }+w]=make_float2(2*X.x,-2*X.y);}}
   else {{
     const float2 a=buf[k*{W}+w],b=buf[({M}-k)*{W}+w],t=twn[k];
     const float c=t.x,s=-t.y;
     float2 A=make_float2(a.x+b.x,a.y-b.y),D=make_float2(a.x-b.x,a.y+b.y);
     float2 B=make_float2(D.x*c-D.y*s,D.x*s+D.y*c);
     buf[k*{W}+w]=make_float2(A.x-B.y,A.y+B.x);
     A=make_float2(b.x+a.x,b.y-a.y);D=make_float2(b.x-a.x,b.y+a.y);
     B=make_float2(-D.x*c-D.y*s,D.x*s-D.y*c);
     buf[({M}-k)*{W}+w]=make_float2(A.x-B.y,A.y+B.x);
   }}
 }}
 __syncthreads();
 {_fft(M, W, NT, +1)}
 #pragma unroll
 for(int q=0;q<{K2};q++) {{
   const int i=tid+q*{NT},l=i/{M},m=i%{M};
   if(i>={LPB * M}||line0+l>={lines})continue;
{pair} }}
 __syncthreads();
 {_fft(M, W, NT, -1)}
 // R2C postprocessing: the k2 = 0 slot returns X0 + i XM.
 for(int i=tid;i<{W * (M // 2 + 1)};i+={NT}) {{
   const int w=i%{W},k=i/{W};
   if(k==0) {{const float2 z=buf[w];buf[w]=make_float2(z.x+z.y,z.x-z.y);}}
   else if(2*k=={M}) {{const float2 z=buf[k*{W}+w];buf[k*{W}+w]=make_float2(z.x,-z.y);}}
   else {{
     const float2 a=buf[k*{W}+w],b=buf[({M}-k)*{W}+w],t=twn[k];
     float2 A=make_float2(a.x+b.x,a.y-b.y),D=make_float2(a.x-b.x,a.y+b.y);
     float2 B=make_float2(D.x*t.x-D.y*t.y,D.x*t.y+D.y*t.x);
     buf[k*{W}+w]=make_float2(.5f*(A.x+B.y),.5f*(A.y-B.x));
     A=make_float2(b.x+a.x,b.y-a.y);D=make_float2(b.x-a.x,b.y+a.y);
     B=make_float2(-D.x*t.x-D.y*t.y,D.x*t.y-D.y*t.x);
     buf[({M}-k)*{W}+w]=make_float2(.5f*(A.x+B.y),.5f*(A.y-B.x));
   }}
 }}
 __syncthreads();
 for(int i=tid;i<{LPB * M * 6};i+={NT}) {{
   const int l=i/{M * 6},j=i%{M * 6};
   if(line0+l<{lines})S[{
        _spectrum_index(shape, Tb, f"(line0+l)/{n1}", f"(line0+l)%{n1}", "j/6", "j%6")
    }]=buf[(j/6)*{W}+l*6+j%6];
 }}
 {_REDUCE}
}}
"""


def _odd_pair(field, a="va", b="vb"):
    """Values of one voxel position on the two lines of an odd-axis pair."""
    return f"make_float2({field}[{a}],hb?{field}[{b}]:0.f)"


def _axis2_material_odd(shape, LPB, NT, Tb, id_type, tangent, args, law, pair):
    """KC on an odd contiguous axis: two real lines per complex FFT of length n2.

    Lines a = 2p and b = 2p+1 of a block share one transform: z = a + i b, so
    Z_k = A_k + i B_k and Z_{n2-k} = conj(A_k) + i conj(B_k) from the stored
    half spectra (k = 0..(n2-1)/2; the k = 0 coefficients are real, as on even
    grids), and A_k = (Z_k + conj Z_{n2-k})/2, B_k = (Z_k - conj Z_{n2-k})/(2i)
    after the forward transform. The material stage sees the two values of a
    complex sample as the pair (a_m, b_m), so the law code is the even one; a
    missing last line of an odd line count is zero and is never stored.
    """
    n0, n1, n2 = shape
    id_pair = "ushort2" if id_type == "unsigned short" else "int2"
    M, P, W = stored_planes(n2), LPB // 2, 3 * LPB
    lines = n0 * n1
    K2 = -(-P * n2 // NT)
    N = n0 * n1 * n2
    declare, load = "", ""
    if tangent:
        names = (
            ("pa", "pb", "pk", "pd", "pe") if tangent == "pressure" else ("pa", "pb")
        )
        fields = (
            ("ta", "tb", "tk", "td", "te") if tangent == "pressure" else ("ta", "tb")
        )
        declare = f"float2 {','.join(f'{v}[{K2}]' for v in names)},pu[{K2}][6];"
        load = "".join(f"{v}[q]={_odd_pair(t)};" for v, t in zip(names, fields))
        need = (
            "pb[q].x!=0.f||pb[q].y!=0.f||pd[q].x!=0.f||pd[q].y!=0.f||pe[q].x!=0.f||pe[q].y!=0.f"
            if tangent == "pressure"
            else "pb[q].x!=0.f||pb[q].y!=0.f"
        )
        load += (
            f"const bool needed={need};"
            f"for(int c=0;c<6;c++)pu[q][c]=needed?{_odd_pair('tu', f'{N}LL*c+va', f'{N}LL*c+vb')}"
            ":make_float2(0.f,0.f);"
        )

    def index(line, k, c):
        return _spectrum_index(shape, Tb, f"({line})/{n1}", f"({line})%{n1}", k, c)

    return f"""
extern "C" __global__ void __launch_bounds__({NT}) kc(float2* S,const {id_type}* ids,const float* table,
 long long materials,{args}double* partial,const float2* tw_g,const float2* twn_g,
 unsigned int* counter,double* result,long long group0,unsigned int RT,const int* done) {{
 if(*done)return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n2 * W};
 const unsigned int RB=group0+blockIdx.x;
 const int tid=threadIdx.x;
 for(int i=tid;i<{n2};i+={NT})tw[i]=tw_g[i];
 double sum=0;
 const long long line0=(long long)RB*{LPB};
 {id_pair} pid[{K2}];{declare}
 #pragma unroll
 for(int q=0;q<{K2};q++) {{
   const int i=tid+q*{NT},l=i/{n2},m=i%{n2};
   if(i<{P * n2}&&line0+2*l<{lines}) {{
     const long long va=(line0+2*l)*{n2}+m,vb=va+{n2};
     const bool hb=line0+2*l+1<{lines};
     pid[q].x=ids[va];pid[q].y=hb?ids[vb]:0;{load}
   }}
 }}
 for(int i=tid;i<{P * M * 6};i+={NT}) {{
   const int l=i/{M * 6},k=(i%{M * 6})/6,c=i%6;
   const long long a=line0+2*l;
   const float2 A=a<{lines}?S[{index("a", "k", "c")}]:make_float2(0.f,0.f);
   const float2 B=a+1<{lines}?S[{index("a+1", "k", "c")}]:make_float2(0.f,0.f);
   if(k==0)buf[l*6+c]=make_float2(A.x,B.x);
   else {{
     buf[k*{W}+l*6+c]=make_float2(A.x-B.y,A.y+B.x);
     buf[({n2}-k)*{W}+l*6+c]=make_float2(A.x+B.y,B.x-A.y);
   }}
 }}
 __syncthreads();
 {_fft(n2, W, NT, +1)}
 #pragma unroll
 for(int q=0;q<{K2};q++) {{
   const int i=tid+q*{NT},l=i/{n2},m=i%{n2};
   if(i>={P * n2}||line0+2*l>={lines})continue;
{pair}
 }}
 __syncthreads();
 {_fft(n2, W, NT, -1)}
 for(int i=tid;i<{P * M * 6};i+={NT}) {{
   const int l=i/{M * 6},k=(i%{M * 6})/6,c=i%6;
   const long long a=line0+2*l;
   const float2 Z=buf[k*{W}+l*6+c],Y=buf[(k?{n2}-k:0)*{W}+l*6+c];
   if(a<{lines})S[{index("a", "k", "c")}]=make_float2(.5f*(Z.x+Y.x),.5f*(Z.y-Y.y));
   if(a+1<{lines})S[{index("a+1", "k", "c")}]=make_float2(.5f*(Z.y+Y.y),.5f*(Y.x-Z.x));
 }}
 {_REDUCE}
}}
"""


# Device copy of fused_cg's scalar control (residual smoothing included), run
# by the last block of ka_fwd_dev. State layout:
#   st[0] rz (current r.r)  st[1] previous rz  st[2] estimate  st[3] weight
#   st[4] squared limit     st[5] blend of the deferred y move  st[6] p.Hp
#   st[7] last r.r          ist[0] done (1 converged, 2 invalid, 3 maxiter)
#   ist[1] iterations  ist[2] smoothing active  ist[3] deferred-update mode
#   ist[4] first direction  ist[5] maxiter  ist[6] smoothing allowed
_CONTROL = r"""
extern "C" __global__ void cg_control(double* st,int* ist) {
 // One thread; separate from ka_fwd_dev to keep its register budget.
 if(ist[0])return;
 const double nw=st[7];
 ist[1]+=1;
 if(!(isfinite(nw)&&nw>=0)){ist[0]=2;return;}
 const int active=ist[2];
 double est=active?st[2]:nw,weight=st[3];
 int mode=active?2:1;
 st[5]=weight;  // the update just done used the previous weight
 const double limit=st[4];
 if(!active){if(ist[6]&&limit<nw&&nw<=100*limit){mode=3;weight=0;ist[2]=1;}}
 else{weight=est/(est+nw);est=weight*nw;}
 st[2]=est;st[3]=weight;ist[3]=mode;
 if(est<=limit){ist[0]=1;return;}
 st[1]=st[0];st[0]=nw;ist[4]=0;
 if(ist[1]>=ist[5])ist[0]=3;
}
"""


# Ordinary mixed-CG steps only. Stop before the host would start smoothing,
# replace a residual or inspect a candidate. Invalid steps stop the remaining
# kernels immediately, even when the host polls only after four launches.
# st[8] is the running residual floor; st[9] mirrors the iteration count so
# one scalar packet suffices for the host control (no separate ist read).
_PREFIX_CONTROL = r"""
extern "C" __global__ void cg_prefix_control(double* st,int* ist,double growth) {
 if(ist[0])return;
 const double previous=st[0],nw=st[7];
 ist[1]+=1;st[9]=ist[1];
 st[1]=previous;st[0]=nw;ist[3]=1;ist[4]=0;
 if(!(isfinite(nw)&&nw>=0)){ist[0]=2;return;}
 st[8]=fmin(st[8],nw);
 if(nw<=st[4]||nw>growth*st[8]||ist[1]>=ist[5])ist[0]=1;
}
"""


_PREFIX_INITIALIZE = r"""
extern "C" __global__ void cg_prefix_initialize(double* st,int* ist,double rr,
 double boundary,double floor,int count) {
 st[0]=rr;st[1]=rr;st[2]=rr;st[3]=0;st[4]=boundary;st[5]=0;
 st[8]=floor;st[9]=0;
 ist[0]=0;ist[1]=0;ist[2]=0;ist[3]=0;ist[4]=1;ist[5]=count;ist[6]=0;
}
"""


def _axis0_forward(shape, T, NT, coord, Tb, device=False, means=False):
    """KA_fwd: forward FFT along axis 0, RQ*, residual update and |r|^2.

    Every block forms the same FP32 step from the FP64 energy; block 0 also
    stores it for the deferred x update in the next KA_inv.
    """
    n0, n1, n2 = shape
    M, W = stored_planes(n2), 6 * T
    K = -(-n0 * T // NT)
    root = ",const float* root" if means else ""
    if device:
        head = (
            "ka_fwd_dev(const float2* S,float2* r,float* step,double* partial,const float2* tw_g,"
            "unsigned int* counter,double* st,int* ist,float transverse,float longitudinal"
            + root
            + ") {\n"
            " if(ist[0])return;\n"
            " const double numerator=st[0];const double* denominator=st+6;"
        )
        reduce = _REDUCE.replace(
            "if(threadIdx.x==0){result[0]=sum;*counter=0u;}",
            "if(threadIdx.x==0){st[7]=sum;*counter=0u;}",
        )
        prefix = coord
    else:
        head = (
            "ka_fwd(const float2* S,float2* r,double numerator,const double* denominator,float* step,"
            "double* partial,const float2* tw_g,unsigned int* counter,double* result,float transverse,"
            "float longitudinal" + root + ") {"
        )
        reduce, prefix = _REDUCE, coord
    return (
        prefix
        + f"""
extern "C" __global__ void __launch_bounds__({NT}) {head}
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n0 * W};
 const int tid=threadIdx.x;
 // Launch order runs k1 fastest (adjacent in the spectrum); blk is the
 // logical tile index: vector chunk and reduction slot, as before.
 const int k1=blockIdx.x%{n1}, tile=blockIdx.x/{n1}, blk=k1*{M // T}+tile, k2=tile*{T};
 const unsigned int RB=blk,RT=gridDim.x;
 // Issue the contiguous residual loads first; they complete during the FFT.
 float2 rv[{K}][3];
 #pragma unroll
 for(int q=0;q<{K};q++) {{
   const int i=tid+q*{NT};
   if(i<{n0 * T}) for(int c=0;c<3;c++)rv[q][c]=r[((long long)blk*{n0 * T}+i)*3+c];
 }}
 for(int i=tid;i<{n0};i+={NT})tw[i]=tw_g[i];
 for(int i=tid;i<{n0 * W};i+={NT}) {{
   const int x0=i/{W},w=i%{W};
   buf[i]=S[{_spectrum_index(shape, Tb, "x0", "k1", "k2+w/6", "w%6")}];
 }}
 __syncthreads();
 {_fft(n0, W, NT, -1)}
 // A failed energy check poisons the reduced norm on every block.
 const double pap=denominator[0];
 const float quotient=(float)(numerator/pap);
 const float alpha=(isfinite(pap)&&pap>0&&isfinite(numerator)&&numerator>0&&isfinite(quotient)&&quotient>0)?quotient:nanf("");
 if(blockIdx.x==0&&tid==0)step[0]=alpha;
 double sum=isfinite(alpha)?0:nan("");
 #pragma unroll
 for(int q=0;q<{K};q++) {{
   const int i=tid+q*{NT};
   if(i>={n0 * T})continue;
   const int k0=i/{T},t=i%{T};
   const long long o=((long long)blk*{n0 * T}+i)*3;
   float2 h[6],ap[3];
   for(int c=0;c<6;c++)h[c]=buf[k0*{W}+t*6+c];
   adjoint_map(h,ap,k0,k1,k2+t,transverse,longitudinal{root and ",root"});
   for(int c=0;c<3;c++) {{
     float2 ri=rv[q][c];
     ri.x-=alpha*ap[c].x;ri.y-=alpha*ap[c].y;r[o+c]=ri;
     sum+=(double)ri.x*ri.x+(double)ri.y*ri.y;
   }}
 }}
 {reduce}
}}
"""
    )


_SETTLE = r"""
extern "C" __global__ void fused_settle(const float2* p,float2* x,float2* y,const float* step,float blend,
 int smooth,long long n) {
 const float alpha=*step;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x) {
   float2 xi=x[i];
   if(smooth){const float2 yi=y[i];y[i]=make_float2(yi.x+blend*(xi.x-yi.x),yi.y+blend*(xi.y-yi.y));}
   xi.x+=alpha*p[i].x;xi.y+=alpha*p[i].y;x[i]=xi;
 }
}
"""


def _view(shape, dtype, workspace):
    """Return a new array or a view into the uint8 workspace."""
    if workspace is None:
        return cp.empty(shape, dtype)
    if int(np.prod(shape)) * np.dtype(dtype).itemsize > workspace.nbytes:
        raise ValueError("The shared workspace is smaller than the requested field")
    return cp.ndarray(shape, dtype, memptr=workspace.data)


_STEP = r"""
extern "C" __global__ void fused_step(double numerator,const double* denominator,float* step){
 // The step of KA_fwd, for an update outside that kernel.
 const double pap=denominator[0];
 const float quotient=(float)(numerator/pap);
 step[0]=(isfinite(pap)&&pap>0&&isfinite(numerator)&&numerator>0&&isfinite(quotient)&&quotient>0)?quotient:nanf("");
}
"""


_SETTLE_DEVICE = r"""
extern "C" __global__ void fused_settle_device(const float2* p,float2* x,float2* y,const float* step,
 const double* st,int* ist,long long n) {
 const int mode=ist[3];
 const float alpha=*step,blend=(float)st[5];
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x) {
   if(!mode)break;
   float2 xi=x[i];
   if(mode==2){const float2 yi=y[i];y[i]=make_float2(yi.x+blend*(xi.x-yi.x),yi.y+blend*(xi.y-yi.y));}
   xi.x+=alpha*p[i].x;xi.y+=alpha*p[i].y;x[i]=xi;
   if(mode==3)y[i]=xi;
 }
}
"""


_COMPACT = r"""
extern "C" __global__ void compact_ids(const int* ids,unsigned short* out,long long n,int rows,int* bad) {
 // Narrow valid material ids to 16 bits; any invalid id keeps int32 storage.
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x) {
   const int id=ids[i];
   if(id<0||id>=rows)*bad=1;
   out[i]=(unsigned short)id;
 }
}
"""
# Shape-independent kernels of the fused path, compiled with the grid kernels.
_GENERIC = (
    ("settle", _SETTLE, "fused_settle"),
    ("settle_device", _SETTLE_DEVICE, "fused_settle_device"),
    ("step", _STEP, "fused_step"),
    ("control", None, "cg_control"),
    ("prefix_control", _PREFIX_CONTROL, "cg_prefix_control"),
    ("prefix_initialize", _PREFIX_INITIALIZE, "cg_prefix_initialize"),
    ("compact", _COMPACT, "compact_ids"),
)


def _generic_kernels(dtype=np.float32):
    dtype = real_dtype(dtype)
    kernels, missing = {}, []
    for name, code, entry in _GENERIC:
        key = (
            name
            if dtype == np.dtype("float32")
            or name in ("control", "prefix_control", "prefix_initialize", "compact")
            else (name, dtype.str)
        )
        if key not in _KERNELS:
            source = _CONTROL if name == "control" else code
            missing.append((key, cuda_source(source, dtype), entry))
        kernels[name] = key
    for (key, _, _), kernel in zip(
        missing, _compile_many([(code, entry, 0) for _, code, entry in missing])
    ):
        _KERNELS[key] = kernel
    return {name: _KERNELS[key] for name, key in kernels.items()}


def _compact_ids(ids, rows):
    """uint16 copy of valid ids (one small kernel instead of CuPy reductions)."""
    if rows > 65535:
        return ids, False
    out = cp.empty(ids.shape, cp.uint16)
    bad = cp.zeros(1, cp.int32)
    n = ids.size
    _KERNELS["compact"](
        (min(4096, (n + 255) // 256),),
        (256,),
        (ids, out, np.int64(n), np.int32(rows), bad),
    )
    return (ids, False) if int(bad[0]) else (out, True)


def _divisor(n, preferred):
    for t in preferred:
        if n % t == 0:
            return t
    return 1


def _compile_many(specs):
    """Compile (source, name, smem) kernels concurrently; NVRTC releases the GIL."""
    specs = list(specs)
    if len(specs) < 2:
        return [_compile(*spec) for spec in specs]
    # The first compile fills CuPy's per-process NVRTC probe cache; threads
    # started before that would each repeat the probe.
    first = _compile(*specs[0])
    with ThreadPoolExecutor(min(8, len(specs) - 1)) as pool:
        return [first] + list(pool.map(lambda spec: _compile(*spec), specs[1:]))


def _compile(source, name, smem):
    key = (source, name)
    kernel = _KERNELS.get(key)
    if kernel is None:
        # Default options: every RawKernel shares one NVRTC environment, whose
        # version probe CuPy compiles once per process and options set.
        kernel = cp.RawKernel(source, name)
        kernel.attributes  # CuPy compiles lazily: force it here (threads, background)
        _KERNELS[key] = kernel
    if smem > 48 * 1024:
        kernel.max_dynamic_shared_size_bytes = smem
    return kernel


def _kernel_specs_base(
    shape, lengths, norm, cfg, compact, means=False, tangent=False, iso=False
):
    """(source, name, shared memory) of every kernel a configuration may use."""
    n0, n1, n2 = shape
    Ta, Tb, LPB = cfg["ta"], cfg["tb"], cfg["lpb"]
    smem_a, smem_b, smem_c = (
        (n0 * 6 * Ta + n0) * 8,
        (n1 * 6 * Tb + n1) * 8,
        kc_shared_complex(n2, LPB) * 8,
    )
    coord = _coordinate_code(shape, lengths, norm, means)
    id_type = "unsigned short" if compact else "int"
    return {
        "k_ainv": (
            _axis0_inverse(shape, Ta, cfg["nta_inv"], coord, Tb, means=means),
            "ka_inv",
            smem_a,
        ),
        "k_binv": (_axis1(shape, Tb, cfg["ntb"], +1), "kb_inv", smem_b),
        "k_c": (
            _axis2_material(shape, LPB, cfg["ntc"], Tb, id_type, tangent, iso),
            "kc",
            smem_c,
        ),
        "k_bfwd": (_axis1(shape, Tb, cfg["ntb"], -1), "kb_fwd", smem_b),
        "k_afwd": (
            _axis0_forward(shape, Ta, cfg["nta_fwd"], coord, Tb, means=means),
            "ka_fwd",
            smem_a,
        ),
        "k_ainv_dev": (
            _axis0_inverse(
                shape, Ta, cfg["nta_inv"], coord, Tb, device=True, means=means
            ),
            "ka_inv_dev",
            smem_a,
        ),
        "k_afwd_dev": (
            _axis0_forward(
                shape, Ta, cfg["nta_fwd"], coord, Tb, device=True, means=means
            ),
            "ka_fwd_dev",
            smem_a,
        ),
    }


def _kernel_specs(
    shape,
    lengths,
    norm,
    cfg,
    compact,
    means=False,
    tangent=False,
    dtype=np.float32,
    iso=False,
):
    dtype = real_dtype(dtype)
    token = _SOURCE_REAL.set(dtype.type)
    try:
        specs = _kernel_specs_base(
            shape, lengths, norm, cfg, compact, means, tangent, iso
        )
    finally:
        _SOURCE_REAL.reset(token)
    scale = dtype.itemsize // 4
    return {
        name: (
            cuda_source(
                _cached_geometry_source(source)
                if dtype.itemsize == 8 and name.startswith(("k_ainv", "k_afwd"))
                else source,
                dtype,
            ),
            entry,
            smem * scale,
        )
        for name, (source, entry, smem) in specs.items()
    }


def _needed(cfg, matvec=False):
    """Kernels of the loop this configuration runs (graph or host axis-0 variants).

    matvec: the host variants also serve true-residual audits (see matvec).
    """
    axis0 = ("k_ainv_dev", "k_afwd_dev") if cfg.get("graph_iterations", 0) > 0 else ()
    if matvec or not axis0:
        axis0 += ("k_ainv", "k_afwd")
    return ("k_binv", "k_c", "k_bfwd") + axis0


# Below ~136^3 one solve cannot repay compiling a new grid's kernels (~0.3 s):
# such a solve keeps the existing path while the kernels compile in the
# background; later solves of that grid use the fused pipeline.
_COMPILE_MIN_VOXELS = 2_500_000
_READY, _BACKGROUND = set(), {}


def _compiled_key(specs, names):
    digest = hashlib.sha256()
    for name in names:
        digest.update(specs[name][0].encode())
        digest.update(specs[name][1].encode())
    device = device_properties()
    digest.update(
        f"{cp.__version__}|{device['major']}.{device['minor']}|{device['name']}".encode()
    )
    return "compiled:" + digest.hexdigest()


def _compile_and_mark(specs, names, key, dtype=np.float32):
    try:
        _compile_many([specs[name] for name in names])
        _generic_kernels(dtype)
        # CuPy's own copy kernel used by the layout conversions (per dtype).
        complex_dtype = (
            cp.complex64 if real_dtype(dtype).itemsize == 4 else cp.complex128
        )
        cp.ascontiguousarray(cp.zeros((2, 3), complex_dtype).T)
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        return
    _READY.add(key)
    save_record(cache_path(), key, dict(kernels=list(names)))


def defer_compile(Q, descriptor):
    """Check whether a small, never-compiled grid should use the existing path.

    The first such call starts compiling its fused kernels in a background thread
    (NVRTC releases the GIL); completion is recorded in the kernel cache file
    so later processes know the CuPy kernel cache already holds them.
    CUFHOM_FUSED_COMPILE=sync always compiles before solving.
    """
    shape = tuple(int(n) for n in Q.shape)
    if (
        os.environ.get("CUFHOM_FUSED_COMPILE", "background") == "sync"
        or int(np.prod(shape)) > _COMPILE_MIN_VOXELS
    ):
        return False
    dtype = descriptor.material_table.dtype
    cfg = default_config(shape, device_properties()["sharedMemPerBlockOptin"], dtype)
    compact = int(descriptor.material_table.shape[0]) <= 65535
    specs = _kernel_specs(shape, Q.P.lengths, Q.norm, cfg, compact, dtype=dtype)
    names = _needed(cfg)
    key = _compiled_key(specs, names)
    if key in _READY:
        return False
    if all(
        (specs[name][0], specs[name][1]) in _KERNELS for name in names
    ) or key in read_records(cache_path()):
        _READY.add(key)
        return False
    worker = _BACKGROUND.get(key)
    if worker is None:
        # Not a daemon: interpreter exit waits for NVRTC (a daemon compile at
        # exit races the CUDA teardown and can segfault).
        worker = threading.Thread(
            target=_compile_and_mark, args=(specs, names, key, dtype)
        )
        _BACKGROUND[key] = worker
        worker.start()
    return key not in _READY


def fused_supported(Q, descriptor):
    """Raw-CUDA contract of this pipeline; anything else keeps production."""
    return (
        supports_energy(Q, descriptor)
        and not Q.P.free
        and grid_supported(Q.shape, descriptor.material_table.dtype)
    )


def grid_supported(shape, dtype=np.float32):
    """Even last axis, small prime radices and line FFTs that fit shared memory."""
    if cp is None:
        return False
    n0, n1, n2 = (int(n) for n in shape)
    if min(n0, n1, n2) < 4:
        return False
    # The contiguous axis is transformed with length n2/2 (even) or n2 (odd).
    for n in (n0, n1, n2 // 2 if n2 % 2 == 0 else n2):
        if max(radices(n)) > _MAX_PRIME:
            return False
    try:
        limit = device_properties()["sharedMemPerBlockOptin"]
    except Exception:
        return False
    return (
        2
        * real_dtype(dtype).itemsize
        * max(6 * n0 + n0, 6 * n1 + n1, kc_shared_complex(n2, 2 if n2 % 2 else 1))
        <= limit
    )


# Spectra above this size go back to the pool during audits (memory peak);
# smaller ones stay put so captured graphs keep valid pointers.
_RELEASE_BYTES = 256 * 2**20


class FusedPipeline:
    """Owns the kernels, twiddles and the single in-place hybrid spectrum.

    j2_bulk selects the nonlinear tangent material (see _axis2_material and
    bind); free mean strains of Q (mixed loading) use Q.device_mean_root().
    """

    def __init__(self, Q, descriptor, config=None, j2_bulk=None, pressure_beta=None):
        self.dtype = real_dtype(descriptor.material_table.dtype)
        self.scalar = self.dtype.type
        self.complex_dtype = np.dtype(
            "complex64" if self.dtype.itemsize == 4 else "complex128"
        )
        self.complex_bytes = 2 * self.dtype.itemsize
        if self.dtype.itemsize == 8 and (j2_bulk is not None or Q.P.free):
            raise ValueError(
                "Fused FP64 currently supports fixed-mean linear elasticity only"
            )
        self.Q, self.shape = Q, tuple(int(n) for n in Q.shape)
        n0, n1, n2 = self.shape
        self.M = M = stored_planes(n2)
        self.count = n0 * n1 * M
        limit = device_properties()["sharedMemPerBlockOptin"]
        cfg = dict(default_config(self.shape, limit, self.dtype), **(config or {}))
        self.config = cfg
        Ta, Tb = cfg["ta"], cfg["tb"]
        if (
            M % Ta
            or M % Tb
            or (n2 % 2 and cfg["lpb"] % 2)
            or max(
                (n0 * 6 * Ta + n0) * self.complex_bytes,
                (n1 * 6 * Tb + n1) * self.complex_bytes,
                kc_shared_complex(n2, cfg["lpb"]) * self.complex_bytes,
            )
            > limit
        ):
            raise ValueError(
                f"Unsupported fused FFT configuration {cfg} for {self.shape}"
            )
        # Reference moduli are kernel arguments: a new material reuses the
        # compiled kernels (sources depend only on grid, cell and tiles).
        means, tangent = bool(Q.P.free), j2_bulk is not None
        if pressure_beta is not None:
            if j2_bulk is None:
                raise ValueError("Pressure tangent requires nonlinear material mode")
            tangent = "pressure"
        self.moduli = (self.scalar(Q.transverse), self.scalar(Q.longitudinal))
        if means:
            self.moduli += (Q.device_mean_root(),)
        LPB = cfg["lpb"]
        self.smem_a = (n0 * 6 * Ta + n0) * self.complex_bytes
        self.smem_b = (n1 * 6 * Tb + n1) * self.complex_bytes
        self.smem_c = kc_shared_complex(n2, LPB) * self.complex_bytes
        # Compact material ids halve their traffic; invalid ids keep int32 so
        # the kernel still poisons them exactly as the production path does.
        self.generic = _generic_kernels(self.dtype)
        self.ids, compact = _compact_ids(
            descriptor.val.reshape(self.shape), int(descriptor.material_table.shape[0])
        )
        # Kernels and twiddles depend only on grid, cell, tiles and id width:
        # repeated solves (new loads, materials, geometries) skip codegen.
        iso = (
            _ISOTROPIC_KC
            and not tangent
            and _isotropic_sym21(descriptor.material_table)
        )
        key = (
            self.shape,
            tuple(float(L) for L in Q.P.lengths),
            float(Q.norm),
            tuple(sorted(cfg.items())),
            compact,
            means,
            tangent,
            self.dtype.str,
            iso,
        )
        parts = _PIPELINE_PARTS.get(key)
        if parts is None:
            specs = _kernel_specs(
                self.shape,
                Q.P.lengths,
                Q.norm,
                cfg,
                compact,
                means,
                tangent,
                self.dtype,
                iso,
            )

            def twiddle(n):
                return cp.asarray(
                    np.exp(-2j * np.pi * np.arange(n) / n).astype(self.complex_dtype)
                )

            # Even KC: half-length FFT plus its packing twiddles; odd KC: length n2.
            twiddles = (
                (
                    twiddle(n0),
                    twiddle(n1),
                    twiddle(M),
                    cp.asarray(
                        np.exp(-2j * np.pi * np.arange(M) / n2).astype(
                            self.complex_dtype
                        )
                    ),
                )
                if n2 % 2 == 0
                else (twiddle(n0), twiddle(n1), twiddle(n2), twiddle(1))
            )
            parts = _PIPELINE_PARTS[key] = dict(
                specs=specs, kernels={}, twiddles=twiddles
            )
        self._parts = parts
        if self.dtype.itemsize == 8:
            inverse_norm = _geometry_for_pipeline(key, parts, self.shape, Q.P.lengths)
            self.moduli += (inverse_norm if inverse_norm is not None else np.uint64(0),)
        # Compile what the expected loop uses now (in parallel); the other
        # axis-0 variants compile on first use (CUFHOM_CUDA_GRAPHS=0, tests).
        self._ensure(_needed(cfg, matvec=tangent))
        self.tw0, self.tw1, self.tw2, self.twn = parts["twiddles"]
        # sym21 rows padded to 24 floats (16-byte aligned) for float4 loads in KC.
        table = descriptor.material_table
        self.table = cp.zeros((table.shape[0], 24), self.dtype)
        self.table[:, :21] = table
        self.j2_bulk = None if j2_bulk is None else self.scalar(j2_bulk)
        self.pressure_beta = (
            None if pressure_beta is None else self.scalar(pressure_beta)
        )
        self.tangent = None
        # Optional device memory (a uint8 array) the spectrum views instead of
        # allocating: a caller shares it with fields that alternate with it.
        self.workspace = None
        self.grid_a = (n1 * M // Ta,)
        self.grid_b = (M * 6 // (6 * Tb), n0)
        if cfg["slabs"] < n0 and (n1 * cfg["slabs"]) % LPB:
            raise ValueError(f"Slab chunks must hold whole line groups: {cfg}")
        self.grid_c = (-(-n0 * n1 // LPB),)
        self.S = None
        self.partial_c = cp.empty(self.grid_c[0], cp.float64)
        self.partial_a = cp.empty(self.grid_a[0], cp.float64)
        self.step = cp.empty((), self.dtype)
        self.counters = cp.zeros(2, cp.uint32)
        # Device CG state for graph launches (layout documented at _CONTROL);
        # the energy scalar is st[6] in both host and device modes.
        self.st = cp.zeros(10, cp.float64)
        self.ist = cp.zeros(7, cp.int32)
        self.energy = self.st[6:7].reshape(())
        self.never_done = cp.zeros(1, cp.int32)
        self.unit_energy = cp.ones(1, cp.float64)
        self.vectors = None
        self.graph = None
        self.expected_iterations = 0
        self.norm = cp.empty((), cp.float64)
        # x += step*p (and the smoothing move of y) deferred to the next KA_inv.
        self.pending = None
        # A host-resident iterate updates during KA_fwd instead: the event of
        # its last kernel, awaited before p changes.
        self.inflight = None
        self.side_step = cp.empty((), self.dtype)

    def _ensure(self, names):
        kernels, specs = self._parts["kernels"], self._parts["specs"]
        missing = [name for name in names if name not in kernels]
        for name, kernel in zip(
            missing, _compile_many([specs[name] for name in missing])
        ):
            kernels[name] = kernel
        for name in names:
            setattr(self, name, kernels[name])

    def __getattr__(self, name):
        """Compile a registered kernel lazily or raise for unknown attributes."""
        # Lazily compiled kernels (see _ensure); anything else is a real miss.
        parts = self.__dict__.get("_parts")
        if parts is not None and name in parts["specs"]:
            self._ensure((name,))
            return self.__dict__[name]
        raise AttributeError(name)

    # Exact conversions with the production (3, n0, n1, n2//2+1) vectors. On
    # even grids the fused k2 = 0 plane carries Q = z0 + i zM: the k2 = 0 and
    # z-Nyquist planes share the wavevector (k0, k1, 0) and are each Hermitian,
    # so Q holds both exactly and isometrically (no plane is dropped).
    # CG vectors are tile-major, [k1][k2 tile][k0][k2 in tile][component],
    # so each axis-0 block streams one contiguous chunk; dots are unchanged.
    # Free mean strains follow in production and fill the k=0 slot here.
    def to_fused(self, z):
        """Convert production coordinates to the tiled layout used by fused kernels."""
        if z.dtype != self.complex_dtype:
            raise ValueError(
                "Fused coordinates and material must have the same precision"
            )
        n0, n1, _ = self.shape
        T, count, free = self.config["ta"], self.Q.count, len(self.Q.P.free)
        planes = z[: 3 * count].reshape(3, n0, n1, self.shape[2] // 2 + 1)
        v = planes[..., : self.M]
        if self.shape[2] % 2 == 0:
            # Pack the k2 Nyquist plane into the k2 = 0 plane: Q = z0 + i zM.
            v = v.copy()
            v[..., 0] += 1j * planes[..., self.M]
        v = v.reshape(3, n0, n1, self.M // T, T).transpose(2, 3, 1, 4, 0)
        out = cp.ascontiguousarray(v).reshape(-1)
        if free:
            out[:3] = 0
            out[:3].view(self.dtype)[:free] = z[3 * count :].real
        return out

    def to_production(self, z):
        """Convert tiled fused coordinates to the production coordinate layout."""
        n0, n1, _ = self.shape
        T, count, free = self.config["ta"], self.Q.count, len(self.Q.P.free)
        out = cp.zeros(3 * count + free, z.dtype)
        v = z.reshape(n1, self.M // T, n0, T, 3).transpose(4, 2, 0, 1, 3)
        planes = out[: 3 * count].reshape(3, n0, n1, self.shape[2] // 2 + 1)
        planes[..., : self.M] = v.reshape(3, n0, n1, self.M)
        if self.shape[2] % 2 == 0:
            # Unpack Q = z0 + i zM: z0 and zM are the Hermitian parts in (k0, k1).
            q = planes[..., 0].copy()
            mirror = cp.roll(cp.flip(q, axis=(1, 2)), 1, axis=(1, 2)).conj()
            planes[..., 0] = (q + mirror) / 2
            planes[..., self.M] = (q - mirror) / 2j
        if free:
            out[3 * count :] = z[:3].view(self.dtype)[:free]
            out[: 3 * count : count] = 0
            if self.shape[2] % 2 == 0:
                out[self.M : 3 * count : count] = (
                    0  # the (0, 0, Nyquist) mode: zero wavevector
                )
        return out

    def release(self):
        """Return a large hybrid spectrum to the pool while audits run.

        Small spectra stay allocated: graph launches keep their pointers.
        """
        if self.S is not None and (
            self.S.nbytes > _RELEASE_BYTES or self.workspace is not None
        ):
            self.S = (
                None  # a view of a shared workspace always: the next field reuses it
            )

    def _join(self):
        if self.inflight is not None:
            cp.cuda.get_current_stream().wait_event(self.inflight)
            self.inflight = None

    def _settle_launch(self, p, step, blend):
        """Launch the settle kernel on a row range of a host-resident iterate."""
        kernel, blend = self.generic["settle"], self.scalar(blend)

        def launch(x, y, base, n):
            kernel(
                (min(4096, (n + 255) // 256),),
                (256,),
                (
                    p[base : base + n],
                    x,
                    x if y is None else y,
                    step,
                    blend,
                    np.int32(y is not None),
                    np.int64(n),
                ),
            )

        return launch

    def settle(self):
        """Apply a deferred x update before x (or y) is read outside the pipeline.

        A host-resident iterate (coordinates._HostVector) waits here only for
        its update kernels; its own methods wait for the copies back.
        """
        self._join()
        if self.pending is None:
            return
        p, x, y, blend = self.pending
        self.pending = None
        if not isinstance(x, cp.ndarray):
            x.settle(self._settle_launch(p, self.step, blend), y)
            return
        n = x.size
        self.generic["settle"](
            (min(4096, (n + 255) // 256),),
            (256,),
            (
                p,
                x,
                x if y is None else y,
                self.step,
                self.scalar(blend),
                np.int32(y is not None),
                np.int64(n),
            ),
        )

    def apply(self, p, direction=None):
        """S <- R2C(C : C2R(QR p)); returns the device FP64 energy p:Hp."""
        if p.dtype != self.complex_dtype or (
            direction is not None and direction[0].dtype != p.dtype
        ):
            raise ValueError("Fused vectors and material must have the same precision")
        if self.S is None:
            n0, n1, _ = self.shape
            self.S = _view((n0, n1, self.M, 6), self.complex_dtype, self.workspace)
        if direction is None:
            self.settle()
            update, r, beta = np.int32(0), p, 0.0
        else:
            r, beta = direction
            update = np.int32(1)
        self._join()  # KA_inv overwrites p
        if self.pending is not None and self.pending[0] is not p:
            self.settle()
        if self.pending is None:
            x, y, blend, pending = p, p, 0.0, 0
        else:
            _, x, y, blend = self.pending
            pending = 1 if y is None else 2
            y = x if y is None else y
            self.pending = None
        cfg = self.config
        self.k_ainv(
            self.grid_a,
            (cfg["nta_inv"],),
            (
                p,
                r,
                self.scalar(beta),
                update,
                self.S,
                self.tw0,
                x,
                y,
                self.step,
                self.scalar(blend),
                np.int32(pending),
            )
            + self.moduli,
            shared_mem=self.smem_a,
        )
        self._slab_passes()
        return self.energy

    def bind(
        self, a, b, u, bulk=None, coupling_to_direction=None, coupling_to_pressure=None
    ):
        """Tangent of the next solves: per-voxel a, b and direction u (6 planes).

        Graph loops copy them to buffers they own (captured pointers);
        host loops read the caller's contiguous float32 arrays directly.
        """
        n = int(np.prod(self.shape))
        values = (a, b, u)
        sizes = (n, n, 6 * n)
        if self.pressure_beta is not None:
            if any(
                v is None for v in (bulk, coupling_to_direction, coupling_to_pressure)
            ):
                raise ValueError("Pressure tangent needs bulk and both coupling fields")
            values += (bulk, coupling_to_direction, coupling_to_pressure)
            sizes += (n, n, n)
        fields = tuple(cp.ascontiguousarray(v, self.dtype).reshape(-1) for v in values)
        if tuple(v.size for v in fields) != sizes:
            raise ValueError("Tangent fields do not match the pipeline grid")
        if self.config["graph_iterations"] > 0:
            if self.tangent is None:
                self.tangent = tuple(cp.empty_like(v) for v in fields)
            for target, value in zip(self.tangent, fields):
                cp.copyto(target, value)
        else:
            self.tangent = fields

    def _material(self):
        base = (self.ids, self.table, np.int64(self.table.shape[0]))
        if self.j2_bulk is None:
            return base
        if self.tangent is None:
            raise ValueError("Bind the nonlinear tangent before applying the operator")
        return (
            base
            + self.tangent
            + (self.j2_bulk if self.pressure_beta is None else self.pressure_beta,)
        )

    def matvec(self, z, out=None):
        """H z as a new vector (or into out), for audits and nonsymmetric Krylov.

        KA_fwd with alpha = p.Hp/p.Hp = 1 on a zero residual gives -Hz exactly.
        """
        self.apply(z)
        if out is None:
            out = cp.zeros_like(z)
        else:
            out.fill(0)
        # Unit step extracts Hz directly. It is valid for nonsymmetric and
        # semidefinite tangents too, and avoids a host energy synchronization.
        self.k_afwd(
            self.grid_a,
            (self.config["nta_fwd"],),
            (
                self.S,
                out,
                np.float64(1),
                self.unit_energy,
                self.step,
                self.partial_a,
                self.tw0,
                self.counters[1:2],
                self.norm,
            )
            + self.moduli,
            shared_mem=self.smem_a,
        )
        return cp.negative(out, out=out)

    def _slab_passes(self, done=None):
        """Run the remaining transform stages in cache-sized first-axis slabs.

        KB_inv, KC, KB_fwd touch only x0 slabs: run them chunk by chunk so a
        chunk written by one kernel is still in L2 when the next one reads it.
        """
        cfg = self.config
        n0, n1, _ = self.shape
        chunk, lpb = cfg["slabs"], cfg["lpb"]
        groups = -(-n0 * n1 // lpb)
        material = self._material()
        done = self.never_done if done is None else done
        streams = self._streams(cfg.get("slab_streams", 1)) if chunk < n0 else None
        if streams:
            # Alternate chunks between streams so one chunk's tail overlaps the
            # next chunk; the energy reduction is indexed by global block.
            origin = cp.cuda.get_current_stream()
            ready = origin.record()
            for side in streams:
                side.wait_event(ready)
        for index, s0 in enumerate(range(0, n0, chunk)):
            c = min(chunk, n0 - s0)
            stream = streams[index % len(streams)] if streams else None
            self.k_binv(
                (self.grid_b[0], c),
                (cfg["ntb"],),
                (self.S, self.tw1, np.int32(s0), done),
                shared_mem=self.smem_b,
                stream=stream,
            )
            self.k_c(
                (-(-c * n1 // lpb),),
                (cfg["ntc"],),
                (self.S,)
                + material
                + (
                    self.partial_c,
                    self.tw2,
                    self.twn,
                    self.counters[0:1],
                    self.energy,
                    np.int64(s0 * n1 // lpb),
                    np.uint32(groups),
                    done,
                ),
                shared_mem=self.smem_c,
                stream=stream,
            )
            self.k_bfwd(
                (self.grid_b[0], c),
                (cfg["ntb"],),
                (self.S, self.tw1, np.int32(s0), done),
                shared_mem=self.smem_b,
                stream=stream,
            )
        if streams:
            for side in streams:
                origin.wait_event(side.record())

    def _streams(self, count):
        if count < 2:
            return None
        return _shared_streams("side", count)

    def update(self, p, numerator, denominator, x, r, smoothing=None):
        """Update r -= alpha*Hp now and x += alpha*p at the next apply/settle.

        A host-resident iterate instead updates now, its rows streaming on a
        side stream while KA_fwd runs (the step computed first, as KA_fwd does).
        """
        self.settle()
        if not isinstance(x, cp.ndarray):
            self.generic["step"](
                (1,), (1,), (np.float64(numerator), denominator, self.side_step)
            )
            y, blend = (None, 0.0) if smoothing is None else smoothing
            self.inflight = x.settle(
                self._settle_launch(p, self.side_step, blend),
                y,
                stream=_shared_streams("host_update", 1)[0],
            )
        self.k_afwd(
            self.grid_a,
            (self.config["nta_fwd"],),
            (
                self.S,
                r,
                np.float64(numerator),
                denominator,
                self.step,
                self.partial_a,
                self.tw0,
                self.counters[1:2],
                self.norm,
            )
            + self.moduli,
            shared_mem=self.smem_a,
        )
        if not isinstance(x, cp.ndarray):
            self.pending = None
        elif smoothing is None:
            self.pending = (p, x, None, 0.0)
        else:
            self.pending = (p, x, smoothing[0], float(smoothing[1]))
        return float(self.norm)

    def prefix(
        self,
        p,
        rr,
        x,
        r,
        direction,
        limit,
        floor,
        reference,
        drop,
        growth,
        smoothing,
        remaining,
    ):
        """At most four ordinary mixed-CG steps, with one host scalar packet.

        Device kernels stop at exactly the host loop's first event. No
        smoothed vector is allocated speculatively: the host starts that
        sequence on the existing boundary and continues its usual control.
        The first inverse pass consumes the host's deferred direction and
        displacement update; later passes use the existing device kernels.
        """
        boundary = max((100.0 if smoothing else 1.0) * limit, drop * drop * reference)
        count = min(4, remaining)
        self.generic["prefix_initialize"](
            (1,),
            (1,),
            (
                self.st,
                self.ist,
                np.float64(rr),
                np.float64(boundary),
                np.float64(floor),
                np.int32(count),
            ),
        )
        self.apply(p, direction)
        control = self.generic["prefix_control"]
        growth = np.float64(np.inf if growth is None else growth)
        cfg = self.config
        # apply() produced p.Hp in st[6] after the scalar reset.
        for j in range(count):
            if j:
                self.k_ainv_dev(
                    self.grid_a,
                    (cfg["nta_inv"],),
                    (p, r, self.S, self.tw0, x, x, self.step, self.st, self.ist)
                    + self.moduli,
                    shared_mem=self.smem_a,
                )
                self._slab_passes(self.ist)
            self.k_afwd_dev(
                self.grid_a,
                (cfg["nta_fwd"],),
                (
                    self.S,
                    r,
                    self.step,
                    self.partial_a,
                    self.tw0,
                    self.counters[1:2],
                    self.st,
                    self.ist,
                )
                + self.moduli,
                shared_mem=self.smem_a,
            )
            control((1,), (1,), (self.st, self.ist, growth))
        state = self.st.get()
        self.pending = (p, x, None, 0.0)
        return int(state[9]), float(state[1]), float(state[7]), float(state[8])

    # ---- CUDA-graph CG: the scalar recurrence runs on the device ----------
    def _graph_vectors(self, like):
        if self.vectors is None or self.vectors[0].shape != like.shape:
            self.vectors = tuple(cp.empty_like(like) for _ in range(4))
        return self.vectors

    def _iteration_device(self):
        x, r, p, y = self.vectors
        cfg = self.config
        self.k_ainv_dev(
            self.grid_a,
            (cfg["nta_inv"],),
            (p, r, self.S, self.tw0, x, y, self.step, self.st, self.ist) + self.moduli,
            shared_mem=self.smem_a,
        )
        self._slab_passes(self.ist)
        self.k_afwd_dev(
            self.grid_a,
            (cfg["nta_fwd"],),
            (
                self.S,
                r,
                self.step,
                self.partial_a,
                self.tw0,
                self.counters[1:2],
                self.st,
                self.ist,
            )
            + self.moduli,
            shared_mem=self.smem_a,
        )
        self.generic["control"]((1,), (1,), (self.st, self.ist))

    def _captured(self):
        if self.S is None:
            n0, n1, _ = self.shape
            self.S = cp.empty((n0, n1, self.M, 6), self.complex_dtype)
        key = (self.S.data.ptr,) + tuple(
            v.data.ptr for v in self.vectors + (self.tangent or ())
        )
        if self.graph is None or self.graph[0] != key:
            stream = self._graph_stream()
            with stream:
                stream.begin_capture()
                for _ in range(self.config.get("graph_iterations", 4)):
                    self._iteration_device()
                graph = stream.end_capture()
            self.graph = (key, graph)
        return self.graph[1]

    def _graph_stream(self):
        return _shared_streams("graph", 1)[0]

    def _reset_state(self, rr, limit, maxiter, iterations=0, smoothing=True):
        self.st.set(np.array([rr, rr, rr, 0.0, limit, 0.0, 0.0, rr, 0.0, 0.0]))
        self.ist.set(np.array([0, iterations, 0, 0, 1, maxiter, smoothing], np.int32))

    def graph_cg(
        self, rhs, dot, tolerance, maxiter, residual_action, accept=None, smoothing=True
    ):
        """fused_cg (residual_smoothing=smoothing) as CUDA graphs of several iterations.

        Each graph runs graph_iterations CG steps; kernels after convergence
        return at once. FP64 scalars (beta, alpha, smoothing weights) follow
        the host loop's operations, so iterates match it bit for bit. The true
        residual, physical acceptance and restarts stay on the host.
        """
        self.last_cg_stats = dict(restarts=0, replacements=0, candidate_checks=0)
        progress = FreshProgress()
        x, r, p, y = self._graph_vectors(rhs)
        x.fill(0)
        cp.copyto(r, rhs)
        rr = dot(r, r)
        limit = tolerance * tolerance * rr
        if rr == 0:
            return x, 0
        cp.copyto(p, r)
        self.pending = None
        self._reset_state(rr, limit, maxiter, smoothing=smoothing)
        stream = self._graph_stream()
        # Solves on one pipeline need similar iteration counts: launch the
        # graphs the previous one needed back to back before reading the state
        # (a host round trip each); iterations after convergence are no-ops.
        ahead = max(
            1, -(-self.expected_iterations // self.config.get("graph_iterations", 4))
        )
        while True:
            graph = self._captured()
            stream.wait_event(cp.cuda.get_current_stream().record())
            for _ in range(ahead):
                graph.launch(stream=stream)
            ahead = 1
            state = self.ist.get(stream=stream)
            if state[0] == 0:
                continue
            if state[0] == 2:
                raise LinearSolveError(
                    "Nonfinite fused linear residual",
                    status="nonfinite",
                    iterations=int(state[1]),
                )
            if state[0] == 3:
                raise LinearSolveError(
                    "Fused linear CG failed true residual",
                    status="maxiter",
                    iterations=int(state[1]),
                )
            scalars = self.st.get(stream=stream)
            self.last_cg_stats["recursive_norm_res"] = float(np.sqrt(scalars[7]))
            n = x.size
            self.generic["settle_device"](
                (min(4096, (n + 255) // 256),),
                (256,),
                (p, x, y, self.step, self.st, self.ist, np.int64(n)),
            )
            if state[2]:
                _finish_smoothing(x, y, scalars[3])
                cp.copyto(x, y)
            residual = rhs - residual_action(x)
            self.last_cg_stats["candidate_checks"] += 1
            new = dot(residual, residual)
            iterations = self.expected_iterations = int(state[1])
            if new <= limit:
                factor = None if accept is None else accept(x)
                if factor is None:
                    return x, iterations
                limit *= factor
            # The device marks a candidate before its iteration cap, so a
            # tolerance below the attainable accuracy would restart forever:
            # stop at the host loop's budget instead.
            if iterations >= maxiter:
                raise LinearSolveError(
                    "Fused linear CG failed true residual",
                    status="maxiter",
                    iterations=iterations,
                )
            progress.check(np.sqrt(new), iterations)
            # Reliable restart from the true residual, keeping x.
            self.last_cg_stats["restarts"] += 1
            cp.copyto(r, residual)
            cp.copyto(p, r)
            del residual
            self._reset_state(new, limit, maxiter, iterations, smoothing)


def default_config(shape, limit=99 * 1024, dtype=np.float32, per_sm=None):
    """Tile and thread choice, calibrated on an RTX 5070 Ti (bandwidth bound).

    per_sm: shared memory per multiprocessor (default: the device's).
    """
    n0, n1, n2 = shape
    M = stored_planes(n2)
    dtype = real_dtype(dtype)
    complex_bytes = 2 * dtype.itemsize
    if per_sm is None:
        try:
            per_sm = device_properties()["sharedMemPerMultiprocessor"]
        except Exception:
            per_sm = 100 * 1024

    def tile(n, preferred):
        # A wider tile only while two blocks still fit a multiprocessor (1 KiB
        # reserved per block): the axis-0/1 lines of large grids would leave
        # one, which measured 10-13 % slower per FP32 iteration at 512^3-576^3
        # and 12 % per FP64 iteration at 256^3 than a tile of 1.
        t = _divisor(M, preferred)
        while t > 1 and (
            (n * 6 * t + n) * complex_bytes > limit
            or 2 * ((n * 6 * t + n) * complex_bytes + 1024) > per_sm
        ):
            t = _divisor(M, [u for u in (4, 3, 2, 1) if u < t])
        return t

    # Best measured choices at 120^3-320^3. Below 96^3 the kernels are latency
    # bound: wider axis-0 tiles, fewer threads and four lines per KC block
    # give each block more independent work (48^3-90^3: 5-11 % per iteration).
    small = n0 < 96
    threads = 512 if n0 >= 160 else 256 if n0 >= 96 else 128 if n0 > 64 else 64
    lpb = 2 if M <= 96 and (n1 % 2 == 0) else 1
    if small and (n0 * n1) % 4 == 0:
        lpb = 4
    while lpb > 1 and kc_shared_complex(n2, lpb) * complex_bytes > limit:
        lpb //= 2
    if n2 % 2:
        # Odd KC transforms line pairs; a block holds an even number of lines.
        lpb = 4 if small and kc_shared_complex(n2, 4) * complex_bytes <= limit else 2
    if dtype.itemsize == 8:
        threads = min(threads, 256)
    # Odd n2 stores (n2+1)/2 planes, often with no factor 2-4 (189: 95 = 5*19):
    # a tile of 5 keeps the axis-0/1 blocks wide instead of falling to 1.
    extra = (5,) if n2 % 2 else ()
    config = dict(
        ta=tile(n0, (5, 6, 4, 3, 2, 1) if small else (2, 4, 3) + extra + (1,)),
        nta_inv=threads,
        nta_fwd=threads,
        tb=tile(n1, (2, 4, 3) + extra + (1,)),
        ntb=256,
        lpb=lpb,
        ntc=128 if M <= 128 else 256,
    )
    config.update(slab_plan(shape, dtype=dtype))
    group = lpb // gcd(n1, lpb)
    if n2 % 2 and config["slabs"] < n0 and config["slabs"] % group:
        # Odd KC line pairs: every chunk but the last holds whole line groups.
        config["slabs"] = min(n0, -(-config["slabs"] // group) * group)
    spectrum = 6 * complex_bytes * n0 * n1 * M
    # Graphs only repay their per-launch tail where the host loop is visible
    # (latency-bound kernels); 0 keeps the host-driven loop.
    config["graph_iterations"] = 8 if spectrum <= 64 * 2**20 else 0
    return config


_L2_SLAB_BYTES = 8 * 2**20


_SHARED_STREAMS = {}


def _shared_streams(role, count):
    """Non-blocking streams reused by every pipeline of the process.

    CUDA multiplexes streams onto a few hardware queues in creation order.
    New streams for every solve eventually share a queue with a busy stream of
    the caller (e.g. a long copy pipeline), and the solver's kernels then wait
    behind it.  A fixed set per device, role and priority keeps the queues
    distinct; the streams take the priority of the calling stream, so a solve
    launched from a high-priority stream stays ahead of concurrent work.
    """
    # CuPy < 14 exposes no stream priority; those streams keep the default.
    priority = getattr(cp.cuda.get_current_stream(), "priority", None)
    key = (cp.cuda.Device().id, role, count, priority)
    if key not in _SHARED_STREAMS:
        options = {} if priority is None else {"priority": priority}
        _SHARED_STREAMS[key] = [
            cp.cuda.Stream(non_blocking=True, **options) for _ in range(count)
        ]
    return _SHARED_STREAMS[key]


def slab_plan(shape, in_flight=_L2_SLAB_BYTES, streams=2, dtype=np.float32):
    """Chunks of x0 slabs for KB_inv/KC/KB_fwd that stay L2 resident.

    A slab holds n1*n2/2 six-component coefficients. When the whole hybrid
    spectrum already fits the L2-sized budget, chunking only adds launches.
    """
    n0, n1, n2 = shape
    slab = 12 * real_dtype(dtype).itemsize * n1 * stored_planes(n2)
    if n0 * slab <= 2 * in_flight * streams:
        return dict(slabs=n0, slab_streams=1)
    return dict(slabs=max(1, min(n0, round(in_flight / slab))), slab_streams=streams)


class FusedStep:
    """DeferredHP-compatible handle: the hybrid spectrum lives in the pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def update(self, p, alpha, x, r, smoothing, *, denominator=None):
        """Apply a fused Krylov update using its device-resident energy denominator."""
        if denominator is None:
            raise ValueError(
                "The fused pipeline requires the device energy denominator"
            )
        return self.pipeline.update(p, alpha, denominator, x, r, smoothing)
