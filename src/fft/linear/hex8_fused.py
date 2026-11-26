"""Generated split-axis FFT preconditioner fused with the HEX8 PCG vectors.

Inside the device-resident PCG of hex8_general._CudaPCG, the chain

    hex8_pcg_update -> cuFFT R2C -> hex8_pcg_prec -> cuFFT C2R -> hex8_pcg_beta -> hex8_pcg_direction

becomes five generated kernels plus the unchanged beta kernel:

    hex8_kz_fwd  x += alpha p, r -= alpha Ap, block max|r|, R2C along z (half-length
                 trick, all M+1 bins including the z-Nyquist plane)
    hex8_ky_fwd  forward FFT along y, in place
    hex8_kx      forward FFT along x, K0(k)^-1 (on the fly or stored, as hex8_pcg_prec),
                 Parseval r.z in FP64, scale 1/N, inverse FFT along x, in place
    hex8_pcg_beta (unchanged: same states, thresholds, history)
    hex8_ky_inv  inverse FFT along y, in place
    hex8_kz_inv  C2R along z, then p = z + beta p (iteration) or z (restart/resume)

The z buffer and the separate update/direction passes disappear from the
iteration, and every kernel returns at once when the PCG state is final, so
queued iterations after convergence do no FFT work. KZ and KY touch only
x-planes and run in chunks of planes that stay in the L2 cache, alternating
two streams (as fused.py's slab passes).

Spectrum layout (complex, working precision): S[x0][tile][k1][t][c] with
k2 = tile*T + t, c the displacement component. The k2 axis holds the M+1 = n2/2+1
RFFT columns padded to tiles*T; padding columns are zero and stay zero.

Contract: n2 even, prime factors of n0, n1 and n2/2 at most 13, line FFTs in
shared memory. Other grids keep the cuFFT path of _CudaPCG. FP32 and FP64
through precision.cuda_source, exactly as fused.py, whose line FFT generator,
staged loads, concurrent compilation and shared streams are reused.
"""

import os as _os
from contextlib import nullcontext

import numpy as np
from cufhom.fft.grids import is_fft_size

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CPU-only environments
    cp = None

import threading

from . import fused as _fused
from .fused import (
    _COMPILE_MIN_VOXELS,
    _compile_many,
    _compiled_key,
    _fft,
    _shared_streams,
    _staged_load,
)
from .hex8_general import (
    _CUDA_REDUCE,
    _CUDA_SYMBOL,
    _REDUCTION_THREADS,
    _CudaPCG,
    _finish,
    _reduction_grid,
    _reduction_kernel,
)
from .kernel_cache import cache_path, device_properties, read_records, save_record
from .precision import cuda_source, real_dtype

# FP32 working precision: vector updates and products in native FP32 arithmetic
# (FMA), with FP64 kept for the accumulation of inner products and the device
# scalars. On a GPU with 1/64-rate FP64 the FP64 updates of the FP32 Krylov
# vectors occupied the FP64 pipe 40-60 % of the time. FP64 builds are unchanged.
_FAST32 = _os.environ.get("CUFHOM_HEX8_FAST32", "1") != "0"
# FP32 builds with a device iterate: x += alpha p moves from KZ_fwd to KZ_inv, which
# already reads p_k before writing p_k+1 (KZ_fwd then reads only r and Ap).
_MOVE_X = _os.environ.get("CUFHOM_HEX8_MOVEX", "1") != "0"
# FP32 builds: 3 MiB chunks for the L2-resident z/y passes (once chunking applies, i.e. above the
# 8 MiB threshold) and at least 4 resident KX blocks per SM. Measured -3 to -6 % at 192^3-384^3;
# FP64 builds keep 8 MiB (3 MiB was +5 % there). _TUNED = False restores the production config.
_TUNED = _os.environ.get("CUFHOM_HEX8_TUNED", "1") != "0"
_FP32_CHUNK = 3 << 20
# Diagnostic: round every stored spectrum value to BF16 (precision study of 16-bit storage).
_SPEC_EMU = _os.environ.get("CUFHOM_HEX8_SPEC_EMU", "")
_EMU_SRC = """
__device__ __forceinline__ float bf16r(float v){unsigned int u=__float_as_uint(v);u+=0x7fffu+((u>>16)&1u);return __uint_as_float(u&0xffff0000u);}
__device__ __forceinline__ float2 R16(float2 v){return make_float2(bf16r(v.x),bf16r(v.y));}
"""
_ID_SRC = "\n#define R16(v) (v)\n"
_MAG32 = """
__device__ __forceinline__ float magnitude32(float v){v=fabsf(v);return v==v?v:__int_as_float(0x7f800000);}
"""
# In-flight bytes of one chunk of x-planes for the L2-local passes (as fused.py).
_L2_CHUNK = 8 << 20


# ---------------------------------------------------------------- configuration
def _smem(shape, T, lpb, dtype):
    """Dynamic shared memory (bytes) of the z, y and x kernels."""
    n0, n1, n2 = shape
    M, cb = n2 // 2, 2 * real_dtype(dtype).itemsize
    return dict(
        z=((M + 1) * 3 * lpb + 2 * M) * cb,
        y=(n1 * 3 * T + n1) * cb,
        x=(n0 * 3 * T + n0) * cb,
    )


def fused_hex8_supported(shape, dtype=np.float32, limit=99 * 1024):
    """Even n2, small prime radices and line FFTs that fit shared memory."""
    n0, n1, n2 = (int(n) for n in shape)
    if n2 % 2 or min(n0, n1, n2) < 4:
        return False
    if not all(is_fft_size(n) for n in (n0, n1, n2 // 2)):
        return False
    return max(_smem((n0, n1, n2), 1, 1, dtype).values()) <= limit


def fused_hex8_config(shape, dtype=np.float32, limit=99 * 1024):
    """Tile T of the k2 columns, lines per z block, threads and slab chunks.

    T: the largest of 4, 3, 2 whose padding of the M+1 columns stays within 5 %
    (else the smallest padding); W = 3T interleaved transforms per y/x block.
    Values follow fused.py.
    """
    n0, n1, n2 = (int(n) for n in shape)
    M, H = n2 // 2, n2 // 2 + 1

    def fits(T, lpb):
        return max(_smem(shape, T, lpb, dtype).values()) <= limit

    candidates = [T for T in (4, 3, 2, 1) if fits(T, 1)]
    T = next((t for t in candidates if -(-H // t) * t - H <= 0.05 * H), None)
    if T is None:
        T = min(candidates, key=lambda t: (-(-H // t) * t, -t))
    lpb = next(lines for lines in (4, 2, 1) if n1 % lines == 0 and fits(T, lines))
    tiles = -(-H // T)
    plane = 3 * n1 * tiles * T * 2 * real_dtype(dtype).itemsize
    tuned = _TUNED and _FAST32 and real_dtype(dtype) == np.float32
    chunk = _FP32_CHUNK if tuned else _L2_CHUNK
    if n0 * plane <= 4 * _L2_CHUNK:
        slabs, streams = n0, 1
    else:
        slabs, streams = max(1, min(n0, round(chunk / plane))), 2
    cfg = dict(
        T=T,
        tiles=tiles,
        lpb=lpb,
        ntz=128 if M <= 64 else 256,
        nty=256,
        ntx=256,
        slabs=slabs,
        slab_streams=streams,
    )
    if tuned:
        cfg["xmin"] = 4
    return cfg


# ------------------------------------------------------------- kernel sources
def _kz_forward(shape, cfg, fast32=False):
    """KZ_fwd: fused residual update (as hex8_pcg_update) and R2C along z."""
    n0, n1, n2 = shape
    M = n2 // 2
    T, tiles, lpb, NT = cfg["T"], cfg["tiles"], cfg["lpb"], cfg["ntz"]
    W, groups, n = 3 * lpb, n1 // lpb, n0 * n1 * n2
    store_open, store_close = ("R16(", ")") if cfg.get("emu") else ("", "")
    if fast32:
        update_loop = f""" const float af=update?(float)s[2]:0.f;
 float top32=0.f;
 for(int i=tid;i<{3 * lpb * M};i+={NT}){{
  const int m=i%{M},l=(i/{M})%{lpb},c=i/{lpb * M};
  const long long o={n}LL*c+((long long)x0*{n1}+y0+l)*{n2}+2*m;
  float2 v=*reinterpret_cast<const float2*>(r+o);
  if(update){{
   const float2 q=*reinterpret_cast<const float2*>(ap+o);
   if(update==1){{
    const float2 dp=*reinterpret_cast<const float2*>(p+o);
    float2 u=*reinterpret_cast<const float2*>(x+o);
    u.x=fmaf(af,dp.x,u.x);u.y=fmaf(af,dp.y,u.y);
    *reinterpret_cast<float2*>(x+o)=u;
   }}
   v.x=fmaf(-af,q.x,v.x);v.y=fmaf(-af,q.y,v.y);
   *reinterpret_cast<float2*>(r+o)=v;
   top32=fmaxf(top32,fmaxf(magnitude32(v.x),magnitude32(v.y)));
  }}
  buf[m*{W}+l*3+c]=v;
 }}"""
        top_reduce = "double top=block_max((double)top32);"
    else:
        update_loop = f""" const double alpha=update?s[2]:0.;
 double top=0;
 for(int i=tid;i<{3 * lpb * M};i+={NT}){{
  const int m=i%{M},l=(i/{M})%{lpb},c=i/{lpb * M};
  const long long o={n}LL*c+((long long)x0*{n1}+y0+l)*{n2}+2*m;
  float2 v=*reinterpret_cast<const float2*>(r+o);
  if(update){{
   const float2 q=*reinterpret_cast<const float2*>(ap+o);
   if(update==1){{
    const float2 dp=*reinterpret_cast<const float2*>(p+o);
    float2 u=*reinterpret_cast<const float2*>(x+o);
    u.x=(float)((double)u.x+alpha*(double)dp.x);u.y=(float)((double)u.y+alpha*(double)dp.y);
    *reinterpret_cast<float2*>(x+o)=u;
   }}
   v.x=(float)((double)v.x-alpha*(double)q.x);v.y=(float)((double)v.y-alpha*(double)q.y);
   *reinterpret_cast<float2*>(r+o)=v;
   top=fmax(top,fmax(magnitude(v.x),magnitude(v.y)));
  }}
  buf[m*{W}+l*3+c]=v;
 }}"""
        top_reduce = "top=block_max(top);"
    return f"""
// One block: lines (c, x0, y0..y0+{lpb}-1), all three components, W={W}
// interleaved half-length transforms. update=1 applies, with the arithmetic of
// hex8_pcg_update, x += alpha p and r -= alpha Ap and writes the block max|r|;
// update=2 leaves x to a separate (host-resident) update; update=0 (fresh
// residual, measurement) only reads r.
extern "C" __global__ void __launch_bounds__({NT}) hex8_kz_fwd(float* __restrict__ r,
 const float* __restrict__ ap,const float* __restrict__ p,float* __restrict__ x,float2* __restrict__ S,
 const float2* __restrict__ tw_g,const float2* __restrict__ twn_g,const double* __restrict__ s,
 const int* __restrict__ state,double* __restrict__ partial,int group0,int update){{
 if(state[0])return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{(M + 1) * W}; float2* twn=tw+{M};
 const int tid=threadIdx.x,g=group0+blockIdx.x,x0=g/{groups},y0=(g%{groups})*{lpb};
 for(int i=tid;i<{M};i+={NT}){{tw[i]=tw_g[i];twn[i]=twn_g[i];}}
{update_loop}
 __syncthreads();
{_fft(M, W, NT, -1)}
 // R2C post-processing of z = x_even + i x_odd; X[0] and X[M] (Nyquist) are real.
 for(int i=tid;i<{W * (M // 2 + 1)};i+={NT}){{
  const int w=i%{W},k=i/{W};
  if(k==0){{const float2 z=buf[w];buf[w]=make_float2(z.x+z.y,0.f);buf[{M * W}+w]=make_float2(z.x-z.y,0.f);}}
  else if(2*k=={M}){{const float2 z=buf[k*{W}+w];buf[k*{W}+w]=make_float2(z.x,-z.y);}}
  else{{
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
 // Columns k2 = 0..{M} into S[x0][tile][y][t][c]; padding columns are never written.
 for(int i=tid;i<{tiles * lpb * 3 * T};i+={NT}){{
  const int c=i%3,t=(i/3)%{T},l=(i/{3 * T})%{lpb},tile=i/{3 * T * lpb},k=tile*{T}+t;
  if(k<={M})S[(((long long)x0*{tiles}+tile)*{n1}+y0+l)*{3 * T}+t*3+c]={store_open}buf[k*{W}+l*3+c]{store_close};
 }}
 {top_reduce}if(threadIdx.x==0)partial[g]=top;
}}
""".replace("{update_loop}", update_loop).replace("{top_reduce}", top_reduce)


def _ky(shape, cfg, sign):
    """KY: in-place FFT along y of one (x0, tile) block, W = 3T transforms."""
    n0, n1, n2 = shape
    T, tiles, NT = cfg["T"], cfg["tiles"], cfg["nty"]
    W = 3 * T
    name = "hex8_ky_inv" if sign > 0 else "hex8_ky_fwd"
    bounds = f",{cfg['ymin']}" if cfg.get("ymin") else ""
    store_open, store_close = ("R16(", ")") if cfg.get("emu") else ("", "")
    return f"""
extern "C" __global__ void __launch_bounds__({NT}{bounds}) {name}(float2* __restrict__ S,const float2* __restrict__ tw_g,
 int slab0,const int* __restrict__ state){{
 if(state[0])return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n1 * W};
 const int tid=threadIdx.x;
 float2* base=S+((long long)(slab0+blockIdx.y)*{tiles}+blockIdx.x)*{n1 * W};
 for(int i=tid;i<{n1};i+={NT})tw[i]=tw_g[i];{_staged_load(n1 * W, NT, "base[i]", "buf[i]")}
 __syncthreads();
{_fft(n1, W, NT, sign)}
 for(int i=tid;i<{n1 * W};i+={NT})base[i]={store_open}buf[i]{store_close};
}}
"""


def _kx(shape, cfg, fast32=False):
    """KX: FFT along x, reference inverse, Parseval r.z, inverse FFT along x."""
    n0, n1, n2 = shape
    M, H = n2 // 2, n2 // 2 + 1
    T, tiles, NT = cfg["T"], cfg["tiles"], cfg["ntx"]
    W, nf = 3 * T, n0 * n1 * H
    bounds = f",{cfg['xmin']}" if cfg.get("xmin") else ""
    store_open, store_close = ("R16(", ")") if cfg.get("emu") else ("", "")
    index = f"(((long long)(i/{W})*{tiles}+tile)*{n1}+k1)*{W}+i%{W}"
    if fast32:
        parseval = (
            f"  acc+=(double)((k2==0||k2=={M}?1.f:2.f)*(u0.x*v0.x+u0.y*v0.y+u1.x*v1.x"
            f"+u1.y*v1.y+u2.x*v2.x+u2.y*v2.y));"
        )
    else:
        parseval = (
            f"  acc+=(k2==0||k2=={M}?1.:2.)*((double)u0.x*v0.x+(double)u0.y*v0.y+(double)u1.x*v1.x\n"
            f"       +(double)u1.y*v1.y+(double)u2.x*v2.x+(double)u2.y*v2.y);"
        )
    return f"""
// One block: column (k1, tile) for all x0. Same multiply and Parseval sum as
// hex8_pcg_prec (FP64 products of working-precision values, RFFT weights 1 at
// k2=0 and k2=n2/2, 2 elsewhere); padding columns are left untouched (zero).
// inverse=0 (measurement) skips the inverse transform and the write-back.
extern "C" __global__ void __launch_bounds__({NT}{bounds}) hex8_kx(float2* __restrict__ S,const float2* __restrict__ tw_g,
 const float* __restrict__ axes,int ox,int oy,int oz,const float* __restrict__ Msym,const float* __restrict__ table,
 double* __restrict__ partial,const int* __restrict__ state,float scale,double weight,int inverse){{
 if(state[0])return;
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n0 * W};
 const int tid=threadIdx.x,k1=blockIdx.x%{n1},tile=blockIdx.x/{n1};
 for(int i=tid;i<{n0};i+={NT})tw[i]=tw_g[i];{_staged_load(n0 * W, NT, f"S[{index}]", "buf[i]")}
 __syncthreads();
{_fft(n0, W, NT, -1)}
 double acc=0;
 for(int i=tid;i<{n0 * T};i+={NT}){{
  const int k0=i/{T},t=i%{T},k2=tile*{T}+t;
  if(k2>{M})continue;
  float2* q=buf+k0*{W}+t*3;
  const float2 u0=q[0],u1=q[1],u2=q[2];
  const long long f=((long long)k0*{n1}+k1)*{H}+k2;
  float w[6];
  if(table){{for(int j=0;j<6;j++)w[j]=table[j*{nf}LL+f];}}
  else reference_inverse<float>(f,{n1},{H},axes,ox,oy,oz,Msym,w);
  const float2 v0=make_float2(w[0]*u0.x+w[3]*u1.x+w[4]*u2.x,w[0]*u0.y+w[3]*u1.y+w[4]*u2.y);
  const float2 v1=make_float2(w[3]*u0.x+w[1]*u1.x+w[5]*u2.x,w[3]*u0.y+w[1]*u1.y+w[5]*u2.y);
  const float2 v2=make_float2(w[4]*u0.x+w[5]*u1.x+w[2]*u2.x,w[4]*u0.y+w[5]*u1.y+w[2]*u2.y);
{parseval}
  q[0]=make_float2(v0.x*scale,v0.y*scale);q[1]=make_float2(v1.x*scale,v1.y*scale);
  q[2]=make_float2(v2.x*scale,v2.y*scale);
 }}
 __syncthreads();
 if(inverse){{
{_fft(n0, W, NT, +1)}
  for(int i=tid;i<{n0 * W};i+={NT})S[{index}]={store_open}buf[i]{store_close};
 }}
 acc=block_sum(acc);if(threadIdx.x==0)partial[blockIdx.x]=acc*weight;
}}
""".replace("{parseval}", parseval)


def _kz_inverse(shape, cfg, fast32=False, move_x=False):
    """KZ_inv: C2R along z, then the direction update (or z itself)."""
    n0, n1, n2 = shape
    M = n2 // 2
    T, tiles, lpb, NT = cfg["T"], cfg["tiles"], cfg["lpb"], cfg["ntz"]
    W, groups, n = 3 * lpb, n1 // lpb, n0 * n1 * n2
    if fast32:
        beta_decl = " const float bf=direction?(float)s[3]:0.f;"
        direction_update = "   q.x=fmaf(bf,q.x,v.x);q.y=fmaf(bf,q.y,v.y);"
        if move_x:
            beta_decl = " const float bf=direction?(float)s[3]:0.f,af=direction?(float)s[2]:0.f;"
            direction_update = (
                "   if(x){float2 u=*reinterpret_cast<const float2*>(x+o);u.x=fmaf(af,q.x,u.x);u.y=fmaf(af,q.y,u.y);"
                "*reinterpret_cast<float2*>(x+o)=u;}\n"
                "   q.x=fmaf(bf,q.x,v.x);q.y=fmaf(bf,q.y,v.y);"
            )
    else:
        beta_decl = " const double beta=direction?s[3]:0.;"
        direction_update = "   q.x=(float)((double)v.x+beta*(double)q.x);q.y=(float)((double)v.y+beta*(double)q.y);"
    head = " const int tid=threadIdx.x,g=group0+blockIdx.x,x0=g/{groups},y0=(g%{groups})*{lpb};".format(
        groups=groups, lpb=lpb
    )
    if fast32 and move_x:
        xargs = ",float* __restrict__ x,int iteration"
        # A final state set by beta in this iteration (state[1]) still owes the iterate x += alpha p_k,
        # which KZ_fwd no longer applies; an earlier final state (or alpha's) owes nothing.
        entry = (
            (
                head
                + f"""
 if(state[0]){{{{
  if(!(direction&&x&&state[1]==iteration))return;
  const float af=(float)s[2];
  for(int i=tid;i<{3 * lpb * M};i+={NT}){{{{
   const int m=i%{M},l=(i/{M})%{lpb},c=i/{lpb * M};
   const long long o={n}LL*c+((long long)x0*{n1}+y0+l)*{n2}+2*m;
   const float2 q=*reinterpret_cast<const float2*>(out+o);
   float2 u=*reinterpret_cast<const float2*>(x+o);u.x=fmaf(af,q.x,u.x);u.y=fmaf(af,q.y,u.y);
   *reinterpret_cast<float2*>(x+o)=u;
  }}}}
  return;
 }}}}"""
            )
            .replace("{{", "{")
            .replace("}}", "}")
        )
        tidline = ""
    else:
        xargs, entry, tidline = "", " if(state[0])return;", head
    return f"""
// direction=1: out is p, p = z + beta p with the arithmetic of hex8_pcg_direction;
// direction=0: out receives z (restart/resume, measurement), like cuFFT C2R.
extern "C" __global__ void __launch_bounds__({NT}) hex8_kz_inv(const float2* __restrict__ S,float* __restrict__ out,
 const float2* __restrict__ tw_g,const float2* __restrict__ twn_g,const double* __restrict__ s,
 const int* __restrict__ state,int group0,int direction{xargs}){{
{entry}
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{(M + 1) * W}; float2* twn=tw+{M};
{tidline}
 for(int i=tid;i<{M};i+={NT}){{tw[i]=tw_g[i];twn[i]=twn_g[i];}}
 for(int i=tid;i<{tiles * lpb * 3 * T};i+={NT}){{
  const int c=i%3,t=(i/3)%{T},l=(i/{3 * T})%{lpb},tile=i/{3 * T * lpb},k=tile*{T}+t;
  if(k<={M})buf[k*{W}+l*3+c]=S[(((long long)x0*{tiles}+tile)*{n1}+y0+l)*{3 * T}+t*3+c];
 }}
 __syncthreads();
 // Hermitian C2R pre-processing to Z = 2(E + i O); imaginary parts of the
 // real bins X[0] and X[M] are ignored, as an unnormalized C2R does.
 for(int i=tid;i<{W * (M // 2 + 1)};i+={NT}){{
  const int w=i%{W},k=i/{W};
  if(k==0){{const float a=buf[w].x,b=buf[{M * W}+w].x;buf[w]=make_float2(a+b,a-b);}}
  else if(2*k=={M}){{const float2 X=buf[k*{W}+w];buf[k*{W}+w]=make_float2(2*X.x,-2*X.y);}}
  else{{
   const float2 a=buf[k*{W}+w],b=buf[({M}-k)*{W}+w],t=twn[k];
   const float cs=t.x,sn=-t.y;
   float2 A=make_float2(a.x+b.x,a.y-b.y),D=make_float2(a.x-b.x,a.y+b.y);
   float2 B=make_float2(D.x*cs-D.y*sn,D.x*sn+D.y*cs);
   buf[k*{W}+w]=make_float2(A.x-B.y,A.y+B.x);
   A=make_float2(b.x+a.x,b.y-a.y);D=make_float2(b.x-a.x,b.y+a.y);
   B=make_float2(-D.x*cs-D.y*sn,D.x*sn-D.y*cs);
   buf[({M}-k)*{W}+w]=make_float2(A.x-B.y,A.y+B.x);
  }}
 }}
 __syncthreads();
{_fft(M, W, NT, +1)}
{beta_decl}
 for(int i=tid;i<{3 * lpb * M};i+={NT}){{
  const int m=i%{M},l=(i/{M})%{lpb},c=i/{lpb * M};
  const long long o={n}LL*c+((long long)x0*{n1}+y0+l)*{n2}+2*m;
  const float2 v=buf[m*{W}+l*3+c];
  if(direction){{
   float2 q=*reinterpret_cast<const float2*>(out+o);
{direction_update}
   *reinterpret_cast<float2*>(out+o)=q;
  }}
  else *reinterpret_cast<float2*>(out+o)=v;
 }}
}}
""".replace("{beta_decl}", beta_decl).replace("{direction_update}", direction_update)


def fused_hex8_sources(shape, cfg, dtype):
    """{key: (CUDA source, entry, dynamic shared memory)} in the working precision."""
    dtype = real_dtype(dtype)
    shape = tuple(int(n) for n in shape)
    smem = _smem(shape, cfg["T"], cfg["lpb"], dtype)
    token = _fused._SOURCE_REAL.set(
        dtype.type
    )  # FP64 literals and exact eighth roots in _dft
    try:
        symbol = _CUDA_SYMBOL.replace("REAL", "float")
        fast = _FAST32 and dtype == np.float32
        emu = fast and _SPEC_EMU == "bf16"
        gcfg = dict(cfg, emu=emu)
        pre = _EMU_SRC if emu else ""
        specs = {
            "zf": (
                pre
                + _CUDA_REDUCE
                + (_MAG32 if fast else "")
                + _kz_forward(shape, gcfg, fast),
                "hex8_kz_fwd",
                smem["z"],
            ),
            "yf": (pre + _ky(shape, gcfg, -1), "hex8_ky_fwd", smem["y"]),
            "x": (
                pre + _CUDA_REDUCE + symbol + _kx(shape, gcfg, fast),
                "hex8_kx",
                smem["x"],
            ),
            "yi": (pre + _ky(shape, gcfg, +1), "hex8_ky_inv", smem["y"]),
            "zi": (
                _kz_inverse(shape, cfg, fast, fast and _MOVE_X),
                "hex8_kz_inv",
                smem["z"],
            ),
        }
    finally:
        _fused._SOURCE_REAL.reset(token)
    return {
        key: (cuda_source(source, dtype), entry, size)
        for key, (source, entry, size) in specs.items()
    }


# ------------------------------------------------------------- preconditioner
class FusedHex8Preconditioner:
    """Generated-FFT K0^-1 of a CUDA Hex8Operator in its working precision.

    forward()  : [r -= alpha Ap, x += alpha p, max|r|] and the forward transform
    multiply() : x transform, K0^-1, Parseval r.z (block partials in self.rho)
    backward() : inverse transform into z, or the direction update of p
    Every kernel returns at once when state[0] != 0 (pass self.idle to force).
    """

    def __init__(self, op, config=None):
        if cp is None or op.xp is not cp:
            raise ValueError("The fused HEX8 preconditioner requires a CUDA operator")
        self.op, self.dtype, self.shape = (
            op,
            np.dtype(op.working),
            tuple(int(n) for n in op.shape),
        )
        limit = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "sharedMemPerBlockOptin"
        ]
        if not fused_hex8_supported(self.shape, self.dtype, limit):
            raise ValueError(
                f"Grid {self.shape} is outside the fused HEX8 FFT contract"
            )
        self.config = dict(
            fused_hex8_config(self.shape, self.dtype, limit), **(config or {})
        )
        cfg = self.config
        n0, n1, n2 = self.shape
        self.M, self.H = n2 // 2, n2 // 2 + 1
        self.tiles, self.T = cfg["tiles"], cfg["T"]
        if self.tiles * self.T < self.H or n1 % cfg["lpb"]:
            raise ValueError(f"Invalid fused HEX8 configuration {cfg}")
        self.groups = n1 // cfg["lpb"]
        # KZ_inv applies x += alpha p (FP32 builds; see _MOVE_X); the source decides.
        self.move_x = _FAST32 and _MOVE_X and self.dtype == np.float32
        specs = fused_hex8_sources(self.shape, cfg, self.dtype)
        keys = list(specs)
        for key, kernel in zip(keys, _compile_many([specs[key] for key in keys])):
            setattr(self, "k_" + key, kernel)
        self.smem = {key: specs[key][2] for key in keys}
        complex_ = np.complex64 if self.dtype == np.float32 else np.complex128

        def twiddle(n):
            return cp.asarray(np.exp(-2j * np.pi * np.arange(n) / n).astype(complex_))

        self.tw0, self.tw1, self.twM = twiddle(n0), twiddle(n1), twiddle(self.M)
        self.twn = cp.asarray(
            np.exp(-2j * np.pi * np.arange(self.M) / n2).astype(complex_)
        )
        # Padding columns must be zero once; no kernel writes nonzero values there.
        self.S = cp.zeros(3 * n0 * n1 * self.tiles * self.T, complex_)
        self.rmax = cp.zeros(n0 * self.groups, cp.float64)
        self.rho = cp.zeros(n1 * self.tiles, cp.float64)
        self.idle = cp.zeros(2, cp.int32)
        self.unit = cp.zeros(
            8, cp.float64
        )  # placeholder scalars when nothing is updated
        self.scale, self.weight = self.dtype.type(1 / op.n), np.float64(1 / op.n)

    def _slabs(self, launch):
        """launch(stream, first plane, planes) per L2-sized chunk, two streams."""
        n0, cfg = self.shape[0], self.config
        chunk = cfg["slabs"]
        streams = (
            _shared_streams("hex8_fused", cfg["slab_streams"])
            if chunk < n0 and cfg["slab_streams"] > 1
            else None
        )
        if streams:
            origin = cp.cuda.get_current_stream()
            ready = origin.record()
            for side in streams:
                side.wait_event(ready)
        for index, x0 in enumerate(range(0, n0, chunk)):
            launch(
                streams[index % len(streams)] if streams else None,
                x0,
                min(chunk, n0 - x0),
            )
        if streams:
            for side in streams:
                origin.wait_event(side.record())

    def forward(self, r, state, *, update=False, ap=None, p=None, x=None, s=None):
        """Run the fused update and forward transforms in cache-sized slabs.

        KZ_fwd (+ fused update when update; x None leaves x to a separate
        update) and KY_fwd, chunk by chunk.
        """
        cfg = self.config
        mode = 0 if not update else 1 if x is not None else 2
        args = (ap, p, p if x is None else x, s) if update else (r, r, r, self.unit)

        def launch(stream, x0, count):
            self._run(
                "zf",
                (count * self.groups,),
                (cfg["ntz"],),
                (
                    r,
                    *args[:3],
                    self.S,
                    self.twM,
                    self.twn,
                    args[3],
                    state,
                    self.rmax,
                    np.int32(x0 * self.groups),
                    np.int32(mode),
                ),
                stream,
            )
            self._run(
                "yf",
                (self.tiles, count),
                (cfg["nty"],),
                (self.S, self.tw1, np.int32(x0), state),
                stream,
            )

        self._slabs(launch)

    def multiply(self, state, inverse=True):
        """KX over all (k1, tile) columns; rho partials in self.rho."""
        self._run(
            "x",
            (self.shape[1] * self.tiles,),
            (self.config["ntx"],),
            (
                self.S,
                self.tw0,
                *self.op.symbol,
                self.rho,
                state,
                self.scale,
                self.weight,
                np.int32(inverse),
            ),
        )

    def backward(self, out, state, *, s=None, direction=False, x=None, iteration=0):
        """Run inverse transforms and the optional Krylov direction update.

        KY_inv and KZ_inv, chunk by chunk: z into out, or p = z + beta p
        (with x += alpha p_k first when move_x and x is given).
        """
        cfg = self.config
        scalars = s if direction else self.unit
        extra = (
            (x if x is not None else np.uint64(0), np.int32(iteration))
            if self.move_x
            else ()
        )

        def launch(stream, x0, count):
            self._run(
                "yi",
                (self.tiles, count),
                (cfg["nty"],),
                (self.S, self.tw1, np.int32(x0), state),
                stream,
            )
            self._run(
                "zi",
                (count * self.groups,),
                (cfg["ntz"],),
                (
                    self.S,
                    out,
                    self.twM,
                    self.twn,
                    scalars,
                    state,
                    np.int32(x0 * self.groups),
                    np.int32(direction),
                    *extra,
                ),
                stream,
            )

        self._slabs(launch)

    def apply(self, r, z):
        """Standalone z = K0^-1 r; returns r.z (Parseval, FP64)."""
        self.forward(r, self.idle)
        self.multiply(self.idle)
        self.backward(z, self.idle)
        return self._sum(self.rho)

    def measure(self, r):
        """r.K0^-1 r without the inverse transform (energy_norm)."""
        self.forward(r, self.idle)
        self.multiply(self.idle, inverse=False)
        return self._sum(self.rho)

    def _run(self, key, grid, block, args, stream=None):
        with stream if stream is not None else nullcontext():
            getattr(self, "k_" + key)(grid, block, args, shared_mem=self.smem[key])

    def _sum(self, partial):
        """Fixed-order FP64 sum of block partials (hex8_general._finish)."""
        return float(_finish(partial, 1)[0])


class FusedCudaPCG(_CudaPCG):
    """_CudaPCG with the preconditioner chain of FusedHex8Preconditioner.

    The action, alpha, beta, direction (restart/resume), accumulate kernels,
    device scalars, history and the host lookahead are those of _CudaPCG; only
    update/R2C/prec/C2R/direction of an iteration and the preconditioning of a
    fresh residual are replaced. Drop-in for operator._pcg in _solve_cuda.
    """

    def __init__(self, op, config=None):
        super().__init__(op, library_fft=False)
        self.fused = FusedHex8Preconditioner(op, config)

    def energy_norm(self):
        """Return the fused reference-energy measure of the current residual."""
        return self.fused.measure(self.r)

    def release(self):
        """Release preconditioner storage for a fresh host-solution evaluation.

        z and the preconditioner spectrum give their memory to a fresh
        evaluation of a host-resident solution (_CudaPCG.release).
        """
        super().release()
        self.fused.S = None

    def restore(self):
        """Restore the Krylov buffers and zero-padded preconditioner spectrum."""
        super().restore()
        f = self.fused
        n0, n1, _ = f.shape
        # Padding columns must be zero (see FusedHex8Preconditioner).
        f.S = cp.zeros(
            3 * n0 * n1 * f.tiles * f.T,
            np.complex64 if f.dtype == np.float32 else np.complex128,
        )

    def released_bytes(self):
        """Return the bytes made available by releasing the current workspaces."""
        f = self.fused
        return (
            super().released_bytes()
            + 3 * f.shape[0] * f.shape[1] * f.tiles * f.T * 2 * f.dtype.itemsize
        )

    def _beta(self, iteration, threshold, mode):
        f = self.fused
        self.k["hex8_pcg_beta"](
            (1,),
            (_REDUCTION_THREADS,),
            (
                f.rho,
                np.int32(f.rho.size),
                f.rmax,
                np.int32(f.rmax.size),
                self.s,
                self.state,
                np.float64(threshold),
                np.int32(iteration),
                np.int32(mode),
                self.replace,
                self.history,
                self.gamma,
                self.energy,
            ),
        )

    def _precondition(self, iteration, threshold, mode):
        """Fresh residual in r (prepare, modes 1/2): z materialized in self.z.

        The inverse passes ignore the state, as the cuFFT C2R does, so z is
        written even if beta rejects rho (state 2 then stops the next iterate).
        """
        f = self.fused
        f.forward(self.r, self.state)
        f.multiply(self.state)
        self._beta(iteration, threshold, mode)
        f.backward(self.z, f.idle)

    def iterate(self, x, first, last, threshold):
        """Queue iterations first+1..last; return (state, finishing iteration).

        Same loop and host lookahead as _CudaPCG.iterate; the update kernel,
        both cuFFT calls, hex8_pcg_prec and the direction kernel are replaced.
        """
        k, f = self.k, self.fused
        x = self.increment if self.mixed else x
        host = not isinstance(x, cp.ndarray)
        if host:
            # A host-resident iterate (coordinates._HostVector): x += alpha p
            # streams on a side stream while the preconditioner runs, as of
            # the state when alpha formed; p changes only after it.
            settle = _reduction_kernel("hex8_settle", self.p.dtype, x.dtype)
            p, side = self.p.reshape(-1), _shared_streams("host_update", 1)[0]

            def launch(xv, yv, base, n):
                settle(
                    (_reduction_grid(n),),
                    (_REDUCTION_THREADS,),
                    (p[base : base + n], xv, self.s, self.go, np.int64(n)),
                )

        polled = None
        for iteration in range(first + 1, last + 1):
            m = self.op._action_partials(self.p, self.ap, self.state, self.pap)
            k["hex8_pcg_alpha"](
                (1,),
                (_REDUCTION_THREADS,),
                (
                    self.pap,
                    np.int32(m),
                    self.s,
                    self.state,
                    self.history,
                    np.int32(iteration),
                ),
            )
            if host:
                cp.copyto(self.go, self.state[:1])
                updated = x.settle(launch, stream=side)
            move = f.move_x and not host
            f.forward(
                self.r,
                self.state,
                update=True,
                ap=self.ap,
                p=self.p,
                x=None if (host or move) else x,
                s=self.s,
            )
            f.multiply(self.state)
            self._beta(iteration, threshold, 0)
            if host:
                cp.cuda.get_current_stream().wait_event(updated)
            f.backward(
                self.p,
                self.state,
                s=self.s,
                direction=True,
                x=x if move else None,
                iteration=iteration,
            )
            slot = iteration % 2
            self.state.get(out=self.host[slot], blocking=False)
            self.events[slot].record()
            if polled is not None:
                self.events[polled % 2].synchronize()
                if self.host[polled % 2, 0]:
                    break
            polled = iteration
        state = self.state.get()
        return int(state[0]), int(state[1])


_READY, _BACKGROUND = set(), {}


def fused_pcg(op):
    """Return the fused PCG of a CUDA operator, or None for the cuFFT path.

    Grids outside the generated-FFT contract keep cuFFT. A small grid whose
    kernels were never compiled also keeps it while they compile in a background
    thread (as fused.defer_compile); the kernel cache file records them for
    later processes. CUFHOM_FUSED_COMPILE=sync compiles before solving.
    """
    import os

    shape, dtype = tuple(int(n) for n in op.shape), np.dtype(op.working)
    limit = device_properties()["sharedMemPerBlockOptin"]
    if not fused_hex8_supported(shape, dtype, limit):
        return None
    cfg = fused_hex8_config(shape, dtype, limit)
    specs = fused_hex8_sources(shape, cfg, dtype)
    key = _compiled_key(specs, list(specs))
    if (
        key not in _READY
        and os.environ.get("CUFHOM_FUSED_COMPILE", "background") != "sync"
        and int(np.prod(shape)) <= _COMPILE_MIN_VOXELS
        and key not in read_records(cache_path())
    ):
        if key not in _BACKGROUND:
            # Not a daemon: interpreter exit waits for NVRTC (see fused.defer_compile).
            _BACKGROUND[key] = threading.Thread(
                target=_compile_and_mark, args=(specs, key)
            )
            _BACKGROUND[key].start()
        return None
    pcg = FusedCudaPCG(op, cfg)
    if key not in _READY:
        _READY.add(key)
        save_record(cache_path(), key, dict(kernels=list(specs)))
    return pcg


def compile_jobs(op):
    """Precompile supported generated kernels when compilation is synchronous.

    Compile callables of this operator's generated kernels, when they will
    be compiled synchronously anyway (supported grid, not deferred).
    """
    import os

    from .fused import _compile

    shape, dtype = tuple(int(n) for n in op.shape), np.dtype(op.working)
    limit = device_properties()["sharedMemPerBlockOptin"]
    if not fused_hex8_supported(shape, dtype, limit):
        return []
    if (
        os.environ.get("CUFHOM_FUSED_COMPILE", "background") != "sync"
        and int(np.prod(shape)) <= _COMPILE_MIN_VOXELS
    ):
        return []
    specs = fused_hex8_sources(shape, fused_hex8_config(shape, dtype, limit), dtype)
    return [lambda spec=spec: _compile(*spec) for spec in specs.values()]


def _compile_and_mark(specs, key):
    try:
        _compile_many(list(specs.values()))
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        return
    _READY.add(key)
    save_record(cache_path(), key, dict(kernels=list(specs)))
