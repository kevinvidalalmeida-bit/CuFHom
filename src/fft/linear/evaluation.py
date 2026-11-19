"""Fused residual evaluations of the mixed-precision GaNi route.

Every replacement and candidate of the mixed route evaluates the FP64
problem in total-strain form,

    r = fl32(-RQ* FFT(C (E + C2R(QR z)))),  ||G h||^2,  <sigma>,

which the library route does with cuFFT FP64 (six components) at the 1:64
FP64 rate of consumer GPUs. Here the five-stage generated pipeline of the
FP32 Krylov operator (fused.py) runs the same formulas in

  df64     pairs of FP32 (Dekker/Hida double-float: about 46 significant bits
           per tested operation, non-contractible FP32 intrinsics), at the FP32
           rate; the default, or
  float64  IEEE FP64 (the generator's FP64 specialization).

The macro strain is added in the material stage; the last stage writes the
FP32 residual in the Krylov vector layout and accumulates r.r (of the
stored values), the physical audit ||G h||^2 from the coordinate residual
(see coordinates._residual_map) and the k = 0 stress sum in FP64. Inputs
are FP64 coordinates, read by second-axis slabs so that host-resident
solutions stream as in coordinates._HostVector.
"""

import os
import re
import threading
from math import gcd

import numpy as np
from cufhom_core.tensors.fft import cp

from .fused import (
    _SOURCE_REAL,
    _axis1,
    _axis2_material,
    _compact_ids,
    _compile_many,
    _coordinate_code,
    _fft,
    _generic_kernels,
    _spectrum_index,
    _view,
    default_config,
    grid_supported,
    kc_shared_complex,
    stored_planes,
)
from .kernel_cache import device_properties
from .precision import cuda_source

ARITHMETICS = ("df64", "float64")


class _DoubleFloat(np.float64):
    """Codegen marker: constants are split into (hi, lo) FP32 pairs."""

    @staticmethod
    def literal(x):
        return _pair(x)


# Double-float arithmetic: (hi, lo) with |lo| <= ulp(hi)/2. The intrinsics
# are never contracted or reassociated by the compiler.
_DF = r"""
struct df {
 float hi, lo;
 df() = default;
 __device__ __forceinline__ df(float h):hi(h),lo(0.f){}
 __device__ __forceinline__ df(int i):hi((float)i),lo(0.f){}
 __device__ __forceinline__ df(float h,float l):hi(h),lo(l){}
};
struct df2 {df x, y;};
struct df4 {df x, y, z, w;};
__device__ __forceinline__ df2 make_df2(df x,df y){df2 v;v.x=x;v.y=y;return v;}
__device__ __forceinline__ df quick_two_sum(float a,float b){
 const float s=__fadd_rn(a,b);return df(s,__fsub_rn(b,__fsub_rn(s,a)));}
__device__ __forceinline__ df two_sum(float a,float b){
 const float s=__fadd_rn(a,b),v=__fsub_rn(s,a);
 return df(s,__fadd_rn(__fsub_rn(a,__fsub_rn(s,v)),__fsub_rn(b,v)));}
__device__ __forceinline__ df operator+(df a,df b){
 df s=two_sum(a.hi,b.hi);const df t=two_sum(a.lo,b.lo);
 s=quick_two_sum(s.hi,__fadd_rn(s.lo,t.hi));
 return quick_two_sum(s.hi,__fadd_rn(s.lo,t.lo));}
__device__ __forceinline__ df operator-(df a){return df(-a.hi,-a.lo);}
__device__ __forceinline__ df operator-(df a,df b){return a+(-b);}
__device__ __forceinline__ df operator*(df a,df b){
 const float p=__fmul_rn(a.hi,b.hi);
 float e=__fmaf_rn(a.hi,b.hi,-p);
 e=__fmaf_rn(a.hi,b.lo,e);e=__fmaf_rn(a.lo,b.hi,e);
 return quick_two_sum(p,e);}
__device__ __forceinline__ df operator/(df a,df b){
 const float q1=__fdiv_rn(a.hi,b.hi);
 const df r=a-b*df(q1);
 const float q2=__fdiv_rn(r.hi,b.hi);
 return df(q1)+df(q2)+(r-b*df(q2))*df(__frcp_rn(b.hi));}
__device__ __forceinline__ df& operator+=(df& a,df b){a=a+b;return a;}
__device__ __forceinline__ df& operator-=(df& a,df b){a=a-b;return a;}
__device__ __forceinline__ df& operator*=(df& a,df b){a=a*b;return a;}
__device__ __forceinline__ df df_sqrt(df a){
 if(!(a.hi>0.f))return df(sqrtf(a.hi));
 const float s=__fsqrt_rn(a.hi);
 return df(s)+(a-df(s)*df(s))*df(__fdiv_rn(.5f,s));}
__device__ __forceinline__ df df_fma(df a,df b,df c){return a*b+c;}
__device__ __forceinline__ float to_f32(df a){return __fadd_rn(a.hi,a.lo);}
__device__ __forceinline__ double to_f64(df a){return (double)a.hi+(double)a.lo;}
__device__ __forceinline__ df from_f64(double a){const float h=(float)a;return df(h,(float)(a-(double)h));}
"""
_FP64 = r"""
__device__ __forceinline__ float to_f32(double a){return (float)a;}
__device__ __forceinline__ double to_f64(double a){return a;}
__device__ __forceinline__ double from_f64(double a){return a;}
"""


def _pair(x):
    """Represent an FP64 value as a rounded two-word FP32 df literal."""
    hi = np.float32(x)
    lo = np.float32(np.float64(x) - np.float64(hi))
    return f"df({float(hi)!r}f,{float(lo)!r}f)"


def df_source(source):
    """FP32 generator source in double-float arithmetic.

    Types float/float2/float4 become df/df2/df4 (float2_t stays FP32);
    generated constants are already two-word pairs (see _f), and FP32 casts of
    literals become pairs too. Hand-written FP32 literals are exact.
    """
    source = re.sub(
        r"\(float\)(\d+\.\d+(?:[eE][+-]?\d+)?)",
        lambda m: _pair(float(m.group(1))),
        source,
    )
    for old, new in [
        ("make_float2", "make_df2"),
        ("float2", "df2"),
        ("float4", "df4"),
        ("float", "df"),
        ("sqrtf", "df_sqrt"),
        ("fmaf", "df_fma"),
    ]:
        source = re.sub(r"\b" + old + r"\b", new, source)
    return _DF + source.replace('nanf("")', 'df(nanf(""))').replace(
        "float2_t", "float2"
    )


def _axis0_inverse(shape, T, TV, NT, coord, Tb):
    """Generate the coordinate-map and first-axis inverse-transform stage.

    KA: FP64 coordinates of second-axis rows [k1_start, k1_start+k1_count)
    (offset base in z), strain map and inverse FFT along axis 0.
    """
    n0, n1, n2 = shape
    M, W = stored_planes(n2), 6 * T
    return (
        coord
        + f"""
extern "C" __global__ void __launch_bounds__({NT}) ka_eval(const SOLUTION2* z,float2* S,const float2* tw_g,
 double transverse_d,double longitudinal_d,int k1_start,int k1_count,long long base) {{
 const float transverse=from_f64(transverse_d),longitudinal=from_f64(longitudinal_d);
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n0 * W};
 const int tid=threadIdx.x;
 const int k1=k1_start+blockIdx.x%k1_count,k2=(blockIdx.x/k1_count)*{T};
 for(int i=tid;i<{n0};i+={NT})tw[i]=tw_g[i];
 for(int i=tid;i<{n0 * T};i+={NT}) {{
   const int k0=i/{T},kk=k2+i%{T};
   const long long o=((((long long)k1*{M // TV}+kk/{TV})*{n0}+k0)*{TV}+kk%{TV})*3-base;
   float2 q[3],e[6];
   for(int c=0;c<3;c++){{const SOLUTION2 v=z[o+c];q[c]=make_float2(from_f64((double)v.x),from_f64((double)v.y));}}
   strain_map(q,e,k0,k1,kk,transverse,longitudinal);
   for(int c=0;c<6;c++)buf[k0*{W}+(i%{T})*6+c]=e[c];
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


def _axis2_total_strain(shape, LPB, NT, Tb, id_type):
    """KC with the macro strain added before the material law (no energy)."""
    code = _axis2_material(shape, LPB, NT, Tb, id_type)
    for old, new in [
        ("kc(float2* S,", "kc_eval(float2* S,const float* macro,"),
        (
            "     const float* x=u[e];",
            "     for(int c=0;c<6;c++)u[e][c]+=macro[c];\n     const float* x=u[e];",
        ),
        (
            "     float ev=0.f;\n     for(int c=0;c<6;c++)ev=fmaf(x[c],v[e][c],ev);\n     sum+=ev;\n",
            "",
        ),
    ]:
        if code.count(old) != 1:
            raise RuntimeError(
                "Review changed the material stage before specializing evaluations"
            )
        code = code.replace(old, new)
    return code


def _axis0_forward(shape, T, TV, NT, coord, Tb, norm):
    """Generate the forward-transform, adjoint-map and fresh-audit stage.

    KA: forward FFT along axis 0, RQ*, the FP32 residual in the Krylov
    layout and FP64 block sums of r.r and of the audit; the k = 0 stress sum.
    """
    n0, n1, n2 = shape
    M, W = stored_planes(n2), 6 * T
    return (
        coord
        + f"""
extern "C" __global__ void __launch_bounds__({NT}) ka_eval_fwd(const float2* S,float2_t* r,double* partial,
 double* mean,const float2* tw_g,double transverse_d,double longitudinal_d) {{
 const float transverse=from_f64(transverse_d),longitudinal=from_f64(longitudinal_d);
 extern __shared__ float2 sm[];
 float2* buf=sm; float2* tw=sm+{n0 * W};
 __shared__ double sums[2][32];
 const int tid=threadIdx.x;
 const int k1=blockIdx.x%{n1},k2=(blockIdx.x/{n1})*{T};
 for(int i=tid;i<{n0};i+={NT})tw[i]=tw_g[i];
 for(int i=tid;i<{n0 * W};i+={NT}) {{
   const int x0=i/{W},w=i%{W};
   buf[i]=S[{_spectrum_index(shape, Tb, "x0", "k1", "k2+w/6", "w%6")}];
 }}
 __syncthreads();
 {_fft(n0, W, NT, -1)}
 if(k1==0&&k2==0&&tid<6)mean[tid]=to_f64(buf[tid].x);
 double rr=0,audit=0;
 for(int i=tid;i<{n0 * T};i+={NT}) {{
   const int k0=i/{T},kk=k2+i%{T};
   const long long o=((((long long)k1*{M // TV}+kk/{TV})*{n0}+k0)*{TV}+kk%{TV})*3;
   float2 h[6],ap[3];
   for(int c=0;c<6;c++)h[c]=buf[k0*{W}+(i%{T})*6+c];
   adjoint_map(h,ap,k0,k1,kk,transverse,longitudinal);
   float k[3];
   if(!unit_k(k0,k1,kk,k))for(int c=0;c<3;c++)k[c]=0;
   float a2=0,px=0,py=0;
   for(int c=0;c<3;c++) {{
     const float2_t v=make_float2_t(to_f32(-ap[c].x),to_f32(-ap[c].y));
     r[o+c]=v;
     rr+=(double)v.x*v.x+(double)v.y*v.y;
     a2+=ap[c].x*ap[c].x+ap[c].y*ap[c].y;px+=k[c]*ap[c].x;py+=k[c]*ap[c].y;
   }}
   const float p2=px*px+py*py;
   audit+=to_f64((a2-p2)/(transverse*transverse)+p2/(longitudinal*longitudinal));
 }}
 for(int o=16;o;o>>=1){{rr+=__shfl_down_sync(0xffffffff,rr,o);audit+=__shfl_down_sync(0xffffffff,audit,o);}}
 if((tid&31)==0){{sums[0][tid>>5]=rr;sums[1][tid>>5]=audit;}}
 __syncthreads();
 if(tid<32) {{
   rr=tid<({NT}>>5)?sums[0][tid]:0;audit=tid<({NT}>>5)?sums[1][tid]:0;
   for(int o=16;o;o>>=1){{rr+=__shfl_down_sync(0xffffffff,rr,o);audit+=__shfl_down_sync(0xffffffff,audit,o);}}
   if(tid==0){{partial[blockIdx.x]=rr;partial[gridDim.x+blockIdx.x]=audit/({float(norm)!r}*{float(norm)!r});}}
 }}
}}
"""
    )


_FLOAT2_T = r"""
typedef float2 float2_t;
__device__ __forceinline__ float2_t make_float2_t(float x,float y){return make_float2(x,y);}
"""
_DOUBLE2_T = r"""
typedef double2 float2_t;
__device__ __forceinline__ float2_t make_float2_t(double x,double y){return make_double2(x,y);}
"""


def _specs(
    shape,
    lengths,
    norm,
    cfg,
    TV,
    compact,
    arithmetic,
    solution="float64",
    residual="float32",
):
    """(source, name, shared memory) of the four evaluation kernels.

    solution and residual are the storage precisions of the coordinates read
    by the inverse kernel and of the residual written by the forward one.
    """
    n0, n1, n2 = shape
    Ta, Tb, LPB = cfg["ta"], cfg["tb"], cfg["lpb"]
    marker = _DoubleFloat if arithmetic == "df64" else np.float64
    token = _SOURCE_REAL.set(marker)
    try:
        coord = _coordinate_code(shape, lengths, norm)
        sources = {
            "inverse": (
                _axis0_inverse(shape, Ta, TV, cfg["nta_inv"], coord, Tb),
                "ka_eval",
                (n0 * 6 * Ta + n0) * 16,
            ),
            "axis1_inv": (
                _axis1(shape, Tb, cfg["ntb"], +1),
                "kb_inv",
                (n1 * 6 * Tb + n1) * 16,
            ),
            "material": (
                _axis2_total_strain(
                    shape, LPB, cfg["ntc"], Tb, "unsigned short" if compact else "int"
                ),
                "kc_eval",
                kc_shared_complex(n2, LPB) * 16,
            ),
            "axis1_fwd": (
                _axis1(shape, Tb, cfg["ntb"], -1),
                "kb_fwd",
                (n1 * 6 * Tb + n1) * 16,
            ),
            "forward": (
                _axis0_forward(shape, Ta, TV, cfg["nta_fwd"], coord, Tb, norm),
                "ka_eval_fwd",
                (n0 * 6 * Ta + n0) * 16,
            ),
        }
    finally:
        _SOURCE_REAL.reset(token)
    specialize = (
        df_source
        if arithmetic == "df64"
        else (lambda s: _FP64 + cuda_source(s, np.float64))
    )
    specs = {}
    for name, (source, entry, smem) in sources.items():
        code = specialize(source).replace(
            "SOLUTION2", "double2" if solution == "float64" else "float2"
        )
        if residual == "float64":
            code = code.replace(
                "make_float2_t(to_f32(-ap[c].x),to_f32(-ap[c].y))",
                "make_float2_t(to_f64(-ap[c].x),to_f64(-ap[c].y))",
            )
        specs[name] = (
            (_DOUBLE2_T if residual == "float64" else _FLOAT2_T) + code,
            entry,
            smem,
        )
    return specs


def _real_pairs(values, arithmetic):
    """FP64 values as kernel storage: (hi, lo) FP32 pairs, or FP64."""
    values = np.asarray(values, np.float64)
    if arithmetic == "float64":
        return cp.asarray(values)
    hi = values.astype(np.float32)
    return cp.asarray(
        np.stack([hi, (values - hi.astype(np.float64)).astype(np.float32)], axis=-1)
    )


def _complex_pairs(values, arithmetic):
    values = np.asarray(values, np.complex128)
    if arithmetic == "float64":
        return cp.asarray(values)
    return _real_pairs(np.stack([values.real, values.imag], axis=-1), arithmetic)


def evaluation_supported(Q, descriptor):
    """Check that the tiled coordinate layout fits the evaluation FFT workspace.

    Tiled coordinates (the fused Krylov layout) and a grid whose
    16-byte-complex line FFTs fit shared memory.
    """
    return Q.tile is not None and grid_supported(Q.shape, np.float64)


class EvaluationPipeline:
    """Fused total-strain residual evaluations of one grid and FP64 material.

    The hybrid spectrum (16-byte complex, the FP64 work field's 48 B/voxel)
    is allocated per evaluation and released with release(). ids: the
    compact material ids of the Krylov pipeline (fused._compact_ids), if any.
    solution and residual: storage precisions of the coordinates read and of
    the residual written ('float64' accumulated coordinates and an FP32
    residual in the mixed configuration, the working precision otherwise).
    """

    def __init__(
        self,
        Q,
        descriptor,
        arithmetic="df64",
        ids=None,
        solution="float64",
        residual="float32",
    ):
        if arithmetic not in ARITHMETICS:
            raise ValueError(f"arithmetic must be one of {ARITHMETICS}")
        if not evaluation_supported(Q, descriptor):
            raise ValueError(
                "Fused evaluations need the tiled layout and a supported grid"
            )
        self.arithmetic, self.Q = arithmetic, Q
        self.solution = np.dtype("complex128" if solution == "float64" else "complex64")
        self.residual = np.dtype("complex128" if residual == "float64" else "complex64")
        self.shape = n0, n1, n2 = tuple(int(n) for n in Q.shape)
        self.M = M = stored_planes(n2)
        self.config = cfg = default_config(
            self.shape, device_properties()["sharedMemPerBlockOptin"], np.float64
        )
        # KC groups lpb consecutive lines. Every FFT slab boundary must also
        # be a group boundary, including the final partial slab. Keep lpb and
        # all workspaces unchanged; only align the number of axis-0 planes.
        planes = cfg["lpb"] // gcd(n1, cfg["lpb"])
        cfg["slabs"] = max(planes, cfg["slabs"] // planes * planes)
        rows = int(descriptor.material_table.shape[0])
        if ids is None:
            _generic_kernels()  # the id compaction kernel
            ids, _ = _compact_ids(descriptor.val.reshape(self.shape), rows)
        self.ids, compact = ids, ids.dtype == cp.uint16
        key = (
            self.shape,
            tuple(float(L) for L in Q.P.lengths),
            float(Q.norm),
            int(Q.tile),
            compact,
            arithmetic,
            solution,
            residual,
        )
        parts = _PARTS.get(key)
        if parts is None:
            specs = _specs(
                self.shape,
                Q.P.lengths,
                Q.norm,
                cfg,
                int(Q.tile),
                compact,
                arithmetic,
                solution,
                residual,
            )

            def twiddle(n, m=None):
                return _complex_pairs(
                    np.exp(-2j * np.pi * np.arange(n) / (m or n)), arithmetic
                )

            # KC twiddles as in FusedPipeline: half-length pair on even grids, length n2 on odd ones.
            parts = _PARTS[key] = dict(
                kernels=None,
                specs=specs,
                twiddles=(
                    (twiddle(n0), twiddle(n1), twiddle(M), twiddle(M, n2))
                    if n2 % 2 == 0
                    else (twiddle(n0), twiddle(n1), twiddle(n2), twiddle(1))
                ),
            )

            def build():
                parts["kernels"] = dict(zip(specs, _compile_many(list(specs.values()))))

            # A new grid's double-float kernels take ~1 s to compile: the first
            # solve compiles them while its FP32 iterations run and evaluates
            # with cuFFT FP64 until they are ready (ready); later solves and
            # processes load them from the kernel cache.
            if os.environ.get("CUFHOM_FUSED_COMPILE", "background") == "sync":
                build()
            else:
                # Not a daemon: interpreter exit waits for NVRTC instead of
                # tearing CUDA down under it (segfault at exit).
                parts["thread"] = threading.Thread(target=build)
                parts["thread"].start()
        self._parts, self.twiddles = parts, parts["twiddles"]
        self.smem = {name: spec[2] for name, spec in parts["specs"].items()}
        table = np.zeros((rows, 24))
        table[:, :21] = cp.asnumpy(descriptor.material_table)
        self.table = _real_pairs(table, arithmetic)
        self.moduli = (np.float64(Q.transverse), np.float64(Q.longitudinal))
        self.grid_a = n1 * M // cfg["ta"]
        self.partial = cp.empty(2 * self.grid_a, cp.float64)
        self.mean = cp.empty(6, cp.float64)
        self.grid_c = -(-n0 * n1 // cfg["lpb"])
        self.material_partial = cp.empty(self.grid_c, cp.float64)
        self.material_scalar = cp.empty(1, cp.float64)
        self.counter = cp.zeros(1, cp.uint32)
        self.never_done = cp.zeros(1, cp.int32)
        self.S = None
        self.workspace = None  # as FusedPipeline.workspace

    @property
    def ready(self):
        """Return whether asynchronous compilation has produced evaluation kernels."""
        return self._parts["kernels"] is not None

    @property
    def kernels(self):
        """Return the compiled evaluation kernels, or None while compilation is pending."""
        return self._parts["kernels"]

    def wait(self):
        """Wait until asynchronous evaluation-kernel compilation finishes.

        Block until the kernels are compiled (instead of the cuFFT FP64
        evaluations meanwhile, which a large grid may not have room for).
        """
        thread = self._parts.get("thread")
        if thread is not None:
            thread.join()

    def zero(self):
        """Initialize the zero-fluctuation spectrum for a macroscopic load.

        The strain spectrum of z = 0: forward(macro, r) then evaluates
        the load, -RQ* C E (the macro strain enters the material stage).
        """
        n0, n1, _ = self.shape
        if self.S is None:
            self.S = _view((n0, n1, self.M, 6), cp.complex128, self.workspace)
        self.S.fill(0)

    def inverse(self, z, k1_start=0, k1_count=None, base=0):
        """Reconstruct strain spectra from a slab of tiled coordinates.

        Strain spectra of second-axis rows [k1_start, k1_start+k1_count)
        from tile-major coordinates z (z[0] is coordinate base).
        """
        if getattr(z, "dtype", self.solution) != self.solution:
            raise ValueError(
                "Coordinates do not have the precision of this evaluation pipeline"
            )
        n0, n1, _ = self.shape
        k1_count = n1 if k1_count is None else k1_count
        if self.S is None:
            self.S = _view((n0, n1, self.M, 6), cp.complex128, self.workspace)
        self.kernels["inverse"](
            (k1_count * self.M // self.config["ta"],),
            (self.config["nta_inv"],),
            (
                z,
                self.S,
                self.twiddles[0],
                *self.moduli,
                np.int32(k1_start),
                np.int32(k1_count),
                np.int64(base),
            ),
            shared_mem=self.smem["inverse"],
        )

    def forward(self, macro, r):
        """Finish the constitutive evaluation and return fresh residual diagnostics.

        Finish the evaluation of the inverted coordinates with macro strain
        E: writes r = fl(-RQ* h) and returns (r.r, ||G h||^2, sum sigma).
        """
        if getattr(r, "dtype", self.residual) != self.residual:
            raise ValueError(
                "Residual does not have the precision of this evaluation pipeline"
            )
        cfg, (n0, n1, _) = self.config, self.shape
        macro = _real_pairs(macro, self.arithmetic)
        tw0, tw1, tw2, twn = self.twiddles
        chunk, lpb = cfg["slabs"], cfg["lpb"]
        material = (self.ids, self.table, np.int64(self.table.shape[0]))
        for s0 in range(0, n0, chunk):
            c = min(chunk, n0 - s0)
            self.kernels["axis1_inv"](
                (self.M // cfg["tb"], c),
                (cfg["ntb"],),
                (self.S, tw1, np.int32(s0), self.never_done),
                shared_mem=self.smem["axis1_inv"],
            )
            self.kernels["material"](
                (-(-c * n1 // lpb),),
                (cfg["ntc"],),
                (self.S, macro)
                + material
                + (
                    self.material_partial,
                    tw2,
                    twn,
                    self.counter,
                    self.material_scalar,
                    np.int64(s0 * n1 // lpb),
                    np.uint32(self.grid_c),
                    self.never_done,
                ),
                shared_mem=self.smem["material"],
            )
            self.kernels["axis1_fwd"](
                (self.M // cfg["tb"], c),
                (cfg["ntb"],),
                (self.S, tw1, np.int32(s0), self.never_done),
                shared_mem=self.smem["axis1_fwd"],
            )
        self.kernels["forward"](
            (self.grid_a,),
            (cfg["nta_fwd"],),
            (self.S, r, self.partial, self.mean, tw0, *self.moduli),
            shared_mem=self.smem["forward"],
        )
        rr, audit = cp.asnumpy(self.partial.reshape(2, -1).sum(axis=1))
        return float(rr), float(audit), cp.asnumpy(self.mean)

    def release(self):
        """Release the retained spectral evaluation workspace."""
        self.S = None


_PARTS = {}
