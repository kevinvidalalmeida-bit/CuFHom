"""CUDA source specialization and shared FP32/FP64/df64 arithmetic.

FP32 source is returned byte for byte. FP64 changes scalar/vector types,
intrinsics and literal suffixes; constants must be generated in FP64 first.
"""

import re

import numpy as np


def real_dtype(dtype):
    """Return the FP32 or FP64 real dtype associated with a supported scalar dtype."""
    dtype = np.dtype(dtype)
    if dtype in (np.dtype("float32"), np.dtype("complex64")):
        return np.dtype("float32")
    if dtype in (np.dtype("float64"), np.dtype("complex128")):
        return np.dtype("float64")
    raise ValueError("Linear CUDA kernels require FP32 or FP64")


def cuda_source(source, dtype):
    """Specialize FP32 CUDA source for the requested FP32 or FP64 precision."""
    if real_dtype(dtype) == np.dtype("float32"):
        return source
    for old, new in [
        ("make_float2", "make_double2"),
        ("make_float4", "make_double4"),
        ("float2", "double2"),
        ("float4", "double4"),
        ("float", "double"),
        ("sqrtf", "sqrt"),
        ("rsqrtf", "rsqrt"),
        ("fmaf", "fma"),
        ("nanf", "nan"),
        ("fabsf", "fabs"),
    ]:
        source = re.sub(r"\b" + old + r"\b", new, source)
    return re.sub(r"(\b(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[fF]\b", r"\1", source)


# Double-float arithmetic: (hi, lo) with |lo| <= ulp(hi)/2. The intrinsics
# are never contracted or reassociated by the compiler.
DF64_CUDA = r"""
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


def double_float_literal(x):
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
        lambda m: double_float_literal(float(m.group(1))),
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
    return DF64_CUDA + source.replace('nanf("")', 'df(nanf(""))').replace(
        "float2_t", "float2"
    )
