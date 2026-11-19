"""Double-float (df64) builds of the constitutive CUDA kernels.

Consumer GPUs run FP64 at 1/64 of the FP32 rate, which makes the FP64
return maps the bottleneck of the mixed-precision nonlinear route. A df64
build keeps the FP64 storage and kernel parameters, and evaluates the law in
pairs of FP32 words (Dekker/Hida double-float, the arithmetic of the linear
mixed route: about 46 significant bits per operation) at the FP32 rate.

Law sources are written with three macros besides the arithmetic type REAL:
MEM (storage type), PARAM (scalar kernel parameter type), L(x) for every
global-memory or parameter read and S(x) for every store. The IEEE builds
define them as the identity, so their expressions are unchanged.
"""

import re

import numpy as np
from cufhom.fft.linear.evaluation import _DF, _pair

_INVERSE_FACTORIALS = "\n".join(
    f" p=p*r+{_pair(1 / np.prod(np.arange(1, n + 1, dtype=float)))};"
    for n in range(10, 0, -1)
)

_LAW = (
    _DF
    + r"""
__device__ __forceinline__ df& operator/=(df& a,df b){a=a/b;return a;}
__device__ __forceinline__ bool operator<(df a,df b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}
__device__ __forceinline__ bool operator>(df a,df b){return b<a;}
__device__ __forceinline__ bool operator<=(df a,df b){return !(b<a);}
__device__ __forceinline__ bool operator>=(df a,df b){return !(a<b);}
__device__ __forceinline__ bool operator==(df a,df b){return a.hi==b.hi&&a.lo==b.lo;}
__device__ __forceinline__ bool operator!=(df a,df b){return !(a==b);}
__device__ __forceinline__ df fabs(df a){return a.hi<0.f?-a:a;}
__device__ __forceinline__ df fmax(df a,df b){return a<b?b:a;}
__device__ __forceinline__ df fmin(df a,df b){return b<a?b:a;}
__device__ __forceinline__ bool isfinite(df a){return isfinite(a.hi)&&isfinite(a.lo);}
__device__ __forceinline__ df sqrt(df a){return df_sqrt(a);}
// exp: k = round(a/ln2), r = (a - k ln2)/8 (|r| < 0.044), degree-10 Taylor
// (truncation < 2e-23), three squarings (8x rounding growth, not 32x) and an
// exact scaling by 2^k. Its relative error is dominated by the df input
// itself (|a| 2^-48: 1e-13 at |a| = 40), as for any argument representation.
__device__ __forceinline__ df exp(df a){
 if(!(a.hi<88.f))return df(a.hi!=a.hi?a.hi:__int_as_float(0x7f800000));
 if(a.hi<-87.f)return df(0.f);
 const float k=rintf(__fdiv_rn(a.hi,0.693147182f));
 const df r=(a-LN2*df(k))*df(0.125f);
 df p=df(0.f);
INVERSE_FACTORIALS
 p=p*r+df(1.f);
 for(int i=0;i<3;i++)p=p*p;
 const int e=(int)k;
 return df(ldexpf(p.hi,e),ldexpf(p.lo,e));}
""".replace("LN2", _pair(np.log(2.0))).replace(
        "INVERSE_FACTORIALS", _INVERSE_FACTORIALS
    )
)


def df64_source(source):
    """Build a law source (REAL, MEM, PARAM, L, S) in double-float arithmetic.

    REAL becomes df; storage and parameters stay FP64. Casts of numeric
    literals to REAL become two-word pairs (an FP32 literal would round).
    """
    source = re.sub(
        r"\(REAL\)(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?![\w.])",
        lambda m: _pair(float(m.group(1))),
        source,
    )
    defines = (
        "#define MEM double\n#define PARAM double\n"
        "#define L(x) from_f64(x)\n#define S(x) to_f64(x)\n"
    )
    return _LAW + defines + re.sub(r"\bREAL\b", "df", source)


DF64_EPSILON = 2.0**-46  # stopping resolution of iterations evaluated in df64
