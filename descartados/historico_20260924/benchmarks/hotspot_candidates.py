"""Exact-operation GPU candidates; imported by benchmarks, never production."""
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import numpy as np
from profile_g09_fft import cp
from pipeline import nonlinear_coordinates as coords
from pipeline import linear_krylov as linear
from pipeline.nonlinear_material import CuPyMaterial
from ffthompy.tensors import fft

_RFFT = fft.rfftn_cached
_IRFFT = fft.irfftn_unscaled_cached


def _split_fft(x, shape, axes, inverse):
    original = _IRFFT if inverse else _RFFT
    expected = tuple(shape[:-1])+(shape[-1]//2+1,) if inverse else tuple(shape)
    if not (isinstance(x, cp.ndarray) and x.ndim == 4 and x.shape[0] == 6
            and tuple(a % 4 for a in axes) == (1,2,3) and x.shape[1:] == expected
            and np.prod(shape) % 2 == 0  # cuFFT complex alignment at each component offset
            and x.flags.c_contiguous and fft.CUPY_PLAN_MODE == 'manual'):
        return original(x, shape, axes)
    dtype = x.real.dtype if inverse else (cp.complex64 if x.dtype == cp.float32 else cp.complex128)
    outshape = tuple(shape) if inverse else tuple(shape[:-1])+(shape[-1]//2+1,)
    out = cp.empty((6,)+outshape, dtype=dtype)
    plan = fft._maybe_cupy_plan(x[:1], (1,2,3), 'C2R' if inverse else 'R2C',
                                shape=shape if inverse else None)
    for component in range(6):
        plan.fft(x[component:component+1], out[component:component+1],
                 cp.cuda.cufft.CUFFT_INVERSE if inverse else cp.cuda.cufft.CUFFT_FORWARD)
    return out


def split_rfft(x, shape, axes):
    return _split_fft(x, shape, axes, False)


def split_irfft(x, shape, axes):
    return _split_fft(x, shape, axes, True)

_TANGENT = r'''
extern "C" __global__ void tangent_registers(
 const REAL* __restrict__ v,const REAL* __restrict__ u,
 const REAL* __restrict__ a,const REAL* __restrict__ b,
 const int* __restrict__ ids,const REAL* __restrict__ table,
 REAL* __restrict__ out,long long n,REAL K) {
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(k>=n)return;
 REAL value[6];
 #pragma unroll
 for(int c=0;c<6;c++)value[c]=v[c*n+k];
 const int id=ids[k];
 if(id) {
   #pragma unroll
   for(int c=0;c<6;c++){
     REAL w=0;
     #pragma unroll
     for(int j=0;j<6;j++)w+=table[(long long)id*36+c*6+j]*value[j];
     out[c*n+k]=w;
   }
 } else {
   const REAL ac=a[k],bc=b[k],tr=value[0]+value[1]+value[2];
   REAL direction[6],dot=0;
   #pragma unroll
   for(int c=0;c<6;c++){direction[c]=u[c*n+k];dot+=direction[c]*value[c];}
   #pragma unroll
   for(int c=0;c<6;c++)out[c*n+k]=ac*value[c]+bc*direction[c]*dot+(c<3?(K-ac/3)*tr:0);
 }
}
'''
_TANGENTS = {}
_COORDINATES = {}


def tangent_registers(material, value, response):
    key = value.dtype.str
    if key not in _TANGENTS:
        code = _TANGENT.replace('REAL', 'float' if value.dtype == cp.float32 else 'double')
        _TANGENTS[key] = cp.RawKernel(code, 'tangent_registers')
    out = cp.empty_like(value)
    _TANGENTS[key](((material.n+255)//256,), (256,),
        (value, response.direction, response.a, response.b, material.ids, material.table,
         out, np.int64(material.n), material.dtype(material.material.K)))
    return out


def coordinate_specialized(Q, x, adjoint):
    root = isinstance(Q, linear.SymmetricCoordinates)
    code = coords._CODE
    if root:
        code = code.replace('int adjoint)', 'int adjoint, REAL transverse, REAL longitudinal)')
        code = code.replace('v[c]=r2*v[c]+(1-r2)*k[c]*dot;',
            'v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;')
    key = (root, Q.shape, x.dtype.str, bool(adjoint))
    if key not in _COORDINATES:
        # Keep the public launch signature. Compile away repeated integer
        # divisions and direction branches, without approximating wavevectors.
        for arg in ('n0', 'n1', 'n2', 'nz'):
            code = code.replace('long long '+arg, 'long long unused_'+arg, 1)
        code = code.replace('int adjoint', 'int unused_adjoint', 1)
        definitions = '\n'.join(f'constexpr long long {arg}={value};' for arg,value in
            zip(('n0','n1','n2','nz'), (*Q.shape,Q.hshape[-1])))
        code = code.replace(' {', ' {\n'+definitions+f'\nconstexpr int adjoint={int(adjoint)};', 1)
        code = code.replace('COMPLEX', 'float2' if x.dtype == cp.complex64 else 'double2')
        name = 'float' if x.dtype == cp.complex64 else 'double'
        code = code.replace('REAL', name).replace('MAKE', 'make_'+name+'2')
        _COORDINATES[key] = cp.RawKernel(code, 'coordinates')
    d = x.real.dtype.type
    out = cp.empty(Q.size if adjoint else (6,)+Q.hshape, dtype=x.dtype)
    args = (x, out, *(np.int64(n) for n in Q.shape), np.int64(Q.hshape[-1]),
        *(d(1/L) for L in Q.P.lengths), d(Q.norm),
        np.int32(sum(1<<i for i in Q.P.free)), np.int32(adjoint))
    if root:
        args += d(Q.transverse), d(Q.longitudinal)
    _COORDINATES[key](((Q.count+255)//256,), (256,), args)
    return out


@contextmanager
def installed(*, tangent=False, coordinates=False, split_fft=False):
    with ExitStack() as stack:
        if tangent:
            stack.enter_context(patch.object(CuPyMaterial, 'tangent', tangent_registers))
        if coordinates:
            stack.enter_context(patch.object(coords.DisplacementCoordinates, '_map', coordinate_specialized))
            stack.enter_context(patch.object(linear.SymmetricCoordinates, '_map', coordinate_specialized))
        if split_fft:
            stack.enter_context(patch.object(fft, 'rfftn_cached', split_rfft))
            stack.enter_context(patch.object(fft, 'irfftn_unscaled_cached', split_irfft))
        yield
