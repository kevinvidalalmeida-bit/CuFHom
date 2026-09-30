"""Temporary A/B linear kernel trials; patches restored on every exit."""
from contextlib import contextmanager
import re
from unittest.mock import patch

import numpy as np
from cufhom.pipeline import linear_coordinates as linear
from cufhom.pipeline import linear_krylov
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_coordinates import _CODE
from cufhom_core.tensors import fft
from profile_g09_linear import cp, parser, run, VERIFIED_SETTINGS


def source(shape, adjoint, specialized, fused, direction=False):
    code = _CODE.replace('int adjoint)', 'int adjoint, REAL transverse, REAL longitudinal)')
    code = code.replace('v[c]=r2*v[c]+(1-r2)*k[c]*dot;',
        'v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;')
    if direction:
        code = code.replace('const COMPLEX* x', 'COMPLEX* x')
        code = code.replace('REAL longitudinal) {',
                            'REAL longitudinal, const COMPLEX* residual, REAL beta) {')
        code = code.replace('if(s>=count)return;', '''if(s>=count)return;
        for(int c=0;c<3;c++) {
          long long index=c*count+s;
          x[index]=MAKE(residual[index].x+beta*x[index].x,
                        residual[index].y+beta*x[index].y);
        }''')
    if fused:
        code = code.replace('extern "C" __global__ void coordinates',
                            '__device__ __forceinline__ void coordinate_one')
        code = code.replace('REAL longitudinal) {',
                            'REAL longitudinal, const COMPLEX* p, double& energy) {')
        code = code.replace('for(int c=0;c<(adjoint?3:6);c++)y[c*count+s]=out[c];', '''
        for(int c=0;c<3;c++) {
          const long long index=c*count+s;
          y[index]=out[c];
          energy+=(double)p[index].x*out[c].x+(double)p[index].y*out[c].y;
        }''')
    if specialized:
        signature, body = code.split(') {', 1)
        for token, value in dict(n0=shape[0], n1=shape[1], n2=shape[2],
                                 nz=shape[2]//2+1, adjoint=int(adjoint), free_mask=0).items():
            body = re.sub(r'\b'+token+r'\b', str(value), body)
        code = signature+') {'+body
    if fused:
        code += r'''
extern "C" __global__ void adjoint_dot(const COMPLEX* x, COMPLEX* y,
 long long n0,long long n1,long long n2,long long nz,
 REAL iy0,REAL iy1,REAL iy2,REAL norm,int free_mask,int adjoint,
 REAL transverse,REAL longitudinal,const COMPLEX* p,double* partial) {
 double energy=0;
 coordinate_one(x,y,n0,n1,n2,nz,iy0,iy1,iy2,norm,free_mask,1,transverse,longitudinal,p,energy);
 for(int offset=16;offset;offset>>=1)energy+=__shfl_down_sync(0xffffffff,energy,offset);
 __shared__ double warps[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)warps[warp]=energy;
 __syncthreads();
 if(warp==0) {
   energy=lane<8?warps[lane]:0;
   for(int offset=16;offset;offset>>=1)energy+=__shfl_down_sync(0xffffffff,energy,offset);
   if(lane==0)partial[blockIdx.x]=energy;
 }
}
'''
    return code.replace('COMPLEX', 'float2').replace('REAL', 'float').replace('MAKE', 'make_float2')


KERNELS = {}


class TrialCoordinates(SymmetricCoordinates):
    specialized = False
    fused = False

    def evaluate(self, x, adjoint, p=None):
        out = cp.empty(self.size if adjoint else (6,)+self.hshape, dtype=x.dtype)
        direction = not adjoint and bool(PENDING)
        key = self.shape, bool(adjoint), self.specialized, p is not None, direction
        if key not in KERNELS:
            KERNELS[key] = cp.RawKernel(source(self.shape, adjoint, self.specialized, p is not None, direction),
                                       'adjoint_dot' if p is not None else 'coordinates')
        d = x.real.dtype.type
        args = (x, out, *(np.int64(n) for n in self.shape), np.int64(self.hshape[-1]),
                *(d(1/L) for L in self.P.lengths), d(self.norm), np.int32(0), np.int32(adjoint),
                d(self.transverse), d(self.longitudinal))
        blocks = (self.count+255)//256
        if direction:
            r, beta, target = PENDING
            assert target is x
            args += r, np.float32(beta)
            PENDING.clear()
        if p is not None:
            partial = cp.empty(blocks, cp.float64)
            args += p, partial
        KERNELS[key]((blocks,), (256,), args)
        return (out, float(cp.sum(partial, dtype=cp.float64))) if p is not None else out

    def _map(self, x, adjoint):
        return self.evaluate(x, adjoint)

    def strain(self, p):
        self.last_p = p
        return super().strain(p)

    def adjoint(self, field):
        if not self.fused or not hasattr(self, 'last_p'):
            return super().adjoint(field)
        h = fft.rfftn_cached(field, self.shape, (1, 2, 3))
        result, scalar = self.evaluate(h, True, self.last_p)
        CACHE[:] = [self.last_p, result, scalar]
        del self.last_p
        return result


CACHE = []
PENDING = []


@contextmanager
def trial(specialized=False, fused=False, direction=False):
    original = linear.fused_cg
    updates = linear_krylov._cg_updates()
    def deferred(r, beta, p):
        assert not PENDING
        PENDING[:] = [r, beta, p]
    def cg(action, rhs, dot, *a, **kw):
        def fused_dot(x, y):
            if len(CACHE) == 3 and x is CACHE[0] and y is CACHE[1]:
                value = CACHE[2]
                CACHE.clear()
                return value
            return dot(x, y)
        return original(action, rhs, fused_dot, *a, **kw)
    try:
        with patch.object(TrialCoordinates, 'specialized', specialized), \
             patch.object(TrialCoordinates, 'fused', fused), \
             patch.object(linear, 'SymmetricCoordinates', TrialCoordinates), \
             patch.object(linear_krylov, '_cg_updates', lambda: (updates[0], deferred if direction else updates[1])), \
             patch.object(linear, 'fused_cg', cg):
            yield
    finally:
        CACHE.clear()
        PENDING.clear()


if __name__ == '__main__':
    from contextlib import nullcontext
    args = parser().parse_args()
    run(args, dict(baseline=(VERIFIED_SETTINGS, nullcontext),
        fused_dot=(VERIFIED_SETTINGS, lambda: trial(fused=True)),
        direction=(VERIFIED_SETTINGS, lambda: trial(direction=True)),
        both=(VERIFIED_SETTINGS, lambda: trial(fused=True, direction=True))))
