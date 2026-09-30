"""Isolated bandwidth experiment: Q* map and p.Ap fused, FP64 reductions.

No solver integration/default changes. Borrowed output, overwritten on reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from profile_g09_fft import cp, ROOT
from benchmark_cuda_graph_operator import sample
from pipeline.nonlinear_coordinates import DisplacementCoordinates, _CODE
from pipeline.nonlinear_solver import MixedProjector
from ffthompy.tensors import fft
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction


def kernel_code(dtype):
    # Preserve the coordinate formula. Early returns are confined to the
    # device function; ALL threads reach the enclosing reduction/barrier.
    code = _CODE.replace('extern "C" __global__ void coordinates',
                         '__device__ __forceinline__ void coordinate_one')
    code = code.replace('int adjoint) {',
        'int adjoint, const COMPLEX* p, double& energy) {')
    old = 'for(int c=0;c<6;c++)if(free_mask&(1<<c))y[3*count+j++]=MAKE(norm*x[c*count].x,0);'
    new = '''for(int c=0;c<6;c++)if(free_mask&(1<<c)) {
       const long long index=3*count+j++;
       const REAL value=norm*x[c*count].x;
       y[index]=MAKE(value,0);
       energy+=(double)p[index].x*value;
     }'''
    if old not in code:
        raise RuntimeError('Coordinate mean implementation changed; re-audit the experiment')
    code = code.replace(old, new)
    old = 'for(int c=0;c<(adjoint?3:6);c++)y[c*count+s]=out[c];'
    new = '''for(int c=0;c<3;c++) {
       const long long index=c*count+s;
       y[index]=out[c];
       energy+=(double)p[index].x*out[c].x+(double)p[index].y*out[c].y;
     }'''
    if old not in code:
        raise RuntimeError('Coordinate output changed; re-audit the experiment')
    code = code.replace(old, new)
    code += r'''
extern "C" __global__ void adjoint_dot(const COMPLEX* x, COMPLEX* y,
 long long n0,long long n1,long long n2,long long nz,
 REAL iy0,REAL iy1,REAL iy2,REAL norm,int free_mask,
 const COMPLEX* p,double* partial) {
 double energy=0;
 coordinate_one(x,y,n0,n1,n2,nz,iy0,iy1,iy2,norm,free_mask,1,p,energy);
 for(int offset=16;offset;offset>>=1)
   energy+=__shfl_down_sync(0xffffffff,energy,offset);
 __shared__ double warps[8];
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)warps[warp]=energy;
 __syncthreads();
 if(warp==0) {
   energy=lane<8?warps[lane]:0;
   for(int offset=16;offset;offset>>=1)
     energy+=__shfl_down_sync(0xffffffff,energy,offset);
   if(lane==0)partial[blockIdx.x]=energy;
 }
}
'''
    name = 'float' if np.dtype(dtype) == np.dtype('complex64') else 'double'
    return code.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')


class FusedAdjointDot:
    def __init__(self, Q, dtype):
        if type(Q) is not DisplacementCoordinates or Q.xp is not cp:
            raise ValueError('Diagnostic supports GPU displacement coordinates only')
        if np.dtype(dtype) not in (np.dtype('complex64'), np.dtype('complex128')):
            raise ValueError('Expected complex64 or complex128')
        self.Q = Q
        self.output = cp.empty(Q.size, dtype=dtype)
        self.blocks = (Q.count+255)//256
        self.partial = cp.empty(self.blocks, dtype=cp.float64)
        self.kernel = cp.RawKernel(kernel_code(dtype), 'adjoint_dot')
        d = self.output.real.dtype.type
        self.geometry = (*(np.int64(n) for n in Q.shape), np.int64(Q.hshape[-1]),
                         *(d(1/L) for L in Q.P.lengths), d(Q.norm),
                         np.int32(sum(1<<i for i in Q.P.free)))

    def __call__(self, h, p):
        if (h.shape != (6,)+self.Q.hshape or p.shape != self.output.shape
                or h.dtype != self.output.dtype or p.dtype != self.output.dtype
                or not h.flags.c_contiguous or not p.flags.c_contiguous):
            raise ValueError('Mismatched shape/precision or non-contiguous input')
        self.kernel((self.blocks,), (256,),
                    (h, self.output, *self.geometry, p, self.partial))
        return self.output, cp.sum(self.partial, dtype=cp.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--inner', type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.inner < 1:
        parser.error('Use a new output file and positive repetition counts')
    shape = (240,240,240)
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    fft.set_cupy_rfft_batch_mode('component')
    P = MixedProjector(shape, (1.,1.,1.), [True,False,False,False,False,False], cp)
    Q = DisplacementCoordinates(P)
    field = cp.random.RandomState(704).standard_normal((6,)+shape, dtype=cp.float32)
    h = fft.rfftn_cached(field, shape, (1,2,3))
    # A compatible test p with guaranteed positive inner product. No material
    # solve in this stage test; testing arbitrary signs is done in unit tests.
    p = Q._map(h, True)
    reduction = CupyWarpCGReduction(SimpleNamespace(N=shape, val=field), max_blocks=16384)
    del field
    def original():
        out = Q._map(h, True)
        reduction.dot_kernel((reduction.blocks,), (256,),
            (p, out, reduction.partial, np.int64(p.size*2)))
        return out, cp.sum(reduction.partial, dtype=cp.float64)
    fused = FusedAdjointDot(Q, p.dtype)
    expected, scalar = original()
    actual, candidate_scalar = fused(h, p)
    equal = bool(cp.array_equal(expected, actual))
    scalar_error = abs(float(scalar)-float(candidate_scalar))/abs(float(scalar))
    if not equal or not np.isfinite(scalar_error) or scalar_error > 5e-13:
        raise RuntimeError('Fused map/product mismatch')
    variants = dict(original=original, fused=lambda: fused(h, p),
        original_with_host_scalar=lambda: float(original()[1]),
        fused_with_host_scalar=lambda: float(fused(h,p)[1]))
    for fn in variants.values():
        fn()
    rows = {name: [] for name in variants}; names = list(variants)
    for rep in range(args.repeats):
        order = names[rep % len(names):]+names[:rep % len(names)]
        for name in order:
            rows[name].append(sample(variants[name], args.inner))
    report = dict(scope='240^3 Q* map + p.Ap stage ONLY, synthetic fields, original '
        'FP64 warp dot and tolerance. Not a CG/full-curve gain. Fused output borrowed.',
        output_bitwise_equal=equal, scalar_relative_error=scalar_error,
        bytes_of_output_read_avoided_per_action=int(p.nbytes), seed=704,
        dtype=str(p.dtype), shape=list(shape), repeats=args.repeats, inner=args.inner,
        device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(), cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        hashes={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), ROOT/'FFT/pipeline/nonlinear_coordinates.py',
                             ROOT/'FFT/benchmarks/benchmark_cuda_graph_operator.py')},
        timings={name: dict(median_event_ms=float(np.median([s['event_ms'] for s in samples])),
                           median_wall_ms=float(np.median([s['wall_ms'] for s in samples])),
                           samples=samples) for name, samples in rows.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
