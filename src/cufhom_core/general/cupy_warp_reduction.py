"""Experimental CG reductions: warp shuffles, FP64 sums, device scalars.

Same interface as CupyCGReduction. No mesh/operator/tolerance changes. The
grid-stride kernels bound the partial buffer and avoid global atomic sums.
Changing reduction order is not bitwise equivalent; audit the true residual.
"""
import operator
import numpy as np

from cufhom_core.general.cupy_cg_reduction import CupyCGReduction
from cufhom_core.tensors.fft import cp

_KERNELS = {}


class CupyWarpCGReduction(CupyCGReduction):
    def __init__(self, tensor, max_blocks=4096):
        max_blocks = operator.index(max_blocks)
        if max_blocks < 1:
            raise ValueError('max_blocks must be positive')
        super().__init__(tensor)
        self.blocks = min(max_blocks, (self.total+self.threads-1)//self.threads)
        self.partial = cp.empty(self.blocks, dtype=cp.float64)
        key = self.dtype.str
        if key not in _KERNELS:
            ctype = 'float' if self.dtype == np.dtype('float32') else 'double'
            code = r'''
            __device__ __forceinline__ double warp_sum(double x) {
                for (int shift=16; shift>0; shift>>=1)
                    x += __shfl_down_sync(0xffffffff, x, shift);
                return x;
            }
            __device__ __forceinline__ void finish(double x, double* partial) {
                __shared__ double warp_values[8];
                const int lane = threadIdx.x & 31;
                const int warp = threadIdx.x >> 5;
                x = warp_sum(x);
                if (lane==0) warp_values[warp] = x;
                __syncthreads();
                if (warp==0) {
                    x = lane<8 ? warp_values[lane] : 0.0;
                    x = warp_sum(x);
                    if (lane==0) partial[blockIdx.x] = x;
                }
            }
            extern "C" __global__
            void warp_dot(const TYPE* __restrict__ x,
                          const TYPE* __restrict__ y,
                          double* __restrict__ partial, const long long total) {
                double sum = 0;
                for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                     i<total; i+=(long long)blockDim.x*gridDim.x)
                    sum += (double)x[i]*(double)y[i];
                finish(sum, partial);
            }
            extern "C" __global__
            void warp_update(const TYPE* alpha,
                             const TYPE* __restrict__ p,
                             const TYPE* __restrict__ ap,
                             TYPE* __restrict__ x, TYPE* __restrict__ r,
                             double* __restrict__ partial, const long long total) {
                const TYPE a = alpha[0];
                double sum = 0;
                for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
                     i<total; i+=(long long)blockDim.x*gridDim.x) {
                    x[i] += a*p[i];
                    const TYPE ri = r[i]-a*ap[i];
                    r[i] = ri;
                    sum += (double)ri*(double)ri;
                }
                finish(sum, partial);
            }
            '''.replace('TYPE', ctype)
            _KERNELS[key] = (cp.RawKernel(code, 'warp_dot'), cp.RawKernel(code, 'warp_update'))
        self.dot_kernel, self.update_kernel = _KERNELS[key]
