"""CG reductions: warp shuffles, FP64 sums, device scalars.

Same interface as CupyCGReduction. No mesh/operator/tolerance changes. The
grid-stride kernels bound the partial buffer and avoid global atomic sums.
Changing reduction order is not bitwise equivalent; audit the true residual.
"""
import operator
import numpy as np

from cufhom_core.general.cupy_cg_reduction import CupyCGReduction
from cufhom_core.tensors.fft import cp

_KERNELS = {}
_SM_COUNT = {}


def reduction_blocks(total, threads=256):
    """Grid of a grid-stride FP64 reduction: eight blocks per SM at most.

    The FP64 tree costs as much as the loads on consumer GPUs (1/64-rate
    FP64), so more blocks than the SMs can hold only add partial sums.
    """
    device = cp.cuda.runtime.getDevice()
    if device not in _SM_COUNT:
        _SM_COUNT[device] = cp.cuda.runtime.getDeviceProperties(device)['multiProcessorCount']
    return max(1, min(8*_SM_COUNT[device], (int(total)+threads-1)//threads))


class CupyWarpCGReduction(CupyCGReduction):
    def __init__(self, tensor, max_blocks=None):
        super().__init__(tensor)
        if max_blocks is None:
            self.blocks = reduction_blocks(self.total, self.threads)
        else:
            max_blocks = operator.index(max_blocks)
            if max_blocks < 1:
                raise ValueError('max_blocks must be positive')
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


_COMPONENT_SUM = {}
_COMPONENT_SUM_CODE = r"""
__device__ __forceinline__ double warp_sum(double x) {
    for (int shift=16; shift>0; shift>>=1) x += __shfl_down_sync(0xffffffff, x, shift);
    return x;
}
extern "C" __global__
void component_sum(const TYPE* __restrict__ x, double* __restrict__ partial,
                   const long long n, const long long pitch) {
    // blockIdx.y selects the component; grid-stride FP64 sum, fixed tree.
    const TYPE* xc = x + (long long)blockIdx.y*pitch;
    double sum = 0;
    for (long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x; i<n;
         i+=(long long)blockDim.x*gridDim.x)
        sum += (double)xc[i];
    __shared__ double warp_values[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    sum = warp_sum(sum);
    if (lane==0) warp_values[warp] = sum;
    __syncthreads();
    if (warp==0) {
        sum = lane<8 ? warp_values[lane] : 0.0;
        sum = warp_sum(sum);
        if (lane==0) partial[(long long)blockIdx.y*gridDim.x+blockIdx.x] = sum;
    }
}
"""


def component_means(values):
    """FP64 means of the rows of a C-contiguous (C, n) FP32/FP64 CuPy array (host float64).

    CuPy reductions with a dtype conversion and one output run on a single
    block (5 GB/s); this grid-stride kernel reads all components in one launch
    at memory speed, with a fixed reduction tree (deterministic).
    """
    components, n = (int(v) for v in values.shape)
    key = values.dtype.str
    if key not in _COMPONENT_SUM:
        ctype = 'float' if values.dtype == np.dtype('float32') else 'double'
        _COMPONENT_SUM[key] = cp.RawKernel(_COMPONENT_SUM_CODE.replace('TYPE', ctype), 'component_sum')
    blocks = max(1, reduction_blocks(n) // max(1, components))
    partial = cp.empty((components, blocks), dtype=cp.float64)
    _COMPONENT_SUM[key]((blocks, components), (256,), (values, partial, np.int64(n), np.int64(n)))
    return cp.asnumpy(partial.sum(axis=1)) / n
