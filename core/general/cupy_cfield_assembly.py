"""Effective-matrix rows without six material-result buffers.

The indexed material action retains FP32/FP64; contraction products and sums use
FP64, as in the existing fused assembly. Every matrix entry is computed.
"""
import numpy as np
import re
from cufhom_core.general.cupy_warp_reduction import reduction_blocks
from cufhom_core.tensors.fft import cp

_KERNELS = {}
_CODE = r'''
extern "C" __global__ void indexed_assembly_row(const int* ids,const float* table,
 const float* s0,const float* s1,const float* s2,
 const float* s3,const float* s4,const float* s5,long long count,long long stride,long long materials,double* partial) {
 const float* fields[6]={s0,s1,s2,s3,s4,s5};
 const float* x=fields[blockIdx.y];
 double sums[6]={0,0,0,0,0,0};
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;
     s+=(long long)gridDim.x*blockDim.x) {
   const int id=ids[s];
   if(id<0||id>=materials) {
     for(int j=0;j<6;j++)sums[j]=nan("");
     continue;
   }
   const float* a=table+21*(long long)id;
   float u[6],v[6];
   for(int c=0;c<6;c++)u[c]=x[c*stride+s];
   v[0]=a[0]*u[0]+a[1]*u[1]+a[2]*u[2]+a[3]*u[3]+a[4]*u[4]+a[5]*u[5];
   v[1]=a[1]*u[0]+a[6]*u[1]+a[7]*u[2]+a[8]*u[3]+a[9]*u[4]+a[10]*u[5];
   v[2]=a[2]*u[0]+a[7]*u[1]+a[11]*u[2]+a[12]*u[3]+a[13]*u[4]+a[14]*u[5];
   v[3]=a[3]*u[0]+a[8]*u[1]+a[12]*u[2]+a[15]*u[3]+a[16]*u[4]+a[17]*u[5];
   v[4]=a[4]*u[0]+a[9]*u[1]+a[13]*u[2]+a[16]*u[3]+a[18]*u[4]+a[19]*u[5];
   v[5]=a[5]*u[0]+a[10]*u[1]+a[14]*u[2]+a[17]*u[3]+a[19]*u[4]+a[20]*u[5];
   for(int j=0;j<6;j++) {
     for(int c=0;c<6;c++)sums[j]+=(double)v[c]*(double)fields[j][c*stride+s];
   }
 }
 __shared__ double warps[6][8];
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int j=0;j<6;j++) {
   for(int offset=16;offset;offset>>=1)sums[j]+=__shfl_down_sync(0xffffffff,sums[j],offset);
   if(lane==0)warps[j][warp]=sums[j];
 }
 __syncthreads();
 if(warp==0) {
   for(int j=0;j<6;j++) {
     double value=lane<8?warps[j][lane]:0;
     for(int offset=16;offset;offset>>=1)value+=__shfl_down_sync(0xffffffff,value,offset);
     if(lane==0)partial[(blockIdx.y*6+j)*gridDim.x+blockIdx.x]=value;
   }
 }
}
'''


def _kernel(dtype):
    key=np.dtype(dtype).str
    if key not in _KERNELS:
        code=_CODE if np.dtype(dtype)==np.dtype('float32') else re.sub(r'\bfloat\b','double',_CODE)
        _KERNELS[key]=cp.RawKernel(code,'indexed_assembly_row')
    return _KERNELS[key]


def indexed_assembly(A, fields):
    """Return a device FP64 matrix, or None outside the raw CUDA contract."""
    from cufhom_core.tensors import Tensor
    # Do not bypass a custom operator/subclass's __call__ implementation.
    if cp is None or type(A) is not Tensor or len(fields) != 6:
        return None
    if A.multype not in ('sym21_indexed', 'indexed_sym21') or A.Fourier:
        return None
    ids, table = A.val, getattr(A, 'material_table', None)
    shape = tuple(int(n) for n in A.N)
    if (not isinstance(ids, cp.ndarray) or not isinstance(table, cp.ndarray)
            or ids.dtype != cp.int32 or table.dtype not in (cp.float32,cp.float64)
            or ids.shape not in (shape, (1,)+shape) or table.ndim != 2
            or table.shape[0] == 0 or table.shape[1] != 21
            or not ids.flags.c_contiguous or not table.flags.c_contiguous):
        return None
    for field in fields:
        if type(field) is not Tensor:
            return None
        value = field.val
        if (not isinstance(value, cp.ndarray) or value.dtype != table.dtype
                or value.shape != (6,)+shape or not value.flags.c_contiguous
                or field.Fourier or field.fft_form != A.fft_form
                or not np.array_equal(field.N, A.N)):
            return None
    kernel=_kernel(table.dtype)
    count = int(np.prod(shape))
    blocks = reduction_blocks(count)
    partial = cp.empty((6, 6, blocks), cp.float64)
    values = tuple(field.val for field in fields)
    # All six rows share one launch and one reduction. Each row retains the
    # same grid-stride products and FP64 tree; no entries are mirrored.
    kernel((blocks, 6), (256,), (ids, table, *values, np.int64(count), np.int64(count),
                               np.int64(table.shape[0]), partial))
    return cp.sum(partial, axis=2, dtype=cp.float64)


def supports_streamed_assembly(A, fields):
    """The indexed FP32/FP64 contract, allowing CPU or device strain fields."""
    from cufhom_core.tensors import Tensor
    if cp is None or type(A) is not Tensor or len(fields) != 6:
        return False
    if A.multype not in ('sym21_indexed', 'indexed_sym21') or A.Fourier:
        return False
    ids, table = A.val, getattr(A, 'material_table', None)
    shape = tuple(int(n) for n in A.N)
    if (not isinstance(ids, cp.ndarray) or not isinstance(table, cp.ndarray)
            or ids.dtype != cp.int32 or table.dtype not in (cp.float32,cp.float64)
            or ids.shape not in (shape, (1,)+shape) or table.ndim != 2
            or table.shape[0] == 0 or table.shape[1] != 21
            or not ids.flags.c_contiguous or not table.flags.c_contiguous):
        return False
    for field in fields:
        if (type(field) is not Tensor or field.Fourier or field.fft_form != A.fft_form
                or not np.array_equal(field.N, A.N)):
            return False
        value = field.val
        if value is None and hasattr(field, 'cufhom_coordinates'):
            # The application materializes these coordinates one load at a time
            # before invoking the assembly, never inside a voxel chunk.
            continue
        if (not isinstance(value, (np.ndarray, cp.ndarray)) or value.dtype != table.dtype
                or value.shape != (6,)+shape or not value.flags.c_contiguous):
            return False
    return True


def indexed_assembly_streamed(A, fields, chunk_voxels=1_048_576):
    """Same-precision material / FP64 energy with bounded GPU staging.

    Fields may be pinned NumPy arrays or device arrays. Two staging blocks
    of at most 144 MiB each overlap strided DMA and contraction. The last,
    shorter interval keeps the allocated component stride explicitly. No
    symmetry mirroring, stress buffers or reduced-precision copies are used.
    """
    import operator
    if not supports_streamed_assembly(A, fields):
        return None
    chunk_voxels = operator.index(chunk_voxels)
    if chunk_voxels < 1:
        raise ValueError('chunk_voxels must be positive')
    if any(field.val is None for field in fields):
        raise ValueError('Reconstruct coordinate solutions before streamed assembly')
    dtype=A.material_table.dtype
    itemsize=dtype.itemsize
    kernel=_kernel(dtype)
    count = int(np.prod(A.N))
    capacity = min(count, chunk_voxels)
    staging = [cp.empty((6, 6, capacity), dtype) for _ in range(2)]
    values = [field.val.reshape(6, count) for field in fields]
    means = np.asarray([getattr(field, 'cufhom_deferred_mean', np.zeros(6, dtype))
                        for field in fields], dtype=dtype)
    if means.shape != (6, 6) or not np.isfinite(means).all():
        raise ValueError('Invalid deferred Mandel means')
    device_means = cp.asarray(means[:, :, None]) if np.any(means) else None
    blocks = reduction_blocks(capacity)
    partial = cp.empty((6, 6, blocks), cp.float64)
    matrix = cp.zeros((6, 6), cp.float64)
    ids = A.val.reshape(-1)
    compute = cp.cuda.get_current_stream()
    transfer = cp.cuda.Stream(non_blocking=True)
    inputs_ready = cp.cuda.Event(disable_timing=True)
    inputs_ready.record(compute)
    transfer.wait_event(inputs_ready)
    ready = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
    consumed = [cp.cuda.Event(disable_timing=True) for _ in range(2)]
    used = [False, False]
    for chunk, start in enumerate(range(0, count, capacity)):
        slot = chunk % 2
        if used[slot]:
            # Wait only when reusing a device slot. Source fields stay alive
            # and unchanged until the last consumer below has completed.
            consumed[slot].synchronize()
        width = min(capacity, count-start)
        with transfer:
            for load, value in enumerate(values):
                if isinstance(value, np.ndarray):
                    # Six component planes have pitch N on host and capacity
                    # on device. Copy them directly from pinned input fields;
                    # gathering through another host buffer doubles RAM traffic.
                    cp.cuda.runtime.memcpy2DAsync(
                        staging[slot][load].data.ptr, capacity*itemsize,
                        value.ctypes.data+start*itemsize, count*itemsize, width*itemsize, 6,
                        cp.cuda.runtime.memcpyHostToDevice, transfer.ptr)
                else:
                    staging[slot][load, :, :width] = value[:, start:start+width]
            ready[slot].record(transfer)
        compute.wait_event(ready[slot])
        if device_means is not None:
            staging[slot] += device_means
        kernel((blocks, 6), (256,),
                (ids[start:], A.material_table, *tuple(staging[slot][j] for j in range(6)),
                 np.int64(width), np.int64(capacity), np.int64(A.material_table.shape[0]), partial))
        matrix += cp.sum(partial, axis=2, dtype=cp.float64)
        consumed[slot].record(compute)
        used[slot] = True
    # Retain all source fields and both streams until the last consumer ends.
    compute.synchronize()
    return matrix
