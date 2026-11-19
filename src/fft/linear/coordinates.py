"""Displacement-coordinate PCG for the linear FFT problem (FP32/FP64 GPU).

Uses the same Q mapping as the nonlinear solver, with all mean fluctuations
fixed to zero. No frequencies, voxels or material coefficients are removed.
CG runs in whitened compatible coordinates; a candidate is accepted only when
the residual of the original strain equation meets the tolerance.
"""

import os
import re

import numpy as np
from cufhom.fft.linear.energy import (
    energy_action,
    energy_update_device,
    supports_energy,
)
from cufhom.fft.linear.fused import (
    FusedPipeline,
    FusedStep,
    defer_compile,
    fused_supported,
)
from cufhom.fft.linear.krylov import (
    SymmetricCoordinates,
    coordinate_source,
    fused_cg,
    replace_once,
    tiled_coordinate_source,
)
from cufhom.fft.operators import MixedProjector, hermitian_plane
from cufhom_core.general.cupy_capacity import library_capacity_required
from cufhom_core.general.cupy_warp_reduction import (
    CupyWarpCGReduction,
    reduction_blocks,
)
from cufhom_core.general.solver import _finalize_cg_info
from cufhom_core.tensors import fft as core_fft
from cufhom_core.tensors.fft import cp
from cufhom_core.tensors.objects import elasticity_direct_array

from .common import (
    LinearSolveError,
    energy_tolerance,
    stored_planes,
    working_material_check,
)
from .precision import cuda_source

_SQUARED_DISTANCE = r"""
extern "C" __global__ void squared_distance(const float* a,const float* b,double* partial,long long n) {
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x) {
   const float d=a[i]-b[i];
   sum+=(double)d*(double)d;
 }
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double sums[8];
 if((threadIdx.x&31)==0)sums[threadIdx.x>>5]=sum;
 __syncthreads();
 if(threadIdx.x<32) {
   sum=threadIdx.x<8?sums[threadIdx.x]:0;
   for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
   if(threadIdx.x==0)partial[blockIdx.x]=sum;
 }
}
"""
_KERNEL_CACHE = {}

_SPECTRAL_DISTANCE = r"""
extern "C" __global__ void spectral_distance(const float2* a,const float2* b,
 double* partial,long long n,long long nz,int even,int subtract,long long offset) {
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x) {
   const float2 ai=a[i],bi=subtract?b[i]:make_float2(0,0);
   const float re=ai.x-bi.x,im=ai.y-bi.y;
   const long long k2=(i+offset)%nz;
   const double weight=(k2==0||(even&&k2==nz-1))?1:2;
   sum+=weight*((double)re*re+(double)im*im);
 }
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double sums[8];
 if((threadIdx.x&31)==0)sums[threadIdx.x>>5]=sum;
 __syncthreads();
 if(threadIdx.x<32) {
   sum=threadIdx.x<8?sums[threadIdx.x]:0;
   for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
   if(threadIdx.x==0)partial[blockIdx.x]=sum;
 }
}
"""


def _spectral_distance(a, shape, b=None):
    """Squared physical RMS of raw-C2R(a-b), using RFFT Parseval weights.

    The core direct projection uses raw inverses, so its physical RMS squared
    is the weighted spectral energy (no extra volume normalization).
    """
    expected = (6,) + tuple(shape[:-1]) + (shape[-1] // 2 + 1,)
    if (
        a.dtype not in (cp.complex64, cp.complex128)
        or a.shape != expected
        or not a.flags.c_contiguous
        or (
            b is not None
            and (b.dtype != a.dtype or b.shape != a.shape or not b.flags.c_contiguous)
        )
    ):
        raise ValueError(
            "The spectral audit expects contiguous Mandel complex64/complex128 RFFT fields"
        )
    key = ("spectral_distance", a.dtype.str)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = cp.RawKernel(
            cuda_source(_SPECTRAL_DISTANCE, a.dtype), "spectral_distance"
        )
    if isinstance(a, np.ndarray):
        if b is None or not isinstance(b, cp.ndarray):
            raise ValueError("A host spectral RHS requires a device candidate")
        capacity = min(a.size, 4_194_304)
        staging = cp.empty(capacity, a.dtype)
        blocks = reduction_blocks(capacity)
        partial = cp.empty(blocks, cp.float64)
        total = cp.zeros((), cp.float64)
        host, candidate = a.reshape(-1), b.reshape(-1)
        for start in range(0, a.size, capacity):
            width = min(capacity, a.size - start)
            staging[:width].set(host[start : start + width])
            _KERNEL_CACHE[key](
                (blocks,),
                (256,),
                (
                    staging,
                    candidate[start:],
                    partial,
                    np.int64(width),
                    np.int64(a.shape[-1]),
                    np.int32(shape[-1] % 2 == 0),
                    np.int32(1),
                    np.int64(start),
                ),
            )
            total += cp.sum(partial, dtype=cp.float64)
        return float(total)
    blocks = reduction_blocks(a.size)
    partial = cp.empty(blocks, cp.float64)
    _KERNEL_CACHE[key](
        (blocks,),
        (256,),
        (
            a,
            a if b is None else b,
            partial,
            np.int64(a.size),
            np.int64(a.shape[-1]),
            np.int32(shape[-1] % 2 == 0),
            np.int32(b is not None),
            np.int64(0),
        ),
    )
    return float(cp.sum(partial, dtype=cp.float64))


_MIXED = r"""
__device__ __forceinline__ double mixed_reduce(double sum,double* partial){
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double sums[8];
 if((threadIdx.x&31)==0)sums[threadIdx.x>>5]=sum;
 __syncthreads();
 if(threadIdx.x<32){
  sum=threadIdx.x<8?sums[threadIdx.x]:0;
  for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
  if(threadIdx.x==0)partial[blockIdx.x]=sum;
 }
 return sum;
}
// Fold the FP32 increment into the FP64 coordinates; the increment restarts at zero.
extern "C" __global__ void mixed_flush(float2* x,double2* z,long long n){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  const float2 a=x[i];double2 b=z[i];b.x+=(double)a.x;b.y+=(double)a.y;z[i]=b;x[i]=make_float2(0.f,0.f);
 }
}
// Load stress spectrum of one macro strain from the phase indicator spectra:
// h[c] = sum_p coef[c][p] chi[p] (FP64, memory bound; P <= 6).
extern "C" __global__ void phase_load(const double2* chi,const double* coef,double2* h,long long m,int phases){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<m;i+=(long long)gridDim.x*blockDim.x){
  double2 v[6]={{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};
  for(int p=0;p<phases;p++){const double2 x=chi[p*m+i];
   for(int c=0;c<6;c++){v[c].x+=coef[c*phases+p]*x.x;v[c].y+=coef[c*phases+p]*x.y;}}
  for(int c=0;c<6;c++)h[c*m+i]=v[c];
 }
}
extern "C" __global__ void mixed_dot(const float2* a,const float2* b,double* partial,long long n){
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  const float2 u=a[i],v=b[i];sum+=(double)u.x*v.x+(double)u.y*v.y;
 }
 mixed_reduce(sum,partial);
}
extern "C" __global__ void mixed_dot64(const double2* a,const float2* b,double* partial,long long n){
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  const double2 u=a[i];const float2 v=b[i];sum+=(double)u.x*v.x+(double)u.y*v.y;
 }
 mixed_reduce(sum,partial);
}
extern "C" __global__ void mixed_dot128(const double2* a,const double2* b,double* partial,long long n){
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  const double2 u=a[i],v=b[i];sum+=u.x*v.x+u.y*v.y;
 }
 mixed_reduce(sum,partial);
}
"""
# Mixed precision: residual-norm drop between FP64 true-residual replacements
# (as HEX8: one FP64 operator per replacement, the search direction is kept).
# Both criteria schedule their target (the energy criterion its estimated
# r.r = tau N gamma_- E):
# FP32 segments of at most _FP32_SEGMENT drop (the recursion stalls near 3e-7
# of its start) with equal drops, so rtol 1e-6 needs no replacement before the
# candidate and 1e-8 one; the recursion growing _FP32_GROWTH times above its
# minimum (r.r) forces a replacement.
_FP32_SEGMENT = 1e-6
_FP32_GROWTH = 100.0
# Interleaved six-load measurements showed a repeatable dispatch benefit at
# 32^3, but none at 64^3 and larger. Keep the extra device-control launches
# out of the production-size loops instead of extrapolating the small case.
_DEVICE_PREFIX_MAX_VOXELS = 32**3


def _segment_drop(relative):
    """Equal FP32 segment drop reaching relative; 0 when one segment suffices."""
    segments = (
        int(np.ceil(np.log(relative) / np.log(_FP32_SEGMENT) - 1e-9))
        if 0 < relative < 1
        else 1
    )
    return 0.0 if segments <= 1 else relative ** (1 / segments) * 1.0001


def _mixed_kernel(name):
    key = ("mixed", name)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = cp.RawKernel(_MIXED, name)
    return _KERNEL_CACHE[key]


def _residual_map(Q, h, r, audit=False):
    """Apply the FP64 adjoint coordinate map, storing r = fl(-RQ* h).

    The output is rounded to the precision of r.

    The same arithmetic as the FP64 map followed by the rounding of its
    negation to complex64 (or none to complex128), without an FP64 coordinate
    vector in between. With audit, the
    same pass returns the squared physical audit ||G h||^2 from the FP64
    coordinates: RQ* is a Parseval-weighted isometry on compatible fields
    scaled by 1/R, so ||G h||^2 = norm^-2 sum (|r_perp|^2/t^2 + |r_par|^2/l^2)
    with t, l the transverse and longitudinal coordinate moduli (no
    projection or spectral-distance pass).
    """
    tiled = Q.tile is not None
    out = "float2" if r.dtype == cp.complex64 else "double2"
    key = ("residual_map", tiled, audit, out)
    if key not in _KERNEL_CACHE:
        code = (tiled_coordinate_source() if tiled else coordinate_source()).replace(
            "COMPLEX* y", "OUT2* y"
        )
        code = re.sub(r"\by\[([^\]]+)\]=([^;]+);", r"y[\1]=negate(\2);", code)
        code = (
            code.replace("COMPLEX", "double2")
            .replace("REAL", "double")
            .replace("MAKE", "make_double2")
        )
        code = code.replace("OUT2", out)
        code = (
            "__device__ float2 negate(double2 v){return make_float2((float)-v.x,(float)-v.y);}\n"
            if out == "float2"
            else "__device__ double2 negate(double2 v){return make_double2(-v.x,-v.y);}\n"
        ) + code
        if audit:
            # The kernel body becomes a per-frequency function in a grid-stride
            # loop that accumulates the audit and reduces it per block.
            store = (
                "y[adjoint?coordinate+c:c*count+s]=negate(out[c]);"
                if tiled
                else "y[c*count+s]=negate(out[c]);"
            )
            code = replace_once(
                code,
                store,
                store
                + r"""
 double a2=0,px=0,py=0;
 for(int c=0;c<3;c++){a2+=out[c].x*out[c].x+out[c].y*out[c].y;px+=k[c]*out[c].x;py+=k[c]*out[c].y;}
 const double p2=px*px+py*py;
 *acc+=((a2-p2)/(transverse*transverse)+p2/(longitudinal*longitudinal))/(norm*norm);""",
            )
            code = replace_once(
                code,
                'extern "C" __global__ void coordinates(',
                "__device__ void frequency(double* acc,long long s,",
            )
            code = replace_once(
                code,
                " const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;\n",
                "",
            )
            start = code.index("frequency(") + len("frequency(")
            params = code[start : code.index(") {", start)]
            names = ",".join(p.split()[-1].lstrip("*") for p in params.split(","))
            code = (
                _MIXED[: _MIXED.index("// Fold the FP32")]
                + code
                + r"""
extern "C" __global__ void coordinates(%s,double* partial){
 double acc=0;
 const long long frequencies=n0*n1*nz;
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<frequencies;s+=(long long)gridDim.x*blockDim.x)
  frequency(%s);
 mixed_reduce(acc,partial);
}
"""
                % (params.split(",", 2)[2], names.replace("acc,s,", "&acc,s,", 1))
            )
        _KERNEL_CACHE[key] = cp.RawKernel(code, "coordinates")
    args = Q._args(h, r, True) + ((np.int32(Q.tile),) if tiled else ())
    if not audit:
        _KERNEL_CACHE[key](((Q.count + 255) // 256,), (256,), args)
        return None
    blocks = reduction_blocks(Q.count)
    partial = cp.empty(blocks, cp.float64)
    _KERNEL_CACHE[key]((blocks,), (256,), args + (partial,))
    return float(cp.sum(partial, dtype=cp.float64))


def _evaluation_label(evaluator, counts):
    """Arithmetic of the evaluations after the load (the load: cuFFT FP64)."""
    after = counts["evaluations"] - 1
    if evaluator is None or counts["fused"] == 0:
        return "float64_library"
    return (
        evaluator.arithmetic
        if counts["fused"] == after
        else evaluator.arithmetic + "+float64_library"
    )


def _evaluation_pipeline(
    pb, Q, descriptor, arithmetic, pipeline, solution="float64", residual="float32"
):
    """Return the compatible fused evaluator for this material and precision.

    The fused evaluation pipeline of this grid, material, arithmetic and
    storage precisions (sharing the Krylov pipeline's compact ids), or None
    outside its layout.
    """
    from .evaluation import ARITHMETICS, EvaluationPipeline, evaluation_supported

    if arithmetic not in ARITHMETICS:
        raise ValueError(f"evaluation must be one of {ARITHMETICS}")
    key = (Q.shape, Q.tile, arithmetic, solution, residual)
    cached = pb.solver.get("_linear_evaluation")
    if cached is not None and cached[0] is descriptor and cached[1] == key:
        return cached[2]
    value = None
    if pipeline is not None and evaluation_supported(Q, descriptor):
        value = EvaluationPipeline(
            Q,
            descriptor,
            arithmetic,
            ids=pipeline.ids,
            solution=solution,
            residual=residual,
        )
    pb.solver["_linear_evaluation"] = (descriptor, key, value)
    return value


CONFIGURATIONS = ("mixed", "float32", "float64")
# 'device' keeps a load on the device; 'host' keeps the mixed FP64
# coordinates in pinned host memory; 'iterate' also keeps the Krylov iterate
# there (in the fixed configurations the iterate is the solution).
RESIDENCIES = ("device", "host", "iterate")
# Device memory reserved beside a load: the allocator's transient margin and
# the cached phase indicator spectra (kept below 6 % of the free memory).
_RESERVE_BYTES = 512 << 20
_INDICATOR_SHARE = 0.06


def mixed_working_bytes(shape, residency="device", configuration="mixed", energy=False):
    """Device bytes of one GaNi coordinate load at its peak phase.

    Beside the material ids (int32 and the pipeline's uint16), a load holds
    in turn its iteration state, the working-precision pipeline spectrum (two
    vectors) with the Krylov vectors r, p, x and, in the smoothing decade of
    the residual criterion, y, and its evaluation state, the padded FP64
    six-component work field (strain, stress and spectrum in place) with r,
    x and, at a mixed scheduled replacement on the device level, p (the host
    levels keep it in host memory meanwhile). The mixed configuration adds
    its FP64 coordinates or, from 'host' on, two staging slabs; 'iterate'
    replaces the device x and y by two staging slabs each (their resident
    rows, chosen per load, use only memory left over). FP32 maps its
    solution to the FP64 library evaluation through one staging slab.
    """
    if residency not in RESIDENCIES:
        raise ValueError(f"residency must be one of {RESIDENCIES}")
    n0, n1, n2 = (int(n) for n in shape)
    half = n0 * n1 * (n2 // 2 + 1)
    mixed, copies = configuration == "mixed", 1 + (not energy)
    vector = (16 if configuration == "float64" else 8) * 3 * half
    iterate = 0 if residency == "iterate" else vector
    staging = 2 * copies * _SLAB_BYTES if residency == "iterate" else 0
    if mixed:
        staging += 16 * 3 * half if residency == "device" else 2 * _SLAB_BYTES
    elif configuration == "float32":
        staging += min(_SLAB_BYTES, 16 * 3 * half)
    iteration = 4 * vector + copies * iterate
    evaluation = (
        96 * half + (2 if mixed and residency == "device" else 1) * vector + iterate
    )
    return max(iteration, evaluation) + staging + 6 * n0 * n1 * n2


def mixed_residency(
    shape, configuration="mixed", energy=False, phases=None, library=False
):
    """Select the first GaNi load residency that fits the device, or return None.

    The host levels stream their vectors in slabs of the fused pipeline's
    layout; a fixed configuration, whose solution is its Krylov iterate,
    goes from 'device' to 'iterate'. On grids of the fused layout the
    evaluations are fused, and a load uses cuFFT FP64 for its load only when
    the plans fit beside it (mixed_load); elsewhere, or with library (strain
    fields requested), every evaluation
    uses cuFFT FP64, whose plans are created before measuring (their work
    areas reach one FP64 field on some radices) and beside which the
    indicator spectra may be cached (phases: material rows; None reserves
    their largest share).
    """
    from cufhom_core.general.cupy_strain_capacity import plans
    from cupy.cuda import cufft

    from .common import device_free_bytes
    from .fused import grid_supported

    shape = tuple(int(n) for n in shape)
    if (
        mixed_working_bytes(shape, "iterate", configuration, energy)
        > device_free_bytes()
    ):
        return None
    fused = grid_supported(shape, np.float64)
    cache = 0
    if library or not fused:
        try:
            plans(shape, np.float64)
        except (cp.cuda.memory.OutOfMemoryError, cufft.CuFFTError):
            return None
        cache = _INDICATOR_SHARE * device_free_bytes()
        if phases is not None:
            n0, n1, n2 = shape
            spectra = 16 * (phases - 1) * n0 * n1 * (n2 // 2 + 1)
            cache = spectra if phases - 1 <= 6 and spectra < cache else 0
    free = device_free_bytes() - cache - _RESERVE_BYTES
    for level in RESIDENCIES if configuration == "mixed" else ("device", "iterate"):
        if level != "device" and not fused:
            return None
        if mixed_working_bytes(shape, level, configuration, energy) < free:
            return level
    return None


def _resident_rows(shape, configuration, smoothing, shared=False):
    """Estimate the number of host-iterate rows that can remain on the device.

    Second-axis rows of a host-resident Krylov iterate kept on the device:
    the most that fit beside the rest of the load (the pipeline spectrum and
    p, or the evaluation work field) and the staging slabs. The
    smoothed copy, alive only in the last decade, stays wholly on the host.
    Called once r and the solution's staging exist (shared: and the FP64
    workspace holding the spectrum and evaluation field).
    """
    from .common import device_free_bytes

    n0, n1, n2 = (int(n) for n in shape)
    item = 16 if configuration == "float64" else 8
    row = 3 * item * n0 * stored_planes(n2)
    vector = n1 * row
    free = device_free_bytes() - _RESERVE_BYTES - 2 * (1 + smoothing) * _SLAB_BYTES
    iteration = (free - (1 if shared else 3) * vector) / row
    evaluation = (free - (0 if shared else 96 * n0 * n1 * (n2 // 2 + 1))) / row
    return int(max(0, min(n1, iteration, evaluation)))


# Staging slab of a host-resident vector (two per vector on the device).
_SLAB_BYTES = 64 << 20


class _HostVector:
    """Store a row-partitioned vector across device and pinned host memory.

    A vector of rows rows of row entries whose rows [resident, rows) live
    in pinned host memory.

    A pass over the vector (_stream) needs only a range of rows on the device:
    the first resident rows stay there, the others cross PCIe through two
    staging slabs, the copies of one overlapping the kernels of the other.
    The arithmetic is that of the device-resident route (same kernels per
    element). GaNi coordinates (coordinates()) use second-axis rows, whose
    tile-major coordinates are contiguous; the mixed FP64 coordinates cross
    once each way per evaluation (fold), a host-resident Krylov iterate and
    its smoothed copy once each way per iteration (settle, possibly on a side
    stream). Host reads wait for the last copy back (wait).
    """

    def __init__(self, rows, row, dtype=np.complex128, resident=0):
        import cupyx

        self.n1, self.row, self.dtype, self.resident = (
            rows,
            row,
            np.dtype(dtype),
            resident,
        )
        self.size = rows * row
        # Widths in 16-byte entries: an FP32 slab widened for the library map
        # still fits one slab.
        self.width = max(1, _SLAB_BYTES // (16 * self.row))
        self.device = cp.zeros(resident * self.row, self.dtype)
        self.host = cupyx.empty_pinned(((rows - resident) * self.row,), self.dtype)
        staged = min(self.width, rows - resident)
        self.slabs = (
            [cp.empty(staged * self.row, self.dtype) for _ in range(2)]
            if staged
            else []
        )
        self.zero = True  # the host rows are not written before the first pass
        self.last = None  # the last copy back to the host

    @classmethod
    def coordinates(cls, Q, dtype=np.complex128, resident=0):
        """GaNi tile-major coordinates, by second-axis rows."""
        n0, n1, n2 = Q.shape
        return cls(n1, 3 * n0 * stored_planes(n2), dtype, resident)

    def wait(self):
        if self.last is not None:
            self.last.synchronize()

    def rows(self, start, stop, slot):
        """Device view of rows [start, stop): resident, or in a staging slab."""
        if stop <= self.resident:
            return self.device[start * self.row : stop * self.row]
        return self.slabs[slot][: (stop - start) * self.row]

    def _copy(self, start, stop, slot, stream, upload):
        from cupy.cuda import runtime

        slab = self.slabs[slot][: (stop - start) * self.row]
        host = (
            self.host.ctypes.data
            + self.dtype.itemsize * (start - self.resident) * self.row
        )
        if upload and self.zero:
            slab.fill(0)
        elif upload:
            runtime.memcpyAsync(
                slab.data.ptr, host, slab.nbytes, runtime.memcpyHostToDevice, stream.ptr
            )
        else:
            runtime.memcpyAsync(
                host, slab.data.ptr, slab.nbytes, runtime.memcpyDeviceToHost, stream.ptr
            )

    def fold(self, x, consume):
        """Accumulate coordinates and consume updated slabs before host transfer.

        z += x (x restarts at zero); consume(view, start, count, base) reads
        the updated coordinates of rows [start, start+count) (offset base)
        before they return to the host.
        """
        flush = _mixed_kernel("mixed_flush")
        host = isinstance(x, _HostVector)

        def rows(views, start, count, base):
            z = views[0]
            increment = views[1] if host else x[base : base + z.size]
            flush((reduction_blocks(z.size),), (256,), (increment, z, np.int64(z.size)))
            consume(z, start, count, base)

        _stream([self, x] if host else [self], rows, [self])
        if host:
            x.zero = True  # its staged rows were zeroed on the device, not copied back

    def read(self, consume):
        """consume(view, start, count, base) for every row range."""
        _stream(
            [self],
            lambda views, start, count, base: consume(views[0], start, count, base),
        )

    def settle(self, launch, y=None, stream=None):
        """Defer an elementwise update across the host vector and its optional copy.

        A deferred elementwise update of this vector (and its smoothed copy
        y): launch(x, y, base, n) per row range, y None without a copy. With
        stream, the ranges are updated on that stream, after the current
        stream's work, without waiting for the copies back; the returned
        event marks the last kernel (the update's inputs are then free).
        """
        vectors = [self] if y is None else [self, y]
        return _stream(
            vectors,
            lambda views, start, count, base: launch(
                views[0], views[1] if y is not None else None, base, views[0].size
            ),
            vectors,
            compute=stream,
            sync=stream is None,
            bounded=False,
        )

    def smooth(self, x, weight):
        """Update the smoothed iterate as self += weight*(x-self)."""
        from .krylov import _finish_smoothing

        _stream(
            [x, self],
            lambda views, start, count, base: _finish_smoothing(
                views[0], views[1], weight
            ),
            [self],
        )

    def copy(self):
        """Return a copy of the smoothed iterate wholly in host memory."""
        self.wait()
        out = _HostVector(self.n1, self.row, self.dtype)
        k = self.resident * self.row
        if k:
            self.device.get(out=out.host[:k])
        if self.zero:
            out.host[k:].fill(0)
        else:
            np.copyto(out.host[k:], self.host)
        out.zero = False
        return out

    def fill_zero(self):
        """Materialize the logical zero (before products or retention)."""
        self.wait()
        self.device.fill(0)
        self.host.fill(0)
        self.zero = False

    def dot(self, r):
        """Re(z^H r) (complex) or z.r (real) accumulated in FP64, r on the device."""
        self.wait()
        k = self.resident * self.row
        total = _real_dot(self.device, r[:k]) if k else 0.0
        return total if self.zero else total + _real_dot(self.host, r[k:])

    def rounded(self, dtype=np.complex64):
        """Return the vector in pinned host memory, rounded on the device to dtype."""
        import cupyx

        self.wait()
        k = self.resident * self.row
        if k == 0 and self.dtype == dtype and not self.zero:
            return self.host  # an FP32 iterate is retained as is
        out = cupyx.empty_pinned((self.size,), dtype)
        step = self.width * self.row
        for lo in range(0, k, step):  # bounded temporaries beside a full device
            self.device[lo : lo + step].astype(dtype).get(
                out=out[lo : min(k, lo + step)]
            )
        if self.zero:
            out[k:].fill(0)
            return out
        for lo in range(0, self.host.size, step):
            hi = min(self.host.size, lo + step)
            cp.asarray(self.host[lo:hi]).astype(dtype).get(out=out[k + lo : k + hi])
        return out

    def gather(self, out):
        """Gather the whole vector into the device array out."""
        self.wait()
        k = self.resident * self.row
        flat = out.reshape(-1)
        flat[:k] = self.device
        if self.zero:
            flat[k:].fill(0)
        else:
            flat[k:].set(self.host)

    def scatter(self, values):
        """Scatter device array values back into this vector."""
        self.wait()
        k = self.resident * self.row
        flat = values.reshape(-1)
        self.device[...] = flat[:k]
        flat[k:].get(out=self.host)
        self.zero = False


def _stream(vectors, consume, written=(), compute=None, sync=True, bounded=True):
    """One pass over the rows of host-resident vectors.

    consume(views, start, count, base) receives each vector's rows
    [start, start+count) on the device (base = start*row, the offset of
    those entries); the staged rows of the vectors in written then return
    to the host. Row ranges never straddle a vector's resident boundary.
    The kernels run on compute (default: the current stream; another stream
    first waits for the current one). Returns the event after the last
    kernel; sync also waits for the copies back. Rows resident in every
    vector form one range unless bounded (consumers with slab-sized
    temporaries).
    """
    from .fused import _shared_streams

    current = cp.cuda.get_current_stream()
    compute = current if compute is None else compute
    if compute is not current:
        compute.wait_event(current.record())
    upload, download = _shared_streams("host_vectors", 2)
    for v in vectors:  # the previous pass's copies back (any slot)
        if v.last is not None:
            upload.wait_event(v.last)
            compute.wait_event(v.last)
    n1, row = vectors[0].n1, vectors[0].row
    width = min(v.width for v in vectors)
    cuts = sorted({0, n1, *(v.resident for v in vectors)})
    done, k = [None, None], 0
    with compute:
        for first, last in zip(cuts, cuts[1:]):
            staged = [v for v in vectors if first >= v.resident]
            back = [v for v in staged if any(v is w for w in written)]
            for start in range(
                first, last, width if staged or bounded else last - first
            ):
                stop, slot = (
                    min(last, start + (width if staged or bounded else last - first)),
                    k % 2,
                )
                if staged:
                    k += 1
                    if done[slot] is not None:  # the slot's previous rows have left it
                        upload.wait_event(done[slot])
                        compute.wait_event(done[slot])
                    for v in staged:
                        v._copy(start, stop, slot, upload, True)
                    compute.wait_event(upload.record())
                consume(
                    [v.rows(start, stop, slot) for v in vectors],
                    start,
                    stop - start,
                    start * row,
                )
                if staged:
                    finished = compute.record()
                    download.wait_event(finished)
                    for v in back:
                        v._copy(start, stop, slot, download, False)
                    done[slot] = download.record() if back else finished
        event = compute.record()
    if written:
        last = download.record()
        for v in written:
            v.last = last
    if sync:
        download.synchronize()
        if compute is not current:
            compute.synchronize()
    for v in written:
        v.zero = False
    return event


def _solution_dot(z, r):
    """Re(z^H r) in FP64 for a device or host-resident solution."""
    return z.dot(r) if isinstance(z, _HostVector) else _real_dot(z, r)


def _slab_map(Q, z, h, start, count, base):
    """Map slab coordinates to h = QR z on [start, start+count).

    z holds the tile-major coordinates of that slab only (offset base).
    """
    key = "slab_map"
    if key not in _KERNEL_CACHE:
        code = replace_once(
            tiled_coordinate_source(),
            "int tile)",
            "int tile, long long i1start, long long i1count, long long base)",
        )
        code = replace_once(
            code,
            " const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;",
            r"""
 const long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=n0*i1count*nz)return;
 const long long s=((t/(nz*i1count))*n1+i1start+(t/nz)%i1count)*nz+t%nz;""",
        )
        code = code.replace(
            "coordinate+c,adjoint", "coordinate+c-base,adjoint"
        )  # the two reads of z
        code = (
            code.replace("COMPLEX", "double2")
            .replace("REAL", "double")
            .replace("MAKE", "make_double2")
        )
        _KERNEL_CACHE[key] = cp.RawKernel(code, "coordinates")
    n0, _, n2 = Q.shape
    threads = n0 * count * (n2 // 2 + 1)
    args = Q._args(z, h, False) + (
        np.int32(Q.tile),
        np.int64(start),
        np.int64(count),
        np.int64(base),
    )
    _KERNEL_CACHE[key](((threads + 255) // 256,), (256,), args)


def _map_solution(Q, operator, x, h):
    """Map h = QR x for FP64 evaluation of a working-precision solution.

    FP64 device coordinates map directly. FP32 or host-resident tile-major
    coordinates map one second-axis slab at a time, FP32 rows widened into a
    single staging buffer, never as a full FP64 copy.
    """
    host = isinstance(x, _HostVector)
    if not host and x.dtype == cp.complex128:
        operator.map(x, h, False)
        return
    if Q.tile is None:
        operator.map(x.astype(cp.complex128), h, False)
        return
    n0, n1, n2 = Q.shape
    row = 3 * n0 * stored_planes(n2)
    width = max(1, min(n1, _SLAB_BYTES // (16 * row)))
    wide = [None]

    def rows(z, start, count, base):
        if z.dtype != cp.complex128:
            if wide[0] is None:
                wide[0] = cp.empty(width * row, cp.complex128)
            wide[0][: z.size] = z
            z = wide[0][: z.size]
        _slab_map(Q, z, h, start, count, base)

    if host:
        x.read(rows)
    else:
        for a in range(0, n1, width):
            b = min(n1, a + width)
            rows(x[a * row : b * row], a, b - a, a * row)
    hermitian_plane(h, n2, packed=True)  # split the packed k2 = 0 plane


def _working_descriptor(pb, descriptor):
    """Return the FP64 indexed material rounded once to FP32 for Krylov kernels.

    Same ids and table rows; every row must remain SPD after rounding.
    """
    import copy

    cached = pb.solver.get("_linear_working_descriptor")
    if cached is not None and cached[0] is descriptor:
        return cached[1]
    table = descriptor.material_table.astype(cp.float32)
    from .common import SYM21_PAIRS

    full = np.zeros((table.shape[0], 6, 6))
    host = cp.asnumpy(table).astype(np.float64)
    for k, (i, j) in enumerate(SYM21_PAIRS):
        full[:, i, j] = full[:, j, i] = host[:, k]
    working_material_check(full, np.float32)
    working = copy.copy(descriptor)
    working.material_table = table
    pb.solver["_linear_working_descriptor"] = (descriptor, working)
    return working


def _host_audit_required(shape, dtype=np.float32):
    """Six full strains exceed device capacity minus the transient reserve."""
    return (
        36 * np.dtype(dtype).itemsize * int(np.prod(shape))
        > cp.cuda.runtime.memGetInfo()[1] - 512 * 2**20
    )


def _retain_host_solution(pb, shape, dtype=np.float32):
    """Avoid a host->device->host round trip for energy-only capacity cases.

    Enable this when the audit already retains a host strain. Requested
    field/stress/sensitivity/partial outputs keep their device-field contract.
    """
    solve = pb.solve
    return (
        not solve.get("store_solution_fields", True)
        and not solve.get("partial_load_output", False)
        and solve.get("affine_sensitivity_cfields") is None
        and not callable(solve.get("stress_field_consumer"))
        and not callable(solve.get("solution_field_consumer"))
        and str(solve.get("postprocess_assembly", "gemm")) == "fused"
        and _host_audit_required(shape, dtype)
    )


def _build_coordinate_rhs(template, A, Q, pipeline):
    """Build RQ* FFT(-C E) once; retain G FFT(-C E) for the physical audit.

    RQ* G = RQ*: the compatible-coordinate map needs no inverse/forward
    round trip through a six-component physical RHS. cuFFT and the production
    direct projection remain independent of the fused iteration kernels.
    """
    if (
        pipeline is None
        and template.val.dtype == cp.float32
        and not library_capacity_required(Q.shape, template.val.dtype)
    ):
        return None
    if pipeline is not None:
        Q = Q.tiled(pipeline.config["ta"])
    stress = A(-template)
    rhs, spectrum = Q.adjoint_with_spectrum(stress.val)
    del stress
    projected = elasticity_direct_array(Q.P.descriptor, spectrum, out=spectrum)
    norm = np.sqrt(_spectral_distance(projected, Q.shape))
    # Above the capacity for six real fields, retain the independent physical
    # RHS in pinned host memory. Audits compare it in bounded spectral chunks;
    # the candidate projection and all norms remain GPU/FP64 operations.
    if _host_audit_required(Q.shape, projected.real.dtype):
        memory = cp.cuda.alloc_pinned_memory(projected.nbytes)
        host = np.frombuffer(memory, projected.dtype, projected.size).reshape(
            projected.shape
        )
        projected.get(out=host)
        projected = host
    # Tensor metadata for copies/solutions, without allocating a physical RHS.
    result = template.copy(
        val=cp.broadcast_to(cp.zeros((), template.val.dtype), template.val.shape)
    )
    result.cufhom_coordinate_rhs = (Q, pipeline, rhs, projected, norm)
    return result


def _squared_distance(a, b):
    """FP64 sum of (a-b)^2 for FP32/FP64 fields, without a residual buffer."""
    if (
        a.dtype not in (cp.float32, cp.float64)
        or b.dtype != a.dtype
        or a.shape != b.shape
        or not a.flags.c_contiguous
        or not b.flags.c_contiguous
    ):
        raise ValueError(
            "The residual audit expects same-precision FP32/FP64 fields of one shape"
        )
    key = ("squared_distance", a.dtype.str)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = cp.RawKernel(
            cuda_source(_SQUARED_DISTANCE, a.dtype), "squared_distance"
        )
    blocks = reduction_blocks(a.size)
    partial = cp.empty(blocks, cp.float64)
    _KERNEL_CACHE[key]((blocks,), (256,), (a, b, partial, np.int64(a.size)))
    return float(cp.sum(partial, dtype=cp.float64))


def _solve_pipeline(pb, Q, descriptor):
    """Return the fused pipeline, or None while a small new grid compiles.

    One decision per solve: all six loads take the same path.
    """
    cached = pb.solver.get("_linear_fused_pipeline")
    if (
        cached is not None
        and cached[0] is descriptor
        and (cached[1] is None or cached[1].shape == tuple(Q.shape))
    ):
        return cached[1]
    pipeline = None
    if fused_supported(Q, descriptor) and not defer_compile(Q, descriptor):
        pipeline = FusedPipeline(Q, descriptor)
    pb.solver["_linear_fused_pipeline"] = (descriptor, pipeline)
    return pipeline


def _fused_pipeline_cg(
    pipeline, action, rhs, dot, relative, maxiter, options, audit_cache
):
    """Run the same CG in the pipeline's tile-major coordinates.

    Search directions use the fused kernels. True-residual audits keep the
    production action, so the accepted strain/stress cache is unchanged. The
    Coordinate kernels read/write tile-major vectors directly, so audits and
    the final strain need no separate layout conversion.
    """
    step = FusedStep(pipeline)

    def audit(z, retain=False):
        # Audits allocate production FFT buffers of the same size; the pool
        # reuses the released spectrum for them.
        pipeline.release()
        return action(z, retain=retain)

    def energy(z, direction=None):
        audit_cache.clear()
        return step, pipeline.apply(z, direction)

    pipeline.pending = None
    options = dict(options, action_and_dot=energy, settle=pipeline.settle)
    if "residual_action" in options:
        options["residual_action"] = lambda z: audit(z, retain=True)
    if (
        pipeline.config["graph_iterations"] > 0
        and os.environ.get("CUFHOM_CUDA_GRAPHS", "1") != "0"
    ):
        # CUDA graphs: several iterations per launch, scalars on the device.
        z, iterations = pipeline.graph_cg(
            rhs,
            dot,
            relative,
            maxiter,
            options.get("residual_action", audit),
            options.get("accept"),
        )
        if options.get("stats") is not None:
            options["stats"].update(pipeline.last_cg_stats)
    else:
        z, iterations = fused_cg(audit, rhs, dot, relative, maxiter, **options)
    pipeline.release()
    return z, iterations


# Indicator spectra recomputed per load when not cached: up to this many fit in
# the Krylov vectors (x, p, y) that do not exist yet at the load evaluation.
_TRANSIENT_INDICATORS = 4
# Many-row materials (no indicator spectra): initial load by the fused evaluator.
_EVALUATOR_LOAD = __import__("os").environ.get("CUFHOM_EVALUATOR_LOAD", "1") != "0"


def _phase_spectra(pb, descriptor, shape, operator, real, spectrum):
    """Unscaled FP64 R2C of the indicators of phases 1..P-1, or None.

    C(x)E = C_0 E + sum_{p>=1} chi_p (C_p - C_0) E, so the spectrum of the load
    stress of any macro strain follows from P-1 scalar transforms (phase 0
    only adds N C_0 E at k = 0) instead of six per load. Kept per geometry
    for P-1 <= 6 when small beside the free memory (it must not compete with
    the large-grid working set); otherwise recomputed per load for
    P-1 <= _TRANSIENT_INDICATORS when they fit beside the load evaluation;
    None leaves the six-component transform. Each indicator is transformed
    by the load's library plan (operator) in one component (real, spectrum)
    of its work field, which the load overwrites afterwards.
    """
    from .common import device_free_bytes

    rows = int(descriptor.material_table.shape[0])
    count = int(np.prod(shape[:-1])) * (shape[-1] // 2 + 1)
    transient = (
        rows - 1 <= _TRANSIENT_INDICATORS
        and 16 * (rows - 1) * count < device_free_bytes() - _RESERVE_BYTES
    )
    cached = pb.solver.get("_linear_phase_spectra")
    if cached is not None and cached[0] is descriptor and cached[1] == shape:
        if cached[2] is not None or not transient:
            return cached[2]
        return _indicators(descriptor, shape, operator, real, spectrum)
    keep = (
        rows - 1 <= 6
        and 16 * (rows - 1) * count < _INDICATOR_SHARE * device_free_bytes()
    )
    spectra = (
        _indicators(descriptor, shape, operator, real, spectrum)
        if keep or transient
        else None
    )
    pb.solver["_linear_phase_spectra"] = (descriptor, shape, spectra if keep else None)
    return spectra


def _indicators(descriptor, shape, operator, real, spectrum):
    from cupy.cuda import cufft

    rows = int(descriptor.material_table.shape[0])
    ids = descriptor.val.reshape(shape)
    spectra = cp.empty(
        (rows - 1,) + tuple(shape[:-1]) + (shape[-1] // 2 + 1,), cp.complex128
    )
    for p in range(1, rows):
        cp.equal(ids, p, out=real)
        operator.f.fft(real, spectrum, cufft.CUFFT_FORWARD)
        spectra[p - 1] = spectrum
    return spectra


def _material_constants(pb, descriptor, bulk, shear):
    """Host FP64 stiffness rows, Voigt mean and material bounds of C0."""
    cached = pb.solver.get("_linear_material_constants")
    if cached is not None and cached[0] is descriptor:
        return cached[1]
    from .common import SYM21_PAIRS, reference_bounds

    table = cp.asnumpy(descriptor.material_table).astype(np.float64)
    full = np.zeros((len(table), 6, 6))
    for k, (i, j) in enumerate(SYM21_PAIRS):
        full[:, i, j] = full[:, j, i] = table[:, k]
    from .common import device_label_counts

    counts = device_label_counts(descriptor.val, len(table)).astype(np.float64)
    c0 = 2 * shear * np.eye(6)
    c0[:3, :3] += bulk - 2 * shear / 3
    value = dict(
        stiffness=full,
        voigt=np.einsum("p,pij->ij", counts / counts.sum(), full),
        bounds=reference_bounds(full, c0),
    )
    pb.solver["_linear_material_constants"] = (descriptor, value)
    return value


def mixed_load(
    pb,
    descriptor,
    bulk,
    shear,
    shape,
    macro,
    par,
    energy=None,
    keep_strain=False,
    residency="device",
    evaluation="df64",
    configuration="mixed",
):
    """One load of the GaNi coordinate route in any arithmetic configuration.

    descriptor holds the problem coefficients in FP64 (FP32 problems promote
    theirs exactly). The fused pipeline of the working precision produces the
    search directions, and every candidate evaluates the operator in
    total-strain form, giving the fresh residual and the physical audit.
    configuration (as HEX8):
      'mixed'    FP64 problem, FP32 Krylov (material rounded once), FP64
                 coordinates with scheduled reliable replacements, df64 or
                 FP64 evaluations;
      'float32'  FP32 problem and Krylov; the Krylov iterate is the solution,
                 evaluated in df64 or FP64;
      'float64'  FP64 throughout.
    Returns the coordinates z (FP64 in the mixed configuration, otherwise the
    working precision), the fresh residual r (working precision) and the mean
    stress of the last evaluation, the CG info and, with keep_strain, the
    fluctuation strain (FP64) of that evaluation. residency (RESIDENCIES;
    the host levels need the fused pipeline's tiled layout): 'host' keeps
    the mixed z in pinned host memory, 'iterate' also the Krylov iterate x
    and its smoothed copy, with the rows that fit on the device
    (_HostVector). evaluation selects the arithmetic of the fused
    evaluations after the load ('df64' or 'float64', always 'float64' in the
    FP64 configuration); the RHS, keep_strain and grids outside the fused
    layout use cuFFT FP64.
    """
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"configuration must be one of {CONFIGURATIONS}")
    if residency not in RESIDENCIES:
        raise ValueError(f"residency must be one of {RESIDENCIES}")
    mixed = configuration == "mixed"
    if residency == "host" and not mixed:
        raise ValueError(
            "A fixed configuration iterates on its solution: its host residency is 'iterate'"
        )
    host_coordinates, host_iterate = (
        mixed and residency != "device",
        residency == "iterate",
    )
    if configuration == "float64":
        evaluation = "float64"
    working = np.dtype("float64" if configuration == "float64" else "float32")
    complex_type = cp.complex128 if working == np.float64 else cp.complex64
    shape = tuple(int(n) for n in shape)
    energy = energy_tolerance(energy)
    volume = int(np.prod(shape))
    P = MixedProjector(shape, tuple(pb.Y), np.ones(6, bool), cp)
    Q = SymmetricCoordinates(P, bulk, shear)
    material = (
        descriptor
        if configuration == "float64"
        else _working_descriptor(pb, descriptor)
    )
    pipeline = _solve_pipeline(pb, Q, material)
    if pipeline is not None:
        Q = Q.tiled(pipeline.config["ta"])
    from .active_capacity import ActiveCapacity

    operator = ActiveCapacity(descriptor, shape, pb.Y, Q)
    if residency != "device" and Q.tile is None:
        raise ValueError(
            "A host-resident GaNi solution needs the fused pipeline layout"
        )
    # Mixed: FP64 coordinates fed by the FP32 increments. Fixed: the Krylov
    # iterate itself, recorded at each evaluation (smoothing may swap it).
    z64 = (
        (
            _HostVector.coordinates(Q)
            if host_coordinates
            else cp.zeros(Q.size, cp.complex128)
        )
        if mixed
        else None
    )
    solution = [None]
    evaluator = (
        None
        if keep_strain
        else _evaluation_pipeline(
            pb,
            Q,
            descriptor,
            evaluation,
            pipeline,
            "float64" if mixed else working.name,
            working.name,
        )
    )
    blocks = reduction_blocks(Q.size)
    partial = cp.empty(blocks, cp.float64)
    physical, counts, fresh = (
        [np.inf],
        dict(replacements=0, evaluations=0, fused=0),
        [np.nan],
    )
    stress, residual, strain = (
        [None],
        [None],
        [None],
    )  # mean stress, residual, strain of the last evaluation
    route = [
        "six_components"
    ]  # evaluation of the load: 'fused', 'indicators' or 'six_components'
    constants = _material_constants(pb, descriptor, bulk, shear)
    dot_kernel = _mixed_kernel(
        "mixed_dot" if complex_type == cp.complex64 else "mixed_dot128"
    )

    def dot(a, b):
        dot_kernel((blocks,), (256,), (a, b, partial, np.int64(a.size)))
        return float(cp.sum(partial, dtype=cp.float64))

    # On the host levels and for loads over half the device, the pipeline
    # spectrum and the FP64 evaluation and work fields take turns in most of
    # its memory. A pool block they leave behind would be split for a smaller
    # array and leave no contiguous room for the next one, so their blocks go
    # back to the device between phases.
    compact = (
        residency != "device"
        or 2 * mixed_working_bytes(shape, residency, configuration, energy is not None)
        > cp.cuda.runtime.memGetInfo()[1]
    )

    # The load itself is evaluated with cuFFT FP64 and the indicator spectra
    # (cheaper) when their plans (work areas up to one FP64 field each) and
    # spectra fit beside the load, decided once per call; otherwise by the
    # fused pipeline at z = 0, without cuFFT, waiting for its compilation.
    library_load = pb.solver.get("_linear_library_load")
    if library_load is None:
        library_load = evaluator is None or residency == "device"
        if evaluator is not None and residency == "device":
            from .common import device_free_bytes

            rows, half = (
                int(descriptor.material_table.shape[0]),
                volume // shape[-1] * (shape[-1] // 2 + 1),
            )
            # Without indicator spectra (more than 6 non-reference rows) the
            # library load is six FP64 transforms of C(x)E per load, slower
            # than the fused evaluator at z = 0 that also certifies acceptance.
            library_load = (
                rows - 1 <= 6 or not _EVALUATOR_LOAD
            ) and mixed_working_bytes(
                shape, residency, configuration, energy is not None
            ) + (
                32 + 16 * min(rows - 1, 6)
            ) * half < device_free_bytes() - _RESERVE_BYTES
        pb.solver["_linear_library_load"] = library_load
    if not library_load and not evaluator.ready:
        evaluator.wait()
    if (
        not library_load
        and pipeline.ids.dtype == cp.uint16
        and descriptor.val is not None
    ):
        # The fused route reads only the pipeline's compact ids from here on
        # (the material constants and pipelines are cached): the int32 ids,
        # 4 B/voxel, leave room for resident rows.
        descriptor.val = material.val = None

    # In FP64 the pipeline spectrum, the evaluation field and the library work
    # field have one size and alternate: a compact load gives them one
    # allocation, kept by the call (no allocation per phase, no split blocks).
    # FP32 and mixed keep the smaller FP32 spectrum while iterating and return
    # the pool's free blocks between phases instead, as the host levels do.
    shared = compact and configuration == "float64"
    if shared:
        work = 96 * (volume // shape[-1]) * (shape[-1] // 2 + 1)
        workspace = pb.solver.get("_linear_workspace")
        if workspace is None or workspace.nbytes < work:
            pb.solver["_linear_workspace"] = None
            cp.get_default_memory_pool().free_all_blocks()
            workspace = pb.solver["_linear_workspace"] = cp.empty(work, cp.uint8)
        for consumer in (pipeline, evaluator, operator):
            if consumer is not None:
                consumer.workspace = workspace

    def return_to_device():
        if compact and (residency != "device" or not shared):
            cp.get_default_memory_pool().free_all_blocks()

    def replace(x, r, initial=False, audit=True):
        """Rebuild the fresh residual in the selected evaluation arithmetic.

        Mixed: z64 += x (x restarts at zero); fixed: z = x. Then
        r = fl(b - H z) = fl(-S* FFT(C(E + S z))) in the evaluation arithmetic.

        initial (z = 0): with library_load, the load stress spectrum comes
        from the phase indicator spectra when available, else from C(x)E
        without the transform of the zero fluctuation; otherwise the fused
        pipeline evaluates it.
        """
        if pipeline is not None:
            pipeline.release()
            return_to_device()
        if not initial and not mixed:
            solution[0] = x
        if not initial and mixed and not host_coordinates:
            _mixed_kernel("mixed_flush")((blocks,), (256,), (x, z64, np.int64(x.size)))
        if evaluator is not None and evaluator.ready and not (initial and library_load):
            if initial:
                evaluator.zero()
                route[0] = "fused"
            elif host_coordinates:
                z64.fold(x, evaluator.inverse)
            elif host_iterate:
                x.read(evaluator.inverse)
            else:
                evaluator.inverse(z64 if mixed else x)
            rr, metric, total = evaluator.forward(macro, r)
            evaluator.release()
            return_to_device()
            stress[0], residual[0], fresh[0] = total / volume, r, rr
            if audit:
                physical[0] = np.sqrt(metric)
            counts["evaluations"] += 1
            counts["fused"] += not initial
            return rr
        w, h, v = operator.work()
        try:
            spectra = (
                _phase_spectra(pb, descriptor, shape, operator, v[0], h[0])
                if initial
                else None
            )
            if spectra is not None:
                c = constants["stiffness"]
                columns = cp.asarray(
                    np.ascontiguousarray((c[1:] @ macro - c[0] @ macro).T)
                )
                count = h[0].size
                _mixed_kernel("phase_load")(
                    (reduction_blocks(count),),
                    (256,),
                    (spectra, columns, h, np.int64(count), np.int32(len(spectra))),
                )
                h[:, 0, 0, 0] += cp.asarray(volume * (c[0] @ macro))
                route[0] = "indicators"
                del spectra
            else:
                if initial:
                    v[...] = cp.asarray(macro)[:, None, None, None]
                else:
                    if host_coordinates:
                        z64.fold(
                            x,
                            lambda slab, start, count, base, spectrum=h: _slab_map(
                                Q, slab, spectrum, start, count, base
                            ),
                        )
                        hermitian_plane(
                            h, shape[-1], packed=True
                        )  # split the packed k2 = 0 plane
                    else:
                        _map_solution(Q, operator, z64 if mixed else x, h)
                    operator.inverse(h, v)
                    # The fluctuation strain of the latest evaluation is the
                    # returned solution when this candidate is accepted.
                    if keep_strain:
                        strain[0] = cp.ascontiguousarray(v)
                    v += cp.asarray(macro)[:, None, None, None]
                operator.constitutive(w)
                operator.forward(v, h)
            stress[0] = cp.asnumpy(h[:, 0, 0, 0].real) / volume
            residual[0] = r
            metric = _residual_map(Q, h, r, audit)
            rr = fresh[0] = dot(r, r)
            if audit:
                physical[0] = np.sqrt(metric)
        finally:
            del w, h, v
            return_to_device()
        counts["evaluations"] += 1
        return rr

    def action_energy(z, direction=None):
        return energy_action(Q, material, z, direction)

    stats = {}
    options = dict(
        residual_smoothing=energy is None,
        action_and_dot=action_energy,
        device_update_callback=energy_update_device,
        replace=replace,
        stats=stats,
        spill_direction=residency != "device",
        fresh_metric=lambda: physical[0] if energy is None else np.sqrt(fresh[0]),
    )
    if pipeline is not None:
        step = FusedStep(pipeline)
        pipeline.pending = None
        options.update(
            action_and_dot=lambda z, direction=None: (
                step,
                pipeline.apply(z, direction),
            ),
            settle=pipeline.settle,
        )
        if energy is None and volume <= _DEVICE_PREFIX_MAX_VOXELS and not host_iterate:
            options["device_prefix"] = pipeline.prefix
    rhs = cp.empty(Q.size, complex_type)
    replace(None, rhs, initial=True)
    iterate = None
    if host_iterate:
        # After the load evaluation: the rows that fit beside r and the rest.
        iterate = options["iterate"] = _HostVector.coordinates(
            Q,
            complex_type,
            _resident_rows(shape, configuration, energy is None, shared),
        )
    rhs_norm = physical[0]
    threshold = max(par.get("atol", 0.0), par["rtol"] * rhs_norm)
    audit_margin = float(pb.solver.get("_linear_audit_margin", 1.0))
    relative = (threshold / rhs_norm if rhs_norm else par["rtol"]) / audit_margin
    tightenings, audit_ratios = [], []
    gamma = constants["bounds"]["gamma_minus"]
    initial_energy = float(macro @ constants["voigt"] @ macro)
    candidate = dict(energy=np.nan, threshold=np.nan)

    def accept(_):
        value = physical[0]
        audit_ratios.append(float(value / threshold))
        if not np.isfinite(value) or value <= threshold or len(tightenings) >= 3:
            return None
        tightenings.append(float(value / threshold))
        return (0.8 * threshold / value) ** 2

    def accept_energy(_):
        # Exact energy of the candidate, C~ = E.<sigma> - Re(z^H r)/N, and
        # its fresh bound G0/gamma_- with G0 = r.r/N (evaluation operator).
        z = z64 if mixed else solution[0]
        product = 0.0 if z is None else _solution_dot(z, residual[0])
        current = float(macro @ stress[0]) - product / volume
        bound = fresh[0] / (volume * gamma)
        candidate.update(energy=current, threshold=energy * (current - bound))
        return None if bound <= candidate["threshold"] else 1.0

    def trigger(work, new):
        bound = new / (volume * gamma)
        return bound <= energy * (initial_energy - work / volume - bound)

    # Scheduled replacements keep an FP32 recursion consistent with its FP64
    # coordinates; the fixed configurations, as in HEX8, replace only at
    # candidates (drop 0, no growth guard).
    if energy is None:
        options.update(
            accept=accept,
            drop=_segment_drop(relative) if mixed else 0.0,
            growth=_FP32_GROWTH if mixed else None,
            margin=pb.solver.setdefault("_linear_drift_margin", [1.0]),
        )
    else:
        # The trigger stops near r.r = tau N gamma_- E (E <= the Voigt energy):
        # the same segment schedule towards that relative drop.
        target = (
            np.sqrt(energy * volume * gamma * initial_energy / fresh[0])
            if fresh[0] > 0
            else 1.0
        )
        options.update(
            accept=accept_energy,
            energy=trigger,
            drop=_segment_drop(target) if mixed else 0.0,
            growth=_FP32_GROWTH if mixed else None,
        )
        relative = 0.0
    iterations = 0
    if rhs_norm > threshold or (energy is not None and fresh[0] > 0):
        try:
            _, iterations = fused_cg(
                None, rhs, dot, relative, par["maxiter"], **options
            )
        except LinearSolveError as error:
            error.diagnostics.update(
                restarts=stats.get("restarts"), replacements=stats.get("replacements")
            )
            raise
    else:
        # No pass materialized a host-resident logical zero. Initialize only
        # this accepted initial field, before energy products or retention.
        if host_coordinates:
            z64.fill_zero()
        elif host_iterate:
            iterate.fill_zero()
        if energy is not None:
            accept_energy(None)  # zero load: exact homogeneous energy, no Krylov step
    if pipeline is not None:
        pipeline.release()
    if audit_ratios and np.isfinite(audit_ratios[0]) and audit_ratios[0] > 1:
        pb.solver["_linear_audit_margin"] = min(
            4.0, audit_margin * audit_ratios[0] * 1.02
        )
    norm = physical[0]
    if energy is None and (not np.isfinite(norm) or norm > threshold):
        raise LinearSolveError(
            "Displacement PCG failed the fresh original-equation residual audit",
            status="fresh_residual_rejected",
            iterations=iterations,
        )
    info = dict(
        kit=iterations,
        norm_res=norm,
        norm_res_log=[norm],
        physical_rhs_host=False,
        retained_solution_host=False,
        coordinate_iterations=iterations,
        correction_iterations=0,
        linear_final_strain_reuse=False,
        linear_final_audit_reuse=True,
        linear_final_spectrum_reuse=False,
        physical_audit_tightenings=tightenings,
        physical_audit_ratios=audit_ratios,
        physical_audit_margin=audit_margin,
        linear_fused_fft=pipeline is not None,
        linear_coordinate_rhs=True,
        linear_coordinate_layout="tiled" if Q.tile is not None else "production",
        linear_fused_fft_config=dict(pipeline.config) if pipeline is not None else None,
        warm_start_requested=False,
        warm_start_accepted=False,
        initial_norm_res_rel=None,
        linear_working_dtype=working.name,
        fp64_evaluations=counts["evaluations"],
        linear_residency=residency,
        iterate_resident_rows=None if iterate is None else iterate.resident,
        replacements=stats.get("replacements", 0),
        restarts=stats.get("restarts", 0),
        recursive_norm_res=stats.get(
            "recursive_norm_res", np.sqrt(fresh[0]) if iterations == 0 else None
        ),
        candidate_checks=stats.get("candidate_checks", 0),
        linear_cg_control="device_prefix4" if "device_prefix" in options else "host",
        control_queries=stats.get("control_queries", 0),
        device_control_iterations=stats.get("device_control_iterations", 0),
        reliable_drop=options["drop"],
        energy_G0=fresh[0] / volume,
        criterion="physical_residual" if energy is None else "energy",
        energy_rtol=energy,
        energy=candidate["energy"],
        energy_threshold=candidate["threshold"],
        phase_spectra_rhs=route[0] == "indicators",
        load_evaluation=route[0],
        evaluation=_evaluation_label(evaluator, counts),
    )
    _finalize_cg_info(info, rhs_norm, threshold, par["maxiter"])
    info.update(
        true_norm_res_rel=info["norm_res_rel"], true_residual_converged=energy is None
    )
    if energy is not None:
        info["converged"] = (
            True  # accepted on the fresh energy bound, not on the residual
        )
    # The last evaluation was of the accepted z (none if the load is met at z = 0).
    if not mixed and solution[0] is not None:
        z64 = solution[0]
    elif not mixed and host_iterate:
        z64 = iterate
        if z64.zero:
            z64.fill_zero()
    elif not mixed:
        z64 = cp.zeros(Q.size, complex_type)
    return dict(
        Q=Q, z=z64, residual=residual[0], stress=stress[0], strain=strain[0], info=info
    )


def make_linear_solver(*, A, GN, pb, fallback, bulk, shear, working=None, energy=None):
    """Per-load Krylov adapter consumed by the core elasticity application.

    fallback is the core strain CG the application offers; this adapter
    never switches to it, so a candidate that fails its audit fails the load.

    working='float32' with an FP64 material: reliable-update CG. The FP32
    fused pipeline (material rounded once) produces the search directions;
    the coordinates accumulate in FP64, and every replacement and candidate
    evaluates the FP64 operator in total-strain form, giving the residual and
    the same physical audit and threshold as the FP64 route.
    """
    descriptor = getattr(A, "op", A)  # profiling wraps A in a timed callable
    if working is not None and (
        np.dtype(working) != np.float32 or descriptor.material_table.dtype != cp.float64
    ):
        raise ValueError(
            "The mixed coordinate route needs an FP64 material and FP32 working precision"
        )
    if energy is not None and (
        working is None or not np.isfinite(energy) or not 0 < energy < 1
    ):
        raise ValueError(
            "The energy criterion needs the mixed coordinate route and 0 < energy < 1"
        )

    def capacity_engine(shape):
        from .active_capacity import ActiveCapacity

        P = MixedProjector(shape, tuple(pb.Y), np.ones(6, bool), cp)
        Q = SymmetricCoordinates(P, bulk, shear)
        pipeline = _solve_pipeline(pb, Q, descriptor)
        if pipeline is not None:
            Q = Q.tiled(pipeline.config["ta"])
        return ActiveCapacity(descriptor, shape, pb.Y, Q, pipeline)

    def solve_mixed(B, par, macro):
        out = mixed_load(
            pb,
            descriptor,
            bulk,
            shear,
            tuple(B.N),
            macro,
            par,
            energy,
            keep_strain=True,
        )
        strain = (
            out["strain"] if out["strain"] is not None else out["Q"].strain(out["z"])
        )
        return B.copy(val=strain), out["info"]

    def solve(solver, Afun, B, x0, par):
        macro = getattr(B, "cufhom_mixed_macro", None)
        if macro is not None and x0 is None:
            del B.cufhom_mixed_macro
            return solve_mixed(B, par, macro)
        shape = tuple(B.N)
        volume = int(np.prod(shape))
        dtype = B.val.dtype
        if dtype != descriptor.material_table.dtype:
            raise ValueError("Coordinate RHS and material must have the same precision")
        P = MixedProjector(shape, tuple(pb.Y), np.ones(6, bool), cp)
        Q = SymmetricCoordinates(P, bulk, shear)
        if not supports_energy(Q, descriptor):
            raise ValueError(
                "Displacement PCG requires the indexed sym21 FP32/FP64 material"
            )
        pipeline = _solve_pipeline(pb, Q, descriptor)
        if pipeline is not None:
            Q = Q.tiled(pipeline.config["ta"])
        reduction = CupyWarpCGReduction(B)
        prepared = getattr(B, "cufhom_coordinate_rhs", None)
        if prepared is not None:
            Q, pipeline, coordinate_rhs, physical_rhs_spectrum, prepared_norm = prepared
            del B.cufhom_coordinate_rhs

        latest = [np.nan]  # r.r of the fresh residual a candidate audit inspects

        def dot(a, b):
            reduction.dot_kernel(
                (reduction.blocks,),
                (reduction.threads,),
                (
                    a.view(dtype),
                    b.view(dtype),
                    reduction.partial,
                    np.int64(a.view(dtype).size),
                ),
            )
            latest[0] = float(cp.sum(reduction.partial, dtype=cp.float64))
            return latest[0]

        audit_cache = []

        def action(z, retain=False):
            audit_cache.clear()
            strain = B.copy(val=Q.strain(z))
            stress = A(strain)
            if not retain:
                del strain
                return Q.adjoint(stress.val)
            if prepared is not None and isinstance(physical_rhs_spectrum, np.ndarray):
                # Large-grid independent audits need not keep the accepted
                # real strain on device while cuFFT holds stress, spectrum
                # and its workspace. Preserve its exact FP32 bytes on host;
                # restore only after CG has released its other vectors.
                memory = cp.cuda.alloc_pinned_memory(strain.val.nbytes)
                host = np.frombuffer(memory, strain.val.dtype, strain.val.size).reshape(
                    strain.val.shape
                )
                strain.val.get(out=host)
                strain.val = host
            value, spectrum = Q.adjoint_with_spectrum(stress.val)
            # With its unscaled spectrum retained, the audit projection reads
            # only the stress metadata: keep a zero-stride view of the same
            # shape/dtype and release the six-component field.
            stress.val = cp.broadcast_to(
                cp.zeros((), stress.val.dtype), stress.val.shape
            )
            audit_cache[:] = [z, strain, stress, spectrum]
            return value

        def action_energy(z, direction=None):
            audit_cache.clear()
            return energy_action(Q, descriptor, z, direction)

        stats = dict(restarts=0, replacements=0, candidate_checks=0)
        fused_options = dict(
            residual_smoothing=True,
            action_and_dot=action_energy,
            device_update_callback=energy_update_device,
            stats=stats,
        )
        # Axis-split FFTs fused with the coordinate, material and update
        # kernels; one hybrid spectrum is shared by the loads of this solve.

        # The existing direct-projection raw-C2R operator multiplies both
        # sides by volume. Remove that common factor in physical coordinates.
        rhs_norm = (
            prepared_norm
            if prepared is not None
            else np.sqrt(dot(B.val, B.val) / volume)
        )
        threshold = max(par.get("atol", 0.0), par["rtol"] * rhs_norm)
        relative = threshold / rhs_norm if rhs_norm else par["rtol"]
        initial, initial_norm = None, None
        if x0 is not None:
            if not bool(cp.all(cp.isfinite(x0.val))):
                raise ValueError("initial_solution_fields must be finite")
            # Snapshots are physical fluctuation strains, not the current
            # material's whitened coordinates. Project with G, NOT QR RQ*.
            # This also removes means/Nyquist modes and owns a new buffer.
            trial = B.copy(val=P(x0.val))
            residual = B.val - Afun(trial).val
            initial_norm = np.sqrt(dot(residual, residual) / volume)
            if not np.isfinite(initial_norm):
                raise ValueError("Nonfinite initial-solution residual")
            # A distant material can give a worse guess than zero. Keep the
            # same sample order and fall back to a cold solve in that case.
            if initial_norm <= max(rhs_norm, threshold):
                initial = trial
            else:
                del residual
            del trial

        strain_reused = stress_reused = spectrum_reused = False
        tightenings, audit_ratios, audit_margin = [], [], 1.0
        energy_rr = [None]
        if initial is not None and initial_norm <= threshold:
            result, iterations, norm = initial, 0, initial_norm
        else:
            if initial is not None:
                rhs = Q.adjoint(residual) / dtype.type(volume)
                warm_squared = dot(rhs, rhs)
                # Solve A_new * delta = b_new - A_new * initial, starting
                # delta at zero. The stopping scale belongs to the ORIGINAL
                # RHS, never to the smaller correction RHS.
                # ||r_physical||_RMS <= sqrt(N)*||R^-1||*||r_coordinates||.
                # Use this bound to avoid an unnecessary strain-CG cleanup
                # when the correction has a different spectral distribution.
                inverse_root_norm = np.sqrt(max(2 * shear, bulk + 4 * shear / 3))
                if warm_squared > 0:
                    relative = threshold / (
                        np.sqrt(volume * warm_squared) * inverse_root_norm
                    )
                del residual
            else:
                rhs = (
                    coordinate_rhs
                    if prepared is not None
                    else Q.adjoint(B.val) / dtype.type(volume)
                )

            def physical_audit(z):
                """Original-equation residual of a candidate in the solver's layout."""
                strain_hit = bool(
                    audit_cache and audit_cache[0] is z and initial is None
                )
                candidate = audit_cache[1] if strain_hit else B.copy(val=Q.strain(z))
                if initial is not None:
                    candidate.val += initial.val
                if prepared is not None:
                    if strain_hit:
                        spectrum = audit_cache[3]
                    else:
                        spectrum = core_fft.rfftn_cached(
                            A(candidate).val, shape, (1, 2, 3)
                        )
                    projected_spectrum = elasticity_direct_array(
                        Q.P.descriptor, spectrum, out=spectrum
                    )
                    value = np.sqrt(
                        _spectral_distance(
                            physical_rhs_spectrum, shape, projected_spectrum
                        )
                    )
                    audit_cache.clear()
                    return candidate, value, strain_hit, strain_hit, strain_hit
                projected, spectrum_hit = None, False
                if strain_hit:
                    projected = Afun.apply_material_result(
                        audit_cache[2], A, GN, spectrum=audit_cache[3]
                    )
                    spectrum_hit = projected is not None
                audit_cache.clear()
                stress_hit = projected is not None
                if not stress_hit:
                    projected = Afun(candidate)
                value = np.sqrt(_squared_distance(B.val, projected.val) / volume)
                del projected
                return candidate, value, strain_hit, stress_hit, spectrum_hit

            audits = []
            if initial is None:
                # Loads of one solve share the operator; the first physical
                # audit ratio of previous loads predicts the coordinate margin
                # this load needs. Acceptance remains the physical audit.
                audit_margin = float(pb.solver.get("_linear_audit_margin", 1.0))
                relative = relative / audit_margin
                # Search-direction actions never populate the accepted-field cache.
                fused_options["residual_action"] = lambda z: action(z, retain=True)

                def accept(z):
                    # The coordinate and physical residual norms differ by the
                    # preconditioner. When the physical audit fails, continue
                    # the same CG with a stricter coordinate limit instead of
                    # the far more expensive strain-CG correction below.
                    # Both CG loops compute the candidate's fresh r.r last.
                    energy_rr[0] = latest[0]
                    audits[:] = [(z,) + physical_audit(z)]
                    value = audits[0][2]
                    audit_ratios.append(float(value / threshold))
                    if (
                        not np.isfinite(value)
                        or value <= threshold
                        or len(tightenings) >= 3
                    ):
                        return None
                    tightenings.append(float(value / threshold))
                    audits.clear()  # the rejected strain field is not retained
                    return (0.8 * threshold / value) ** 2

                fused_options["accept"] = accept
            if pipeline is None:
                z, iterations = fused_cg(
                    action, rhs, dot, relative, par["maxiter"], **fused_options
                )
            else:
                z, iterations = _fused_pipeline_cg(
                    pipeline,
                    action,
                    rhs,
                    dot,
                    relative,
                    par["maxiter"],
                    fused_options,
                    audit_cache,
                )
            if audit_ratios and np.isfinite(audit_ratios[0]) and audit_ratios[0] > 1:
                pb.solver["_linear_audit_margin"] = min(
                    4.0, audit_margin * audit_ratios[0] * 1.02
                )
            if audits and audits[0][0] is z:
                _, result, norm, strain_reused, stress_reused, spectrum_reused = audits[
                    0
                ]
            else:
                result, norm, strain_reused, stress_reused, spectrum_reused = (
                    physical_audit(z)
                )
            audits.clear()
        # A candidate that fails its audit after the tightenings is a failure,
        # as in the coordinate route and HEX8: no other Krylov method follows.
        correction_iterations = 0
        if not np.isfinite(norm) or norm > threshold:
            raise LinearSolveError(
                "Displacement PCG failed the original-equation residual audit",
                status="fresh_residual_rejected",
                iterations=iterations,
            )
        retain_host = _retain_host_solution(pb, shape, dtype)
        if retain_host and isinstance(result.val, cp.ndarray):
            memory = cp.cuda.alloc_pinned_memory(result.val.nbytes)
            host = np.frombuffer(memory, result.val.dtype, result.val.size).reshape(
                result.val.shape
            )
            result.val.get(out=host)
            result.val = host
        elif not retain_host and isinstance(result.val, np.ndarray):
            result.val = cp.asarray(result.val)
        if retain_host:
            result.cufhom_host_energy_only = True
        info = dict(
            kit=iterations + correction_iterations,
            norm_res=norm,
            norm_res_log=[norm],
            restarts=stats["restarts"] + int(correction_iterations > 0),
            recursive_norm_res=stats.get("recursive_norm_res"),
            replacements=stats["replacements"],
            candidate_checks=stats["candidate_checks"],
            physical_rhs_host=isinstance(physical_rhs_spectrum, np.ndarray)
            if prepared is not None
            else False,
            retained_solution_host=retain_host,
            coordinate_iterations=iterations,
            correction_iterations=correction_iterations,
            linear_final_strain_reuse=strain_reused,
            linear_final_audit_reuse=stress_reused,
            linear_final_spectrum_reuse=spectrum_reused,
            physical_audit_tightenings=tightenings,
            physical_audit_ratios=audit_ratios,
            physical_audit_margin=audit_margin,
            linear_fused_fft=pipeline is not None,
            linear_coordinate_rhs=prepared is not None,
            linear_coordinate_layout="tiled" if Q.tile is not None else "production",
            linear_fused_fft_config=dict(pipeline.config)
            if pipeline is not None
            else None,
            warm_start_requested=x0 is not None,
            warm_start_accepted=initial is not None,
            initial_norm_res_rel=(
                float(initial_norm / rhs_norm)
                if initial_norm is not None and rhs_norm > 0
                else None
            ),
            # G0 = R^T K0^+ R (volume-averaged energy) = r_z.r_z/N for
            # the accepted coordinates of this route's own operator.
            energy_G0=(
                energy_rr[0] / volume
                if energy_rr[0] is not None
                and correction_iterations == 0
                and initial is None
                else None
            ),
        )
        _finalize_cg_info(info, rhs_norm, threshold, par["maxiter"])
        info.update(
            true_norm_res_rel=info["norm_res_rel"], true_residual_converged=True
        )
        if initial is None and correction_iterations == 0 and not retain_host:
            # The returned strain is exactly Q.strain(z) (same plan, same
            # rounding), so a caller short of memory may keep only z.
            # Graph CG reuses x across loads. Compressed snapshots must own
            # their coordinates before the next load overwrites that buffer.
            if (
                pipeline is not None
                and pipeline.vectors is not None
                and z is pipeline.vectors[0]
            ):
                z = z.copy()
            result.cufhom_coordinates = (Q, z)
        return result, info

    def build_rhs(template):
        if pb.solve.get("initial_solution_fields") is not None or pb.solve.get(
            "profile_timing", False
        ):
            return None
        P = MixedProjector(tuple(template.N), tuple(pb.Y), np.ones(6, bool), cp)
        Q = SymmetricCoordinates(P, bulk, shear)
        if not supports_energy(Q, descriptor):
            return None
        if working is not None:
            # The mixed route evaluates its load in total-strain form.
            result = template.copy(
                val=cp.broadcast_to(
                    cp.zeros((), template.val.dtype), template.val.shape
                )
            )
            result.cufhom_mixed_macro = cp.asnumpy(
                template.val.reshape(6, -1)[:, 0]
            ).astype(np.float64)
            return result
        return _build_coordinate_rhs(template, A, Q, _solve_pipeline(pb, Q, descriptor))

    solve.build_rhs = build_rhs
    solve.capacity_engine = capacity_engine
    return solve


def _real_dot(a, b):
    """Accumulate the real Hermitian product in FP64 without conversion copies.

    Re(a^H b) accumulated in FP64, in chunks: a complex64/complex128 on the
    device or in (pinned) host memory, b complex64/complex128 on the device;
    one pass per chunk, no conversion copies.
    """
    total = 0.0
    for start in range(0, a.size, 1 << 24):
        x, y = cp.asarray(a[start : start + (1 << 24)]), b[start : start + (1 << 24)]
        if x.dtype == cp.complex64 and y.dtype == cp.complex128:
            x, y = y, x  # the products are symmetric
        name = (
            "mixed_dot"
            if x.dtype == cp.complex64
            else "mixed_dot64"
            if y.dtype == cp.complex64
            else "mixed_dot128"
        )
        blocks = reduction_blocks(x.size)
        partial = cp.empty(blocks, cp.float64)
        _mixed_kernel(name)((blocks,), (256,), (x, y, partial, np.int64(x.size)))
        total += float(cp.sum(partial, dtype=cp.float64))
    return total


def homogenize_gani_mixed(
    material,
    lengths,
    reference,
    *,
    rtol=1e-6,
    atol=0.0,
    maxiter=3000,
    energy=None,
    residency="device",
    evaluation="df64",
    field_consumer=None,
    configuration="mixed",
):
    """Six unit Mandel loads of the GaNi coordinate route with identity assembly.

    For b <= a, C[a,b] = <sigma_a>_b - Re(z_b^H r_a)/N with r_a the fresh
    residual of load a (stored in the working precision): the field energy of
    the compatible solutions, with no strain field reconstructed or retained.
    Earlier coordinates are kept in FP32 (they enter only the small product
    with a residual), on the device when they fit, otherwise in pinned host
    memory. material is an IndexedMaterial (FP64 for the mixed and FP64
    configurations, FP32 for the FP32 one; see mixed_load); reference an
    isotropic C0; residency one of RESIDENCIES (mixed_residency; the host
    levels retain earlier coordinates on the host); evaluation is the
    arithmetic of the fused residual evaluations ('df64' or 'float64').
    """
    from types import SimpleNamespace

    from cufhom_core.tensors import Tensor

    from .common import isotropic_reference

    energy = energy_tolerance(energy)
    shape = tuple(int(n) for n in material.shape)
    _, bulk, shear = isotropic_reference(material.stiffness, reference)
    descriptor = Tensor(
        name="A_GaNi",
        val=cp.asarray(material.index, cp.int32)[None],
        order=1,
        N=np.array(shape),
        Y=np.asarray(lengths, float),
        multype="sym21_indexed",
        Fourier=False,
        origin=0,
        fft_form="r",
    )
    descriptor.material_table = cp.asarray(material.packed, cp.float64)
    pb = SimpleNamespace(Y=np.asarray(lengths, float), solver={}, solve={})
    par = dict(rtol=rtol, atol=atol, maxiter=maxiter)
    previous = core_fft.FFT_BACKEND
    core_fft.set_fft_backend("cupy")
    core_fft.set_cupy_plan_mode("manual", preserve_cache=True)
    try:
        return _six_mixed_loads(
            material,
            descriptor,
            pb,
            bulk,
            shear,
            shape,
            par,
            energy,
            residency,
            evaluation,
            field_consumer,
            configuration,
        )
    finally:
        core_fft.set_fft_backend(previous)


def _six_mixed_loads(
    material,
    descriptor,
    pb,
    bulk,
    shear,
    shape,
    par,
    energy,
    residency,
    evaluation,
    field_consumer=None,
    configuration="mixed",
):
    import cupyx

    from .common import device_free_bytes

    volume = int(np.prod(shape))
    retained, infos = [], []
    stress, correction = np.empty((6, 6)), np.zeros((6, 6))
    slots, on_device = (
        None,
        0,
    )  # earlier solutions the device keeps; the others in pinned host memory
    for a, macro in enumerate(np.eye(6)):
        try:
            out = mixed_load(
                pb,
                descriptor,
                bulk,
                shear,
                shape,
                macro,
                par,
                energy,
                residency=residency,
                evaluation=evaluation,
                keep_strain=field_consumer is not None,
                configuration=configuration,
            )
        except LinearSolveError as error:
            error.diagnostics.update(
                load_id=a,
                accepted=False,
                problem_precision="float32"
                if configuration == "float32"
                else "float64",
                working_precision="float64"
                if configuration == "float64"
                else "float32",
            )
            raise
        if field_consumer is not None:
            field_consumer(
                np.zeros((6, *shape))
                if out["strain"] is None
                else cp.asnumpy(out["strain"])
            )
        stress[:, a] = out["stress"]
        r = out["residual"]
        for b, zb in enumerate(retained):
            correction[a, b] = correction[b, a] = _real_dot(zb, r) / volume
        correction[a, a] = _solution_dot(out["z"], r) / volume
        if a < 5 and isinstance(out["z"], _HostVector):
            retained.append(out["z"].rounded())
            slots = 0
        elif a < 5:
            z32 = out["z"].astype(
                cp.complex64, copy=False
            )  # an FP32 solution is retained as is
            if slots is None:  # as many FP32 copies as fit beside the working set
                # The estimate already counts this load's z, r and ids, which
                # are allocated now (z is released or becomes the first copy).
                held = (
                    out["z"].nbytes
                    + r.nbytes
                    + (6 if descriptor.val is not None else 2) * volume
                )
                room = (
                    device_free_bytes()
                    + held
                    - _RESERVE_BYTES
                    - mixed_working_bytes(
                        shape, "device", configuration, energy is not None
                    )
                )
                slots = int(max(0, min(5, room // z32.nbytes)))
            if len(retained) >= slots:
                pinned = cupyx.empty_pinned(z32.shape, z32.dtype)
                z32.get(out=pinned)
                retained.append(pinned)
            else:
                retained.append(z32)
                on_device += 1
            del z32
        out["info"]["mean_stress"] = out["stress"].tolist()
        infos.append(out["info"])
        del out, r
    lower = np.tril(stress.T - correction)
    C = lower + np.tril(lower, -1).T
    return dict(
        C=C,
        C_stress=stress,
        energy_correction=correction,
        infos=infos,
        retained_device=on_device,
        retained="device"
        if on_device == len(retained)
        else "host"
        if on_device == 0
        else "device+host",
    )
