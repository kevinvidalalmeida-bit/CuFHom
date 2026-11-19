"""Bounded-device FP64 CG for indexed, energy-only elasticity calls.

FFT actions and vector arithmetic stay in FP64 on GPU. Compact routes keep
residual and direction resident; strain keeps the direction resident. The
remaining state lives in pinned host storage. This is a capacity fallback, not a resident
performance claim. The fused and library operators remain distinct. A fresh
library-transform equilibrium residual alone permits acceptance.
"""

import os
import sys
import time

import numpy as np
from cufhom.fft.operators import MixedProjector, hermitian_plane
from cufhom_core.general.cupy_capacity import _device_capacity, pinned_array
from cufhom_core.general.cupy_strain_capacity import kernel as padded_material
from cufhom_core.general.cupy_strain_capacity import plans
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp
from cufhom_core.tensors.objects import (
    _get_cupy_indexed_sym21_kernel,
    elasticity_direct_array,
)

from .krylov import _coordinate_kernel

CHUNK = 1_048_576
_INPLACE = None
_PADDED_KERNELS = None


def padded_kernels():
    """Return cached gather, scatter and direction kernels for padded storage."""
    global _PADDED_KERNELS
    if _PADDED_KERNELS is None:
        inputs = "raw float64 a, int64 start, int64 spatial, int64 last, int64 outlast, int64 outspatial"
        offset = "long j=i+start; long s=j%spatial; long k=(j/spatial)*outspatial+(s/last)*outlast+s%last;"
        gather = cp.ElementwiseKernel(
            inputs, "float64 value", offset + " value=a[k];", "capacity_padded_gather"
        )
        scatter = cp.ElementwiseKernel(
            "float64 value, " + inputs.split(", ", 1)[1],
            "raw float64 a",
            offset + " a[k]=value;",
            "capacity_padded_scatter",
        )
        direction = cp.ElementwiseKernel(
            "float64 beta, " + inputs,
            "float64 p",
            offset + " p=a[k]+beta*p;",
            "capacity_padded_direction",
        )
        _PADDED_KERNELS = (gather, scatter, direction)
    return _PADDED_KERNELS


def progress(count, rr, initial):
    """Emit optional capacity-solver progress every ten CG updates."""
    if os.environ.get("CUFHOM_CAPACITY_PROGRESS") == "1" and count % 10 == 0:
        print(
            f"CAPACITY CG update={count} relative-coordinate-residual={np.sqrt(rr / max(initial, np.finfo(float).tiny)):.3e}",
            file=sys.stderr,
            flush=True,
        )


def required(shape, dtype):
    # Keep existing <=384^3 observations on their original path. Beyond this
    # estimate, even the reduced native working sets need a different policy.
    """Test whether the estimated FP64 working set requires the capacity path."""
    return (
        np.dtype(dtype) == np.dtype("float64")
        and (27 * 8 + 4) * int(np.prod(shape))
        > _device_capacity(cp.cuda.runtime.getDevice()) - 512 * 2**20
    )


def eligible(pb, A, shape, dtype):
    """Check that the indexed material and requested outputs support host storage."""
    s = pb.solve
    return (
        cp is not None
        and isinstance(getattr(A, "val", None), cp.ndarray)
        and getattr(A, "multype", None) in ("sym21_indexed", "indexed_sym21")
        and len(shape) == 3
        and required(shape, dtype)
        and np.dtype(A.material_table.dtype) == np.dtype(dtype)
        and not s.get("store_solution_fields", True)
        and not s.get("partial_load_output", False)
        and s.get("initial_solution_fields") is None
        and s.get("affine_sensitivity_cfields") is None
        and not callable(s.get("stress_field_consumer"))
        and not callable(s.get("solution_field_consumer"))
        and str(s.get("postprocess_assembly", "gemm")) == "fused"
    )


def inplace_material():
    """Return the cached constitutive kernel for aliased padded FP64 fields."""
    global _INPLACE
    if _INPLACE is None:
        code = _get_cupy_indexed_sym21_kernel(np.float64).code
        code = code.replace("void indexed_sym21_apply(", "void padded_inplace(")
        code = code.replace(
            "const long materials) {",
            "const long materials, const long last, const long outlast, const long outspatial) {",
        )
        code = code.replace(
            "const long base = b * spatial + s;",
            "const long base = b * outspatial + (s / last) * outlast + s % last;",
        )
        code = code.replace(
            "const long stride = batch * spatial;",
            "const long stride = batch * outspatial;",
        )
        # All six inputs are loaded before any output; exact aliasing is safe.
        code = code.replace("__restrict__", "")
        _INPLACE = cp.RawKernel(code, "padded_inplace")
    return _INPLACE


class ActiveCapacity:
    """Apply FP64 elasticity with bounded device workspaces and pinned host state."""

    def __init__(self, A, shape, lengths, Q=None, pipeline=None):
        self.A = A
        self.shape = tuple(int(n) for n in shape)
        self.N = int(np.prod(shape))
        self.padded = self.shape[:-1] + (2 * (self.shape[-1] // 2 + 1),)
        self.P = MixedProjector(self.shape, tuple(lengths), np.ones(6, bool), cp)
        self.Q = Q
        self.pipeline = pipeline
        self.coordinate = Q is not None
        self.transfers = 0
        self.streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
        self.workspace = None  # optional uint8 device memory the work field views

    # The library plans (and their cuFFT work areas) exist only once a
    # library transform is needed.
    @property
    def f(self):
        """Return the lazily created forward FP64 FFT plan."""
        return plans(self.shape, np.float64)[0]

    @property
    def b(self):
        """Return the lazily created inverse FP64 FFT plan."""
        return plans(self.shape, np.float64)[1]

    def raw_padded(self, value):
        """View an allocation as a flat padded six-component FP64 work field."""
        return cp.ndarray(
            (6 * int(np.prod(self.padded)),), dtype=cp.float64, memptr=value.data
        )

    def padded_args(self, raw, start=0):
        """Return indexing arguments for a padded-field kernel launch."""
        return (
            raw,
            np.int64(start),
            np.int64(self.N),
            np.int64(self.shape[-1]),
            np.int64(self.padded[-1]),
            np.int64(np.prod(self.padded)),
        )

    def join_streams(self):
        """Make the current stream wait for the capacity transfer streams."""
        origin = cp.cuda.get_current_stream()
        for stream in self.streams:
            origin.wait_event(stream.record())

    def work(self):
        """Return aliased padded, complex-spectrum and physical-field work views."""
        from .fused import _view

        w = _view((6,) + self.padded, cp.float64, self.workspace)
        return w, w.view(cp.complex128), w[..., : self.shape[-1]]

    def forward(self, real, spectrum):
        """Transform six physical components into the supplied spectral workspace."""
        for c in range(6):
            self.f.fft(real[c], spectrum[c], cp.cuda.cufft.CUFFT_FORWARD)

    def inverse(self, spectrum, real):
        """Apply six unnormalized inverse transforms into the physical workspace."""
        for c in range(6):
            self.b.fft(spectrum[c], real[c], cp.cuda.cufft.CUFFT_INVERSE)

    def map(self, source, output, adjoint):
        """Apply the coordinate map or its adjoint into a supplied output buffer."""
        q = self.Q
        mode = "map" if q.tile is None else "tiled"
        args = q._args(source, output, adjoint)
        if q.tile is not None:
            args += (np.int32(q.tile),)
        _coordinate_kernel(source.dtype, mode)(((q.count + 255) // 256,), (256,), args)
        if q.tile is not None and not adjoint:
            hermitian_plane(
                output, self.shape[-1], packed=True
            )  # split the packed k2 = 0 plane

    def slabs(self):
        """Yield bounded contiguous slabs along the first spatial axis."""
        plane = self.shape[1] * self.shape[2]
        width = max(1, CHUNK // plane)
        for a in range(0, self.shape[0], width):
            yield a, min(self.shape[0], a + width)

    def upload(self, host, real, mean=None):
        """Copy host slabs into a device field and optionally add a uniform mean."""
        for a, b in self.slabs():
            v = cp.asarray(np.ascontiguousarray(host[:, a:b]))
            if mean is not None:
                v += cp.asarray(mean)[:, None, None, None]
            cp.copyto(real[:, a:b], v)
            self.transfers += v.nbytes
            del v

    def download(self, real):
        """Copy a device field into pinned host storage and account for transfers."""
        host = pinned_array((6,) + self.shape, np.float64)
        for a, b in self.slabs():
            v = cp.ascontiguousarray(real[:, a:b])
            h = v.get()
            host[:, a:b] = h
            self.transfers += v.nbytes
            del v, h
        return host

    def constitutive(self, work):
        """Apply the indexed material in place to the padded work allocation."""
        inplace_material()(
            ((self.N + 255) // 256,),
            (256,),
            (
                self.A.val,
                self.A.material_table,
                work,
                work,
                np.int64(1),
                np.int64(self.N),
                np.int64(self.N),
                np.int64(self.A.material_table.shape[0]),
                np.int64(self.shape[-1]),
                np.int64(self.padded[-1]),
                np.int64(np.prod(self.padded)),
            ),
        )

    def project(self, spectrum):
        """Project a spectral field onto the compatible strain subspace in place."""
        if elasticity_direct_array(self.P.descriptor, spectrum, out=spectrum) is None:
            raise RuntimeError("FP64 capacity projection unavailable")

    def strain(self, z):
        """Return a host strain field, decoding compact coordinates when required."""
        if not self.coordinate:
            return z
        w, h, v = self.work()
        d = cp.asarray(z)
        try:
            self.map(d, h, False)
            del d
            self.inverse(h, v)
            return self.download(v)
        finally:
            del w, h, v

    def physical(self, strain, mean, *, rhs=False):
        """Independent library FFT equilibrium norm and optionally compact RHS."""
        from .coordinates import _spectral_distance

        w, h, v = self.work()
        try:
            if strain is None:
                v[:] = cp.asarray(mean)[:, None, None, None]
            else:
                self.upload(strain, v, mean)
            self.constitutive(w)
            self.forward(v, h)
            self.project(h)
            norm = np.sqrt(_spectral_distance(h, self.shape)) / self.N
            if not rhs:
                return float(norm)
            if self.coordinate:
                out = cp.empty(self.Q.size, cp.complex128)
                # QR* G = QR*. The normalized spectrum is still the ordinary
                # forward transform, so the coordinate adjoint needs no 1/N.
                self.map(h, out, True)
                out *= -1
                host = pinned_array(out.shape, out.dtype)
                out.get(out=host)
                return float(norm), host
            h *= -1.0 / self.N
            self.inverse(h, v)
            return float(norm), self.download(v)
        finally:
            del w, h, v

    def action(self, p):
        """Apply the elasticity operator using the selected capacity workspace."""
        if self.pipeline is not None:
            out = self.pipeline.matvec(p)
            self.pipeline.release()
            return out
        w, h, v = self.work()
        try:
            if self.coordinate:
                self.map(p, h, False)
                self.inverse(h, v)
                self.constitutive(w)
                self.forward(v, h)
                # Production coordinates and spectra are component-major.
                # Each thread reads all six values at its own frequency before
                # replacing the first three. No thread reads another mode.
                out = h.reshape(-1)[: self.Q.size]
                self.map(h, out, True)
                return out
            padded_material(np.float64)(
                ((self.N + 255) // 256,),
                (256,),
                (
                    self.A.val,
                    self.A.material_table,
                    p,
                    w,
                    np.int64(1),
                    np.int64(self.N),
                    np.int64(self.N),
                    np.int64(self.A.material_table.shape[0]),
                    np.int64(self.shape[-1]),
                    np.int64(self.padded[-1]),
                    np.int64(np.prod(self.padded)),
                ),
            )
            self.forward(v, h)
            self.project(h)
            h *= 1.0 / self.N
            self.inverse(h, v)
            return v
        finally:
            del w, h, v

    def project_host(self, host):
        """Project a host field through the independent library FFT path."""
        w, h, v = self.work()
        try:
            self.upload(host, v)
            self.forward(v, h)
            self.project(h)
            h *= 1.0 / self.N
            self.inverse(h, v)
            return self.download(v)
        finally:
            del w, h, v

    def solve(self, mean, par):
        """Solve one macroscopic load with FP64 capacity CG and fresh acceptance."""
        if self.coordinate:
            return self.solve_compact(mean, par)
        from cufhom_core.general.solver import _cupy_kernels, _finalize_cg_info

        started = time.perf_counter()
        self.transfers = 0
        rhsnorm, f = self.physical(None, mean, rhs=True)
        r = pinned_array(f.shape, f.dtype)
        np.copyto(r, f)
        x = pinned_array(f.shape, f.dtype)
        x.fill(0)
        p = cp.asarray(r)
        size = p.view(cp.float64).size
        kernels = _cupy_kernels()
        capacity = min(CHUNK, size)
        stages = [
            tuple(cp.empty(capacity, cp.float64) for _ in range(3))
            for _ in self.streams
        ]
        xs, rs, aps = stages[0]
        partial = cp.empty((size + capacity - 1) // capacity, cp.float64)
        gather, scatter, direction = padded_kernels()
        rf = r.view(np.float64).ravel()
        xf = x.view(np.float64).ravel()
        ff = f.view(np.float64).ravel()
        pf = p.view(cp.float64).ravel()

        def dot(a, b):
            for s in range(0, size, capacity):
                e = min(size, s + capacity)
                n = e - s
                aa = a[s:e] if isinstance(a, cp.ndarray) else rs[:n]
                bb = b[s:e] if isinstance(b, cp.ndarray) else aps[:n]
                if not isinstance(a, cp.ndarray):
                    aa.set(a[s:e])
                    self.transfers += n * 8
                if not isinstance(b, cp.ndarray):
                    bb.set(b[s:e])
                    self.transfers += n * 8
                partial[s // capacity] = cp.sum(aa * bb, dtype=cp.float64)
            return float(cp.sum(partial, dtype=cp.float64))

        rr = dot(pf, pf)
        initial = rr
        threshold = max(float(par.get("atol", 0)), float(par["rtol"]) * rhsnorm)
        limit = (threshold / rhsnorm) ** 2 * initial if rhsnorm else 0.0
        count = 0
        restarts = 0
        physical = rhsnorm
        accepted = None
        maxiter = int(par.get("maxiter", 1000))
        while True:
            if rr <= limit or count >= maxiter:
                del p, pf
                strain = self.strain(x)
                if not self.coordinate:
                    strain = self.project_host(strain)
                physical = self.physical(strain, mean)
                if np.isfinite(physical) and physical <= threshold:
                    accepted = strain
                    break
                if count >= maxiter or restarts >= 3:
                    raise RuntimeError(
                        f"FP64 capacity CG failed physical audit: {physical:g} > {threshold:g}; updates={count}"
                    )
                del strain
                p = cp.asarray(x)
                ap = self.action(p)
                del p
                raw = self.raw_padded(ap)
                for s in range(0, size, capacity):
                    e = min(size, s + capacity)
                    n = e - s
                    rs[:n].set(ff[s:e])
                    gather(*self.padded_args(raw, s), aps[:n])
                    rs[:n] -= aps[:n]
                    rs[:n].get(out=rf[s:e])
                del ap, raw
                p = cp.asarray(r)
                pf = p.view(cp.float64).ravel()
                rr = dot(pf, pf)
                limit *= min(0.25, (0.8 * threshold / physical) ** 2)
                restarts += 1
                continue
            ap = self.action(p)
            raw = self.raw_padded(ap)
            for s in range(0, size, capacity):
                e = min(size, s + capacity)
                n = e - s
                gather(*self.padded_args(raw, s), aps[:n])
                partial[s // capacity] = cp.sum(pf[s:e] * aps[:n], dtype=cp.float64)
            denominator = float(cp.sum(partial, dtype=cp.float64))
            if not np.isfinite(denominator) or denominator <= 0:
                raise RuntimeError(
                    "FP64 capacity CG has a nonpositive/nonfinite denominator"
                )
            alpha = rr / denominator
            ready = cp.cuda.get_current_stream().record()
            for stream in self.streams:
                stream.wait_event(ready)
            for index, s in enumerate(range(0, size, capacity)):
                e = min(size, s + capacity)
                n = e - s
                stream = self.streams[index % 2]
                xs, rs, aps = stages[index % 2]
                with stream:
                    xs[:n].set(xf[s:e], stream=stream)
                    rs[:n].set(rf[s:e], stream=stream)
                    gather(*self.padded_args(raw, s), aps[:n])
                    kernels["xr"](np.float64(alpha), pf[s:e], aps[:n], xs[:n], rs[:n])
                    partial[index] = cp.sum(rs[:n] * rs[:n], dtype=cp.float64)
                    scatter(rs[:n], *self.padded_args(raw, s)[1:], raw)
                    xs[:n].get(out=xf[s:e], stream=stream, blocking=False)
                    rs[:n].get(out=rf[s:e], stream=stream, blocking=False)
                self.transfers += n * 32
            self.join_streams()
            new = float(cp.sum(partial, dtype=cp.float64))
            beta = new / rr
            rr = new
            count += 1
            progress(count, rr, initial)
            if not np.isfinite(rr):
                raise RuntimeError("FP64 capacity CG has a nonfinite residual")
            direction(np.float64(beta), *self.padded_args(raw), pf)
            del ap, raw
        info = dict(
            kit=count,
            norm_res=physical,
            true_norm_res=physical,
            restarts=restarts,
            replacements=0,
            true_norm_res_rel=physical / max(rhsnorm, np.finfo(float).tiny),
            true_residual_converged=True,
            linear_fused_fft=self.pipeline is not None,
            coordinate_iterations=count if self.coordinate else 0,
            correction_iterations=0,
            host_cg_iterate=True,
            resident_cg_iterate=False,
            active_state_strategy="pinned_x_r_gpu_direction",
            active_state_transfer_bytes=self.transfers,
            physical_audit_tightenings=restarts,
            retained_solution_host=True,
            linear_coordinate_layout="tiled"
            if self.pipeline is not None
            else "production",
            time=time.perf_counter() - started,
        )
        _finalize_cg_info(info, rhsnorm, threshold, maxiter)
        result = Tensor(
            name="accepted_capacity_fluctuation",
            val=accepted,
            order=1,
            N=self.shape,
            Y=self.P.lengths,
            Fourier=False,
            fft_form="r",
        )
        result.cufhom_host_energy_only = True
        result.cufhom_deferred_mean = np.array(mean, copy=True)
        return result, info

    def solve_compact(self, mean, par):
        """Retain r/p in GPU; stream only x, avoiding an explicit fused Hp."""
        from cufhom_core.general.solver import _cupy_kernels, _finalize_cg_info

        started = time.perf_counter()
        self.transfers = 0
        rhsnorm, f = self.physical(None, mean, rhs=True)
        x = pinned_array(f.shape, f.dtype)
        x.fill(0)
        r = cp.asarray(f)
        p = r.copy()
        rf = r.view(cp.float64).ravel()
        pf = p.view(cp.float64).ravel()
        xf = x.view(np.float64).ravel()
        size = rf.size
        capacity = min(CHUNK, size)
        stages = [cp.empty(capacity, cp.float64) for _ in self.streams]
        stage = stages[0]
        kernels = _cupy_kernels()
        partial = cp.empty((size + capacity - 1) // capacity, cp.float64)

        def dot(a, b):
            for s in range(0, size, capacity):
                partial[s // capacity] = cp.sum(
                    a[s : s + capacity] * b[s : s + capacity], dtype=cp.float64
                )
            return float(cp.sum(partial, dtype=cp.float64))

        rr = dot(rf, rf)
        initial = rr
        threshold = max(float(par.get("atol", 0)), float(par["rtol"]) * rhsnorm)
        limit = (threshold / rhsnorm) ** 2 * initial if rhsnorm else 0.0
        count = 0
        restarts = 0
        maxiter = int(par.get("maxiter", 1000))
        physical = rhsnorm
        while True:
            if rr <= limit or count >= maxiter:
                del p, pf, r, rf
                if self.pipeline is not None:
                    self.pipeline.release()
                strain = self.strain(x)
                physical = self.physical(strain, mean)
                if np.isfinite(physical) and physical <= threshold:
                    break
                del strain
                if count >= maxiter or restarts >= 3:
                    raise RuntimeError(
                        f"FP64 capacity CG failed physical audit: {physical:g} > {threshold:g}; updates={count}"
                    )
                z = cp.asarray(x)
                ap = self.action(z)
                del z
                # Own exactly three components for the next iteration: a
                # library adjoint may currently alias a six-component block.
                r = cp.empty_like(ap)
                rf = r.view(cp.float64).ravel()
                af = ap.view(cp.float64).ravel()
                ff = f.view(np.float64).ravel()
                for s in range(0, size, capacity):
                    e = min(size, s + capacity)
                    n = e - s
                    stage[:n].set(ff[s:e])
                    cp.subtract(stage[:n], af[s:e], out=rf[s:e])
                    self.transfers += n * 8
                del ap, af
                p = r.copy()
                pf = p.view(cp.float64).ravel()
                rr = dot(rf, rf)
                limit *= min(0.25, (0.8 * threshold / physical) ** 2)
                restarts += 1
                continue
            if self.pipeline is not None:
                denominator = self.pipeline.apply(p)
                denom = float(denominator)
                if not np.isfinite(denom) or denom <= 0:
                    raise RuntimeError(
                        "FP64 capacity CG has a nonpositive/nonfinite denominator"
                    )
                new = self.pipeline.update(p, rr, denominator, p, r)
                # The actual x update is streamed below, rather than deferred
                # to a full GPU x. Never let settle() modify the placeholder p.
                self.pipeline.pending = None
                alpha = float(self.pipeline.step)
            else:
                ap = self.action(p)
                af = ap.view(cp.float64).ravel()
                denom = dot(pf, af)
                if not np.isfinite(denom) or denom <= 0:
                    raise RuntimeError(
                        "FP64 capacity CG has a nonpositive/nonfinite denominator"
                    )
                alpha = rr / denom
                kernels["r"](np.float64(alpha), af, rf)
                new = dot(rf, rf)
                del ap, af
            ready = cp.cuda.get_current_stream().record()
            for stream in self.streams:
                stream.wait_event(ready)
            for index, s in enumerate(range(0, size, capacity)):
                e = min(size, s + capacity)
                n = e - s
                stage = stages[index % 2]
                stream = self.streams[index % 2]
                with stream:
                    stage[:n].set(xf[s:e], stream=stream)
                    kernels["x"](np.float64(alpha), pf[s:e], stage[:n])
                    stage[:n].get(out=xf[s:e], stream=stream, blocking=False)
                self.transfers += n * 16
            self.join_streams()
            beta = new / rr
            rr = new
            count += 1
            progress(count, rr, initial)
            if not np.isfinite(rr):
                raise RuntimeError("FP64 capacity CG has a nonfinite residual")
            kernels["p"](np.float64(beta), rf, pf)
        info = dict(
            kit=count,
            norm_res=physical,
            true_norm_res=physical,
            restarts=restarts,
            replacements=0,
            true_norm_res_rel=physical / max(rhsnorm, np.finfo(float).tiny),
            true_residual_converged=True,
            linear_fused_fft=self.pipeline is not None,
            coordinate_iterations=count,
            correction_iterations=0,
            host_cg_iterate=True,
            resident_cg_iterate=False,
            active_state_strategy="pinned_x_gpu_residual_direction",
            active_state_transfer_bytes=self.transfers,
            physical_audit_tightenings=restarts,
            retained_solution_host=True,
            linear_coordinate_layout="tiled"
            if self.pipeline is not None
            else "production",
            time=time.perf_counter() - started,
        )
        _finalize_cg_info(info, rhsnorm, threshold, maxiter)
        result = Tensor(
            name="accepted_capacity_fluctuation",
            val=strain,
            order=1,
            N=self.shape,
            Y=self.P.lengths,
            Fourier=False,
            fft_form="r",
        )
        result.cufhom_host_energy_only = True
        result.cufhom_deferred_mean = np.array(mean, copy=True)
        return result, info
