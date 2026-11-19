"""GPU-only periodic implicit-gradient regularization for a matrix phase.

The discrete equation is ``A h = source``, with ``A = I + G* a G`` and
``a = length_scale**2 * matrix_mask``. G is the real Fourier gradient and
G* its adjoint (minus divergence). Thus
``<v, A v> = ||v||**2 + sum(a * |G v|**2) > 0`` for nonzero v, even when
a is zero in the fibers. No division by the phase coefficient is needed.

On even grids ALL modes on ANY Nyquist plane have zero gradient, matching
the convention of cufhom.fft.operators.MixedProjector. Their mass term is
retained: these modes are NOT smoothed or removed from the solution. The
homogeneous FFT inverse uses this same discrete symbol, not a continuum
Laplacian with a different Nyquist treatment.

Positive definiteness is an energy property, not a discrete maximum
principle. Spectral collocation with discontinuous phases can produce
oscillations and negative values even from a nonnegative source. Neither
the source nor the solution is clamped. In particular, zero source and
zero diffusion coefficient in fibers do NOT force the discrete solution
to vanish there. The material/coupling must assess admissibility itself.

The heterogeneous Helmholtz construction and constant-coefficient FFT
preconditioning follow Magri et al. (2021), arXiv:2103.04770, Sec. 3.2.3:
https://arxiv.org/abs/2103.04770 . Here length_scale is a physical length;
the coefficient in the PDE is its square times the binary phase mask.
This module supplies regularization only, not a damage evolution law.
"""

import math
import operator
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
from cufhom.fft.krylov import NonlinearConvergenceError
from cufhom_core.tensors.fft import cp

_KERNELS = None


def _kernels():
    """Lazy compilation; importing/collecting tests does not launch CUDA."""
    global _KERNELS
    if _KERNELS is None:
        _KERNELS = SimpleNamespace(
            dot=cp.ReductionKernel(
                "T x, T y",
                "float64 result",
                "(double)x * (double)y",
                "a + b",
                "result = a",
                "0",
                "nonlocal_helmholtz_dot",
                reduce_type="double",
            ),
            source_flags=cp.ReductionKernel(
                "T source, bool mask",
                "int32 result",
                "(!isfinite(source) ? 1 : 0) | (!mask && source != 0 ? 2 : 0)",
                "a | b",
                "result = a",
                "0",
                "nonlocal_helmholtz_source_flags",
            ),
            gradient=cp.ElementwiseKernel(
                "T spectrum, R k, bool valid",
                "T result",
                "result = valid ? T(0, 1) * k * spectrum : T(0, 0);",
                "nonlocal_helmholtz_gradient",
            ),
            adjoint_add=cp.ElementwiseKernel(
                "T flux, R k, bool valid",
                "T result",
                "if (valid) result -= T(0, 1) * k * flux;",
                "nonlocal_helmholtz_adjoint_add",
            ),
            update=cp.ElementwiseKernel(
                "T p, T ap, float64 alpha",
                "T x, T r",
                "x += (T)alpha * p; r -= (T)alpha * ap;",
                "nonlocal_helmholtz_update",
            ),
            direction=cp.ElementwiseKernel(
                "T z, float64 beta",
                "T p",
                "p = z + (T)beta * p;",
                "nonlocal_helmholtz_direction",
            ),
        )
    return _KERNELS


class NonlocalConvergenceError(NonlinearConvergenceError):
    """An unconverged field is never returned; diagnostics describe failure."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class NonlocalHelmholtz:
    """Reusable matrix-free Helmholtz operator on a three-dimensional GPU grid.

    Parameters
    ----------
    shape, lengths : sequences of length 3
        Voxel counts and positive physical periodic cell lengths.
    length_scale : float
        Nonnegative scalar length in the same units as lengths. Zero gives
        the identity. Spatial variation is supplied by matrix_mask only.
    matrix_mask : array
        Binary 1/True for matrix, 0/False for fiber. A private GPU copy is
        retained; host masks may be transferred at construction.
    xp : module
        Must be CuPy. There is no CPU solve or FFT fallback.
    dtype : float32 or float64
        Field/FFT precision. Products and reductions for CG are FP64.
    timers : optional
        An object with ``measure(name)`` context managers, such as Timings.

    solve requires CuPy source/initial fields with the configured shape,
    dtype and device. The source MUST be exactly zero in fibers; invalid
    sources are rejected instead of being silently masked. Signed matrix
    sources are allowed by this linear operator.

    Heterogeneous masks use PCG with the fixed FFT inverse symbol
    ``1 / (1 + mean(a) * |k|**2)``. Uniform matrix uses an exact discrete
    FFT inverse. Every accepted solve audits ``||source - A h||`` and
    conserves the global source mean to field precision. Constant-mode
    correction solves the uncoupled zero mode; it is not a positivity clamp.
    No lower bound on rtol is silently imposed; unattainable tolerances raise.

    Fourier/CG work buffers are reused. Axis gradients are evaluated
    sequentially, avoiding storage of a three-component real gradient.
    cuFFT outputs/workspaces still allocate through CuPy's pool/plan cache.
    Only control/diagnostic scalars cross to host. The instance belongs to
    its construction device and is NOT reentrant; use one instance per
    concurrent stream/solve. Returned fields own storage and survive reuse.
    """

    def __init__(
        self,
        shape,
        lengths,
        length_scale,
        matrix_mask,
        xp=cp,
        dtype=np.float32,
        timers=None,
    ):
        if cp is None:
            raise RuntimeError("NonlocalHelmholtz requires CuPy")
        if xp is not cp:
            raise ValueError("NonlocalHelmholtz is GPU-only; xp must be CuPy")
        try:
            self.shape = tuple(operator.index(n) for n in shape)
            self.lengths = tuple(float(v) for v in lengths)
        except (TypeError, ValueError) as error:
            raise ValueError("shape and lengths must be three-dimensional") from error
        if len(self.shape) != 3 or any(n < 1 for n in self.shape):
            raise ValueError("shape must contain three positive integer counts")
        if len(self.lengths) != 3 or any(
            not math.isfinite(v) or v <= 0 for v in self.lengths
        ):
            raise ValueError("lengths must contain three finite positive values")
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        self.length_scale = float(length_scale)
        squared_length = self.length_scale * self.length_scale
        if (
            not math.isfinite(self.length_scale)
            or self.length_scale < 0
            or not math.isfinite(squared_length)
        ):
            raise ValueError("length_scale must be finite and nonnegative")
        with np.errstate(over="ignore", under="ignore"):
            coefficient = self.dtype.type(squared_length)
        if not np.isfinite(coefficient) or (squared_length > 0 and coefficient == 0):
            raise ValueError("length_scale squared is not representable in dtype")
        self.xp, self.timers = xp, timers
        self._device = int(xp.cuda.runtime.getDevice())
        if (
            isinstance(matrix_mask, xp.ndarray)
            and matrix_mask.device.id != self._device
        ):
            raise ValueError("matrix_mask must be on the construction device")
        mask = xp.asarray(matrix_mask)
        if mask.shape != self.shape or mask.dtype.kind not in "buif":
            raise ValueError("matrix_mask must be a real binary array matching shape")
        if not bool(xp.all((mask == 0) | (mask == 1)).item()):
            raise ValueError("matrix_mask must contain only zero and one")
        self._mask = xp.array(mask, dtype=xp.bool_, order="C", copy=True)
        self._coefficient = self._mask.astype(self.dtype) * coefficient
        self._mean_coefficient = float(
            xp.mean(self._coefficient, dtype=xp.float64).item()
        )
        self._constant = bool(xp.all(self._mask).item())
        self._identity = coefficient == 0 or self._mean_coefficient == 0
        self.hshape = self.shape[:-1] + (self.shape[-1] // 2 + 1,)
        self._valid = xp.ones(self.hshape, dtype=xp.bool_)
        self._wave_numbers = []
        k_squared = xp.zeros(self.hshape, dtype=self.dtype)
        for axis, (n, length) in enumerate(zip(self.shape, self.lengths)):
            frequencies = (
                xp.fft.rfftfreq(n, d=length / n)
                if axis == 2
                else xp.fft.fftfreq(n, d=length / n)
            )
            k = (2 * math.pi * frequencies).astype(self.dtype)
            broadcast_shape = tuple(k.size if i == axis else 1 for i in range(3))
            k = k.reshape(broadcast_shape)
            self._wave_numbers.append(k)
            k_squared += k * k
            if n % 2 == 0:
                plane = [slice(None)] * 3
                plane[axis] = n // 2
                self._valid[tuple(plane)] = False
        k_squared *= self._valid
        self._diffusion_symbol = coefficient * k_squared
        self._inverse_symbol = xp.reciprocal(
            1 + self.dtype.type(self._mean_coefficient) * k_squared
        )
        if not bool(xp.isfinite(self._diffusion_symbol).all().item()) or not bool(
            (self._inverse_symbol > 0).all().item()
        ):
            raise ValueError("Fourier symbol is not finite/positive in dtype")
        complex_dtype = (
            xp.complex64 if self.dtype == np.dtype("float32") else xp.complex128
        )
        heterogeneous = not self._constant and not self._identity
        self._derivative_hat = (
            xp.empty(self.hshape, dtype=complex_dtype) if heterogeneous else None
        )
        self._diffusion_hat = (
            xp.empty_like(self._derivative_hat) if heterogeneous else None
        )
        self._r = xp.empty(self.shape, dtype=self.dtype)
        self._z = xp.empty_like(self._r) if heterogeneous else None
        self._p = xp.empty_like(self._r) if heterogeneous else None
        self._ap = xp.empty_like(self._r)
        self._operator_calls = self._preconditioner_calls = 0

    def _measure(self, name):
        return nullcontext() if self.timers is None else self.timers.measure(name)

    def _field(self, value, name):
        if not isinstance(value, self.xp.ndarray):
            raise TypeError(f"{name} must be a CuPy array; no implicit CPU fallback")
        if (
            value.shape != self.shape
            or value.dtype != self.dtype
            or value.device.id != self._device
        ):
            raise ValueError(f"{name} must match the operator shape, dtype and device")
        return self.xp.ascontiguousarray(value)

    def _dot(self, x, y):
        with self._measure("nonlocal_reductions"):
            return float(_kernels().dot(x, y).item())

    def _apply_into(self, field, out):
        """I + G* a G; gradient and adjoint share precisely the same symbol."""
        self._operator_calls += 1
        xp = self.xp
        with self._measure("nonlocal_operator"):
            if self._identity:
                xp.copyto(out, field)
                return
            spectrum = xp.fft.rfftn(field)
            if self._constant:
                spectrum *= self._diffusion_symbol
                diffusion = xp.fft.irfftn(spectrum, s=self.shape)
            else:
                self._diffusion_hat.fill(0)
                kernels = _kernels()
                for k in self._wave_numbers:
                    kernels.gradient(spectrum, k, self._valid, self._derivative_hat)
                    flux = xp.fft.irfftn(self._derivative_hat, s=self.shape)
                    flux *= self._coefficient
                    flux_hat = xp.fft.rfftn(flux)
                    kernels.adjoint_add(flux_hat, k, self._valid, self._diffusion_hat)
                diffusion = xp.fft.irfftn(self._diffusion_hat, s=self.shape)
            xp.add(field, diffusion, out=out)

    def apply(self, field):
        """Return A(field) on GPU without mutating field; useful for audits."""
        with self.xp.cuda.Device(self._device):
            field = self._field(field, "field")
            out = self.xp.empty_like(field)
            self._apply_into(field, out)
            return out

    def _precondition_into(self, field, out):
        self._preconditioner_calls += 1
        with self._measure("nonlocal_preconditioner"):
            spectrum = self.xp.fft.rfftn(field)
            spectrum *= self._inverse_symbol
            self.xp.copyto(out, self.xp.fft.irfftn(spectrum, s=self.shape))

    def _restore_mean(self, field, source_mean):
        # The constant mode has eigenvalue one and does not couple to G.
        delta = source_mean - self.xp.mean(field, dtype=self.xp.float64)
        self.xp.add(field, delta.astype(self.dtype), out=field)

    def _true_residual(self, field, source, source_mean):
        self._restore_mean(field, source_mean)
        self._apply_into(field, self._ap)
        self.xp.subtract(source, self._ap, out=self._r)
        return self._dot(self._r, self._r)

    def _diagnostics(
        self,
        field,
        source_mean,
        rhs_squared,
        residual_squared,
        iterations,
        method,
        rtol,
        converged,
    ):
        mean_source = float(source_mean.item())
        mean_field = float(self.xp.mean(field, dtype=self.xp.float64).item())
        rhs_norm = math.sqrt(rhs_squared)
        residual_norm = (
            math.sqrt(residual_squared) if residual_squared >= 0 else math.nan
        )
        relative = residual_norm / rhs_norm if rhs_norm else residual_norm
        return dict(
            converged=converged,
            method=method,
            iterations=iterations,
            rtol=rtol,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            relative_residual=relative,
            residual=relative,
            mean_source=mean_source,
            mean_field=mean_field,
            mean_error=abs(mean_field - mean_source),
            operator_calls=self._operator_calls,
            preconditioner_calls=self._preconditioner_calls,
            preconditioner_coefficient=self._mean_coefficient,
            dtype=self.dtype.name,
        )

    def solve(self, source, rtol=1e-6, maxiter=500, initial=None):
        """Return ``(GPU field, diagnostics dict)`` or raise on nonconvergence.

        rtol is relative to the full, unpreconditioned source L2 norm.
        Diagnostics['residual'] and ['relative_residual'] both report the
        recomputed relative true residual, not the CG recurrence. Exact
        zero source returns zero even with a nonzero
        initial guess. initial is copied, never modified. A failed solve
        raises NonlocalConvergenceError with a diagnostics attribute.
        """
        rtol = float(rtol)
        maxiter = operator.index(maxiter)
        if not math.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError("rtol must be finite and between zero and one")
        if maxiter < 1:
            raise ValueError("maxiter must be a positive integer")
        with self.xp.cuda.Device(self._device), self._measure("nonlocal_solve"):
            return self._solve(source, rtol, maxiter, initial)

    def _solve(self, source, rtol, maxiter, initial):
        xp = self.xp
        source = self._field(source, "source")
        flags = int(_kernels().source_flags(source, self._mask).item())
        if flags & 1:
            raise ValueError("source must be finite")
        if flags & 2:
            raise ValueError("source must be exactly zero in fibers")
        if initial is not None:
            initial = self._field(initial, "initial")
            if not bool(xp.isfinite(initial).all().item()):
                raise ValueError("initial must be finite")
        self._operator_calls = self._preconditioner_calls = 0
        source_mean = xp.mean(source, dtype=xp.float64)
        rhs_squared = self._dot(source, source)
        target = rtol * math.sqrt(rhs_squared)

        def finish(x, rr, count, method, reason=None):
            accepted = math.isfinite(rr) and rr >= 0 and math.sqrt(rr) <= target
            diagnostics = self._diagnostics(
                x, source_mean, rhs_squared, rr, count, method, rtol, accepted
            )
            if not accepted:
                raise NonlocalConvergenceError(
                    reason
                    or f"Nonlocal Helmholtz failed true residual after {count} iterations",
                    diagnostics,
                )
            return x, diagnostics

        if not math.isfinite(rhs_squared):
            raise NonlinearConvergenceError("Nonfinite nonlocal source norm")
        if rhs_squared == 0 and bool(xp.any(source != 0).item()):
            raise NonlinearConvergenceError("Nonlocal source norm underflowed in FP64")
        if rhs_squared == 0 or self._identity:
            x = source.copy()
            return finish(x, 0.0, 0, "zero" if rhs_squared == 0 else "identity")
        if self._constant:
            x = xp.empty_like(source)
            self._precondition_into(source, x)
            return finish(x, self._true_residual(x, source, source_mean), 0, "fft")

        x = xp.zeros_like(source) if initial is None else initial.copy()
        rr = self._true_residual(x, source, source_mean)
        if math.isfinite(rr) and math.sqrt(rr) <= target:
            return finish(x, rr, 0, "pcg")
        self._precondition_into(self._r, self._z)
        rz = self._dot(self._r, self._z)
        xp.copyto(self._p, self._z)
        for iteration in range(1, maxiter + 1):
            self._apply_into(self._p, self._ap)
            pap = self._dot(self._p, self._ap)
            if not math.isfinite(pap) or not math.isfinite(rz) or min(pap, rz) <= 0:
                rr = self._true_residual(x, source, source_mean)
                return finish(
                    x,
                    rr,
                    iteration - 1,
                    "pcg",
                    "Nonlocal PCG lost positive curvature/preconditioner",
                )
            _kernels().update(self._p, self._ap, np.float64(rz / pap), x, self._r)
            rr = self._dot(self._r, self._r)
            if not math.isfinite(rr):
                return finish(
                    x, rr, iteration, "pcg", "Nonfinite nonlocal PCG residual"
                )
            # Audit before acceptance, periodically, and at budget exhaustion.
            # Failed estimated convergence restarts with the true residual.
            audit = (
                math.sqrt(rr) <= target or iteration % 32 == 0 or iteration == maxiter
            )
            if audit:
                rr = self._true_residual(x, source, source_mean)
                if math.isfinite(rr) and math.sqrt(rr) <= target:
                    return finish(x, rr, iteration, "pcg")
                if iteration == maxiter or not math.isfinite(rr):
                    return finish(x, rr, iteration, "pcg")
            self._precondition_into(self._r, self._z)
            next_rz = self._dot(self._r, self._z)
            if audit:
                xp.copyto(self._p, self._z)
            else:
                _kernels().direction(self._z, np.float64(next_rz / rz), self._p)
            rz = next_rz
        raise AssertionError("Unreachable PCG termination")
