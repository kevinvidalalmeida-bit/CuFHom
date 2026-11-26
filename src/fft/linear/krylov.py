"""Symmetric linear coordinates, CUDA fusions and residual-audited CG."""

import copy
import operator
from types import SimpleNamespace

import numpy as np
from cufhom.fft.operators import (
    _COORDINATE_CODE,
    DisplacementCoordinates,
    hermitian_plane,
)
from cufhom_core.tensors import fft as core_fft
from cufhom_core.tensors.fft import cp

from .common import FreshProgress, LinearSolveError, stored_planes


def replace_once(code, old, new):
    """Replace one CUDA source fragment, rejecting missing or duplicate matches."""
    if code.count(old) != 1:
        raise RuntimeError("Coordinate kernel changed: review linear fusions")
    return code.replace(old, new)


def direction_source(code):
    """Derive the coordinate-direction update from the reference kernel source."""
    code = replace_once(code, "const COMPLEX* x", "COMPLEX* x")
    code = replace_once(
        code,
        "REAL longitudinal) {",
        "REAL longitudinal, const COMPLEX* residual, REAL beta) {",
    )
    return replace_once(
        code,
        "if(s>=count)return;",
        r"""if(s>=count)return;
 // Update ALL coordinates before handling zero/Nyquist modes. Each thread
 // owns the same frequency in all three components; no inter-thread read.
 for(int c=0;c<3;c++) {
   const long long index=c*count+s;
   x[index]=MAKE(residual[index].x+beta*x[index].x,
                 residual[index].y+beta*x[index].y);
 }
""",
    )


_KERNELS = {}
_MEAN_ROOT = r"""
extern "C" __global__ void mean_root(const REAL* root,COMPLEX* y,long long first,long long stride,
 int free_mask,int adjoint) {
 // One thread: y's free means <- L m (strain map) or L^T m (adjoint), L the
 // packed 6x6 root (rows Mandel components, columns free means).
 int free[6],n=0;
 for(int c=0;c<6;c++)if(free_mask&(1<<c))free[n++]=c;
 REAL in[6],out[6];
 for(int j=0;j<n;j++)in[j]=y[first+(adjoint?j:free[j]*stride)].x;
 for(int i=0;i<n;i++) {
   REAL v=0;
   for(int j=0;j<n;j++)v+=(adjoint?root[6*free[j]+i]:root[6*free[i]+j])*in[j];
   out[i]=v;
 }
 for(int i=0;i<n;i++)y[first+(adjoint?i:free[i]*stride)]=MAKE(out[i],0);
}
"""


def coordinate_source():
    """Shared QR/RQ* formula, including the symmetric reference root."""
    code = replace_once(
        _COORDINATE_CODE,
        "int adjoint)",
        "int adjoint, REAL transverse, REAL longitudinal)",
    )
    return replace_once(
        code,
        "v[c]=r2*v[c]+(1-r2)*k[c]*dot;",
        "v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;",
    )


def tiled_coordinate_source():
    """QR/RQ* directly between production spectra and tile-major vectors.

    On even grids the tile-major k2 = 0 coordinates are Q = z0 + i zM (the
    z-Nyquist plane packed; see FusedPipeline.to_fused): the adjoint reads
    the stress planes 0 and Nyquist as h0 + i hM into that slot, and the
    strain map writes S Q into plane 0 only, for
    hermitian_plane(h, n2, packed=True) to split before an inverse transform.
    Odd grids store every plane. Only the three-coordinate address and that
    combined read differ from coordinate_source.
    Free means stay on the established production route.
    """
    code = replace_once(
        coordinate_source(), "REAL longitudinal)", "REAL longitudinal, int tile)"
    )
    code = replace_once(
        code,
        " if(s==0)",
        r"""
 const long long m=n2%2?nz:nz-1;
 const long long coordinate=(((i1*(m/tile)+i2/tile)*n0+i0)*tile+i2%tile)*3;
 // The z-Nyquist plane lives in the packed k2 = 0 coordinates (see above).
 if(n2%2==0 && i2==m)return;
 const bool packed=n2%2==0 && i2==0;
 if(s==0)""",
    )
    code = code.replace(
        "x[c*count+s]", "xin(x,c,count,s,m,coordinate+c,adjoint,packed)"
    )
    code = (
        """__device__ __forceinline__ COMPLEX xin(const COMPLEX* x,int c,long long count,long long s,long long m,
 long long coordinate,int adjoint,bool packed) {
 if(!adjoint)return x[coordinate];
 const COMPLEX u=x[c*count+s];
 if(!packed)return u;
 const COMPLEX v=x[c*count+s+m];
 return MAKE(u.x-v.y,u.y+v.x);
}
"""
        + code
    )
    code = code.replace("y[c*count+s]", "y[adjoint?coordinate+c:c*count+s]")
    code = code.replace("y[c*count]=MAKE(0,0)", "y[c]=MAKE(0,0)")
    return code


def _coordinate_kernel(dtype, mode="map"):
    key = ("root", dtype.str, mode)
    if key not in _KERNELS:
        code = (
            _MEAN_ROOT
            if mode == "means"
            else tiled_coordinate_source()
            if mode == "tiled"
            else coordinate_source()
        )
        if mode == "direction":
            code = direction_source(code)
        name = "float" if dtype == cp.complex64 else "double"
        code = (
            code.replace("COMPLEX", name + "2")
            .replace("REAL", name)
            .replace("MAKE", "make_" + name + "2")
        )
        _KERNELS[key] = cp.RawKernel(
            code, "mean_root" if mode == "means" else "coordinates"
        )
    return _KERNELS[key]


class SymmetricCoordinates(DisplacementCoordinates):
    """QR and RQ*, R=M^-1/2, fused into the compatible-coordinate kernel.

    Free mean strains (mixed loading) use L, the Cholesky factor of the
    reference compliance on the free components: E = L m, adjoint L^T.
    """

    def __init__(self, projector, bulk, shear):
        super().__init__(projector)
        if not np.isfinite([bulk, shear]).all() or min(bulk, shear) <= 0:
            raise ValueError("Symmetric coordinates require positive reference moduli")
        self.transverse = np.sqrt(1 / (2 * shear))
        self.longitudinal = np.sqrt(1 / (bulk + 4 * shear / 3))
        free = self.P.free
        C = np.eye(6) * 2 * shear
        C[:3, :3] += bulk - 2 * shear / 3
        # Rows are Mandel components, columns the packed free means.
        self.mean_root = np.zeros((6, 6))
        if free:
            self.mean_root[free, : len(free)] = np.linalg.cholesky(
                np.linalg.inv(C[np.ix_(free, free)])
            )
        self._device_roots = {}
        self._whitening = None
        self.tile = None

    def tiled(self, tile):
        """Apply the same map using fused pipeline storage without copies."""
        tile = operator.index(tile)
        planes = stored_planes(self.shape[-1])
        if self.xp is not cp or self.P.free or tile < 1 or planes % tile:
            raise ValueError(
                "Tiled coordinates require GPU, fixed means and a last-axis plane count divisible by the tile"
            )
        out = copy.copy(self)
        out.tile = tile
        out.size = 3 * self.shape[0] * self.shape[1] * planes
        return out

    def whiten(self, value):
        """Apply R to isometric adjoint coordinates without a transform.

        Per frequency R = transverse I + (longitudinal - transverse) k k^T on
        the displacement coordinates, and L^T on the packed free means.
        """
        if self._whitening is None:
            free = self.P.free
            mean = np.zeros((6, 6))
            for i, c in enumerate(free):
                for j, d in enumerate(free):
                    mean[c, d] = self.mean_root[d, i]
            self._whitening = SimpleNamespace(
                a=self.transverse,
                b=self.longitudinal,
                mean=self.xp.asarray(mean, dtype=value.real.dtype),
            )
        return self.precondition(value, self._whitening)

    def device_mean_root(self, dtype=np.float32):
        """Return the cached reference mean-block root in the requested precision."""
        dtype = np.dtype(dtype)
        if dtype not in self._device_roots:
            self._device_roots[dtype] = cp.asarray(self.mean_root, dtype=dtype)
        return self._device_roots[dtype]

    def root(self, value, adjoint=False):
        """Apply the reference square-root factor or its adjoint to coordinates."""
        k, _, _ = self._cpu_geometry()
        v = value[: 3 * self.count].reshape((3,) + self.hshape)
        out = (
            self.transverse * v
            + (self.longitudinal - self.transverse) * k * np.sum(k * v, axis=0)
        ).ravel()
        free = self.P.free
        if free:
            L = self.mean_root[free, : len(free)]
            out = np.concatenate(
                [out, (L.T if adjoint else L) @ value[3 * self.count :]]
            )
        return out

    def adjoint_with_spectrum(self, field):
        """Keep the unscaled RFFT only for a final original-equation audit.

        The coordinate map reads its input. The physical projection creates
        a separate spectrum, so the inverse FFT never consumes this snapshot.
        """
        with self.P.timers.measure("coordinates_adjoint"):
            h = core_fft.rfftn_cached(field, self.shape, (1, 2, 3))
            return self._map(h, True), h

    def _map(self, x, adjoint):
        if self.xp is np:
            return (
                self.root(super()._map(x, True), True)
                if adjoint
                else super()._map(self.root(x), False)
            )
        out = cp.empty(self.size if adjoint else (6,) + self.hshape, dtype=x.dtype)
        mode = "map" if self.tile is None else "tiled"
        args = self._args(x, out, adjoint)
        if self.tile is not None:
            args += (np.int32(self.tile),)
        _coordinate_kernel(x.dtype, mode)(((self.count + 255) // 256,), (256,), args)
        if self.P.free:
            _coordinate_kernel(x.dtype, "means")(
                (1,),
                (1,),
                (
                    self.device_mean_root(x.real.dtype),
                    out,
                    np.int64(3 * self.count if adjoint else 0),
                    np.int64(self.count),
                    np.int32(sum(1 << i for i in self.P.free)),
                    np.int32(adjoint),
                ),
            )
        return out

    def _args(self, x, out, adjoint):
        d = x.real.dtype.type
        return (
            x,
            out,
            *(np.int64(n) for n in self.shape),
            np.int64(self.hshape[-1]),
            *(d(1 / L) for L in self.P.lengths),
            d(self.norm),
            np.int32(sum(1 << i for i in self.P.free)),
            np.int32(adjoint),
            d(self.transverse),
            d(self.longitudinal),
        )

    def _check_vector(self, vector):
        if (
            self.xp is not cp
            or not isinstance(vector, cp.ndarray)
            or vector.dtype not in (cp.complex64, cp.complex128)
            or vector.shape != (self.size,)
            or not vector.flags.c_contiguous
        ):
            raise ValueError(
                "Linear fusion requires contiguous complex GPU coordinates"
            )

    def strain_updated_direction(self, p, residual, beta):
        """Set p = residual + beta*p and return QRp in one coordinate pass.

        Only p is updated in place; residual and the returned strain are owned
        independently. CG calls this immediately before using the new p.
        """
        self._check_vector(p)
        self._check_vector(residual)
        if p.dtype != residual.dtype:
            raise ValueError("Direction and residual must have the same dtype")
        h = cp.empty((6,) + self.hshape, dtype=p.dtype)
        args = self._args(p, h, False) + (residual, p.real.dtype.type(beta))
        _coordinate_kernel(p.dtype, "direction")(
            ((self.count + 255) // 256,), (256,), args
        )
        return core_fft.irfftn_unscaled_cached(
            hermitian_plane(h, self.shape[-1]), self.shape, (1, 2, 3)
        )


def _finish_smoothing(x, y, weight):
    if not isinstance(y, cp.ndarray):
        y.smooth(x, weight)  # host-resident iterates (coordinates._HostVector)
        return
    key = ("smooth", x.real.dtype.str)
    if key not in _KERNELS:
        _KERNELS[key] = cp.ElementwiseKernel(
            "T x, " + x.real.dtype.name + " weight",
            "T y",
            "y = y + weight * (x-y);",
            "linear_cg_finish_smoothing",
        )
    _KERNELS[key](x, x.real.dtype.type(weight), y)


def fused_cg(
    action,
    rhs,
    dot,
    tolerance,
    maxiter,
    *,
    action_and_dot,
    device_update_callback,
    residual_smoothing=False,
    residual_action=None,
    accept=None,
    settle=None,
    replace=None,
    drop=0.0,
    energy=None,
    growth=None,
    margin=None,
    fresh_metric=None,
    stats=None,
    device_prefix=None,
    iterate=None,
    spill_direction=False,
):
    """CG in symmetrically preconditioned coordinates with true-residual restarts.

    action_and_dot(p, direction=None) first applies a deferred direction
    update p <- r + beta*p, then returns (step, p.Hp) with p.Hp an FP64 device
    scalar. device_update_callback(p, step, r.r, p.Hp, x, r[, (y, weight)])
    updates x and r and returns the new r.r; an invalid energy
    must give a nonfinite norm, without a host readback.
    Late residual smoothing uses Euclidean orthogonality of CG residuals. Its
    harmonic estimate is NEVER an acceptance test in float32: the true
    residual rhs - residual_action(x) (default: action) must pass, and then
    accept(x) returns None to stop, or a factor < 1 for the squared limit to
    continue. settle() completes an update the operator deferred, before x
    is read here.

    replace(x, r) (mixed precision): fold the low-precision increment x into
    the high-precision solution (x becomes zero), write the fresh residual of
    that solution into r and return r.r (audit=False: the caller needs no
    acceptance norm of that solution). It then replaces every candidate
    residual, and, outside the smoothing decade, the recursive residual once
    r.r drops by drop^2 since the last replacement; CG keeps its search
    direction (reliable update). x is then an increment, and the returned
    x is zero: the solution lives with the caller.

    growth (mixed precision): the FP32 recursion stalls at a floor and then
    grows; a recursive r.r above growth times its minimum since the last
    fresh residual forces a replacement that restarts the direction (or,
    inside the smoothing decade, a candidate inspection). margin, a one-element list shared by the loads of
    a call, divides the squared limit by the ratio of fresh to recursive r.r
    learned at rejected candidates (times 1.2, at most 100): later loads stop
    their recursion where its candidate passes.

    energy(work, new) (goal-oriented stop): work = sum_j alpha_j rho_j of the
    plain CG recursion (the energy decrease; it continues through restarts
    and replacements). A True return inspects a candidate exactly as the
    residual limit does, and accept(x) then decides on its own fresh bound,
    independently of that limit. Requires residual_smoothing=False: the
    energy recursion describes the plain CG iterate.

    device_prefix optionally groups ordinary unsmoothed steps. It must stop
    on device at the first candidate, smoothing boundary, reliable-update
    trigger or invalid step. The last step's scalar control and every fresh
    acceptance remain here; it returns (steps, previous_rr, new_rr, floor).

    iterate, a zero vector of the caller's storage (a host-resident
    coordinates._HostVector), replaces the device x; its updates, copy and
    smoothing go through the pipeline's settle and its own methods.
    spill_direction keeps the search direction in pinned host memory during
    a scheduled replacement, whose evaluation then has its device memory.
    """
    if not isinstance(rhs, cp.ndarray) or rhs.dtype not in (
        cp.complex64,
        cp.float32,
        cp.complex128,
        cp.float64,
    ):
        raise ValueError("Fused linear CG requires FP32/FP64 GPU vectors")
    settle = settle or (lambda: None)
    progress = FreshProgress()
    stats = {} if stats is None else stats
    stats.update(restarts=0, replacements=0, candidate_checks=0)
    # With replace, every later residual is recomputed: CG may own rhs.
    x = cp.zeros_like(rhs) if iterate is None else iterate
    r = rhs if replace is not None else rhs.copy()
    rr = dot(r, r)
    limit = tolerance * tolerance * rr / (1.0 if margin is None else margin[0])
    if rr == 0:
        return x, 0
    p = r.copy()
    direction = None
    y, estimate, weight = None, rr, 0.0
    reference = floor = rr
    if energy is not None and residual_smoothing:
        raise ValueError(
            "The energy stop needs the plain CG iterate (no residual smoothing)"
        )
    work = 0.0
    i = 0
    while i < maxiter:
        if not np.isfinite(rr) or rr <= 0:
            raise LinearSolveError(
                "Fused linear CG lost positivity", status="lost_curvature", iterations=i
            )
        if device_prefix is not None and y is None and energy is None:
            steps, rr, new, floor = device_prefix(
                p,
                rr,
                x,
                r,
                direction,
                limit,
                floor,
                reference,
                drop,
                growth,
                residual_smoothing,
                maxiter - i,
            )
            i += steps - 1
            stats["control_queries"] = stats.get("control_queries", 0) + 1
            stats["device_control_iterations"] = (
                stats.get("device_control_iterations", 0) + steps
            )
        else:
            step, pap = action_and_dot(p, direction=direction)
            if (
                not isinstance(pap, cp.ndarray)
                or pap.shape != ()
                or pap.dtype != cp.float64
            ):
                raise ValueError(
                    "Device energy update requires a float64 device denominator"
                )
            new = device_update_callback(
                p, step, rr, pap, x, r, *(() if y is None else ((y, weight),))
            )
            stats["control_queries"] = stats.get("control_queries", 0) + 1
        direction = None
        if not np.isfinite(new) or new < 0:
            raise LinearSolveError(
                "Nonfinite fused linear residual or unsafe PCG step",
                status="nonfinite",
                iterations=i + 1,
            )
        if energy is not None:
            work += rr * rr / float(pap)  # alpha*rho; the update already synchronized
        estimate = new if y is None else estimate
        if residual_smoothing:
            if y is None:
                # Spend an extra vector pass only in the final decade of
                # residual norms. The factor is independent of the material,
                # mesh, geometry and requested tolerance.
                if limit < new <= 100 * limit:
                    settle()
                    y, weight = x.copy(), 0.0
            else:
                weight = estimate / (estimate + new)
                estimate = weight * new
        floor = min(floor, new)
        grown = growth is not None and new > growth * floor
        if (
            estimate <= limit
            or (grown and y is not None)
            or (energy is not None and energy(work, new))
        ):
            stats["candidate_checks"] += 1
            stats["recursive_norm_res"] = float(np.sqrt(estimate))
            settle()
            if y is not None:
                _finish_smoothing(x, y, weight)
                x = y
            # Neither the old residual nor direction is used after a fresh
            # candidate inspection: a rejection restarts both below. Release
            # them before the library FFT audit instead of overlapping their
            # storage with full strain/stress/spectral buffers.
            del p
            if replace is None:
                del r
                r = rhs - (residual_action or action)(x)
                new = dot(r, r)
            else:
                recursive = estimate
                new = reference = replace(x, r)
                if margin is not None and limit < new < np.inf:
                    margin[0] = min(100.0, max(margin[0], 1.2 * new / recursive))
            if new <= limit or energy is not None:
                factor = None if accept is None else accept(x)
                if factor is None:
                    return x, i + 1
                # The original equation is not met yet: keep the current x
                # and restart from its true residual with a stricter limit.
                limit *= factor
            progress.check(
                np.sqrt(new) if fresh_metric is None else fresh_metric(), i + 1
            )
            stats["restarts"] += 1
            # Reset both sequences after replacing the recursive residual.
            # x owns the old y buffer; a later smoothing start makes a copy.
            y, estimate, weight = None, new, 0.0
            p = r.copy()
            floor = new
        elif (
            replace is not None
            and y is None
            and (new <= drop * drop * reference or grown)
        ):
            stats["replacements"] += 1
            settle()
            if spill_direction:
                import cupyx

                # A restart (grown) discards the direction.
                held = (
                    None if grown else p.get(out=cupyx.empty_pinned(p.shape, p.dtype))
                )
                del p
            fresh = reference = replace(x, r, audit=False)
            if spill_direction:
                p = cp.empty_like(r)
                if held is not None:
                    p.set(held)
                del held
            if not np.isfinite(fresh) or fresh <= 0:
                raise LinearSolveError(
                    "Nonfinite or zero replaced linear residual before acceptance",
                    status="nonfinite",
                    iterations=i + 1,
                )
            # A scheduled replacement keeps the search direction; past the
            # recursion floor (grown) it is no longer conjugate: restart.
            direction = (r, 0.0 if grown else fresh / rr)
            stats["restarts"] += bool(grown)
            new = floor = fresh
        else:
            direction = (r, new / rr)
        rr = new
        i += 1
    raise LinearSolveError(
        "Fused linear CG failed true residual: iteration budget exhausted",
        status="maxiter",
        iterations=maxiter,
    )
