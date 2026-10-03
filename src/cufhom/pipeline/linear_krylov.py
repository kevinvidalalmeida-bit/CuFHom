"""Symmetric reference coordinates and FP64-reduced CG for linear solves.

No nonlinear defaults are changed. The physical-equation audit remains in
linear_coordinates; a small recursive residual alone never accepts a solution.
"""
import numpy as np
from .nonlinear_coordinates import DisplacementCoordinates, _CODE
from .nonlinear_solver import _cg_updates, NonlinearConvergenceError, cp
from .linear_fusions import adjoint_dot_source, direction_source, replace_once
from cufhom_core.tensors import fft as core_fft

_KERNELS = {}


def coordinate_source():
    """Shared QR/RQ* formula, including the symmetric reference root."""
    code = replace_once(_CODE, 'int adjoint)',
                        'int adjoint, REAL transverse, REAL longitudinal)')
    return replace_once(code, 'v[c]=r2*v[c]+(1-r2)*k[c]*dot;',
        'v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;')


def _coordinate_kernel(dtype, mode='map'):
    key = ('root', dtype.str, mode)
    if key not in _KERNELS:
        code = coordinate_source()
        if mode == 'dot':
            code = adjoint_dot_source(code)
        elif mode == 'direction':
            code = direction_source(code)
        name = 'float' if dtype == cp.complex64 else 'double'
        code = code.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
        _KERNELS[key] = cp.RawKernel(code, 'adjoint_dot' if mode == 'dot' else 'coordinates')
    return _KERNELS[key]


class SymmetricCoordinates(DisplacementCoordinates):
    """QR and RQ*, R=M^-1/2, fused into the compatible-coordinate kernel."""
    def __init__(self, projector, bulk, shear):
        super().__init__(projector)
        if self.P.free or not np.isfinite([bulk, shear]).all() or min(bulk, shear) <= 0:
            raise ValueError('Symmetric linear coordinates require fixed means and positive moduli')
        self.transverse = np.sqrt(1/(2*shear))
        self.longitudinal = np.sqrt(1/(bulk+4*shear/3))

    def root(self, value):
        k, _, _ = self._cpu_geometry()
        v = value.reshape((3,)+self.hshape)
        return (self.transverse*v+(self.longitudinal-self.transverse)*k*np.sum(k*v, axis=0)).ravel()

    def adjoint_with_spectrum(self, field):
        """Keep the unscaled RFFT only for a final original-equation audit.

        The coordinate map reads its input. The physical projection creates
        a separate spectrum, so the inverse FFT never consumes this snapshot.
        """
        with self.P.timers.measure('coordinates_adjoint'):
            h = core_fft.rfftn_cached(field, self.shape, (1, 2, 3))
            return self._map(h, True), h

    def _map(self, x, adjoint):
        if self.xp is np:
            return self.root(super()._map(x, True)) if adjoint else super()._map(self.root(x), False)
        out = cp.empty(self.size if adjoint else (6,)+self.hshape, dtype=x.dtype)
        _coordinate_kernel(x.dtype)(((self.count+255)//256,), (256,), self._args(x, out, adjoint))
        return out

    def _args(self, x, out, adjoint):
        d = x.real.dtype.type
        return (x, out, *(np.int64(n) for n in self.shape), np.int64(self.hshape[-1]),
                *(d(1/L) for L in self.P.lengths), d(self.norm), np.int32(0), np.int32(adjoint),
                d(self.transverse), d(self.longitudinal))

    def _check_vector(self, vector):
        if (self.xp is not cp or not isinstance(vector, cp.ndarray)
                or vector.dtype not in (cp.complex64, cp.complex128)
                or vector.shape != (self.size,) or not vector.flags.c_contiguous):
            raise ValueError('Linear fusion requires contiguous complex GPU coordinates')

    def strain_updated_direction(self, p, residual, beta):
        """Set p = residual + beta*p and return QRp in one coordinate pass.

        Only p is updated in place; residual and the returned strain are owned
        independently. CG calls this immediately before using the new p.
        """
        self._check_vector(p); self._check_vector(residual)
        if p.dtype != residual.dtype:
            raise ValueError('Direction and residual must have the same dtype')
        h = cp.empty((6,)+self.hshape, dtype=p.dtype)
        args = self._args(p, h, False)+(residual, p.real.dtype.type(beta))
        _coordinate_kernel(p.dtype, 'direction')(((self.count+255)//256,), (256,), args)
        return core_fft.irfftn_unscaled_cached(h, self.shape, (1, 2, 3))

    def adjoint_and_dot(self, field, p):
        """Return RQ*field and Re(p* RQ*field), accumulating products in FP64."""
        self._check_vector(p)
        if (field.shape != (6,)+self.shape or field.dtype != p.real.dtype
                or not field.flags.c_contiguous):
            raise ValueError('Field must match the coordinate grid and real dtype')
        h = core_fft.rfftn_cached(field, self.shape, (1, 2, 3))
        out = cp.empty(self.size, dtype=p.dtype)
        blocks = (self.count+255)//256
        partial = cp.empty(blocks, dtype=cp.float64)
        _coordinate_kernel(p.dtype, 'dot')((blocks,), (256,), self._args(h, out, True)+(p, partial))
        return out, float(cp.sum(partial, dtype=cp.float64))


_UPDATE = r'''
extern "C" __global__ void update_rr(const float* p, const float* ap,
 float alpha, float* x, float* r, double* partial, long long count) {
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     i<count;i+=(long long)gridDim.x*blockDim.x) {
   x[i]+=alpha*p[i];
   float ri=r[i]-alpha*ap[i]; r[i]=ri;
   sum+=(double)ri*(double)ri;
 }
 for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 __shared__ double sums[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)sums[warp]=sum;
 __syncthreads();
 if(warp==0) {
   sum=lane<8?sums[lane]:0;
   for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
   if(lane==0)partial[blockIdx.x]=sum;
 }
}
'''


def fused_update(p, ap, alpha, x, r, partial, smoothing=None):
    key = 'update' if smoothing is None else 'update_smooth'
    if key not in _KERNELS:
        code = _UPDATE
        if smoothing is not None:
            code = replace_once(code, 'double* partial, long long count)',
                'double* partial, long long count, float* y, float weight)')
            # Complete the PREVIOUS smoothing step before changing x. Its
            # weight was only known after the preceding residual reduction.
            code = replace_once(code, 'x[i]+=alpha*p[i];',
                'float xi=x[i]; y[i]+=weight*(xi-y[i]); x[i]=xi+alpha*p[i];')
        _KERNELS[key] = cp.RawKernel(code, 'update_rr')
    args = (p, ap, np.float32(alpha), x, r, partial, np.int64(r.nbytes//4))
    if smoothing is not None:
        args += (smoothing[0], np.float32(smoothing[1]))
    _KERNELS[key]((partial.size,), (256,), args)
    return float(cp.sum(partial, dtype=cp.float64))


def _finish_smoothing(x, y, weight):
    if 'smooth' not in _KERNELS:
        _KERNELS['smooth'] = cp.ElementwiseKernel('T x, float32 weight', 'T y',
            'y = y + weight * (x-y);', 'linear_cg_finish_smoothing')
    _KERNELS['smooth'](x, np.float32(weight), y)


def fused_cg(action, rhs, dot, tolerance, maxiter, xp, project=None, fused=False,
             precondition=None, initial=None, action_and_dot=None, residual_smoothing=False,
             update_callback=None, residual_action=None):
    """CG/PCG with true-residual restart and optional late residual smoothing.

    Smoothing uses Euclidean orthogonality of ordinary CG residuals, so it
    requires coordinates that already contain the symmetric preconditioner.
    Its harmonic residual estimate is NEVER an acceptance test in float32.
    """
    if xp is not cp or rhs.dtype not in (cp.complex64, cp.float32) or project is not None or initial is not None:
        raise ValueError('Fused linear CG requires zero-start complex64/float32 GPU vectors')
    if residual_smoothing and precondition is not None:
        raise ValueError('Residual smoothing requires symmetrically preconditioned coordinates')
    x = cp.zeros_like(rhs); r = rhs.copy()
    rr = dot(r, r); limit = tolerance*tolerance*rr
    if rr == 0:
        return x, 0
    partial = cp.empty(min(16384, (rhs.nbytes//4+255)//256), dtype=cp.float64)
    z = precondition(r) if precondition else r
    rz = dot(r, z) if precondition else rr
    p = z.copy()
    direction = None
    y, estimate, weight = None, rr, 0.
    for i in range(maxiter):
        if action_and_dot is None:
            ap = action(p); pap = dot(p, ap)
        else:
            # The optional operator consumes the deferred p update before
            # applying A. Audits and reliable restarts use the plain action.
            ap, pap = action_and_dot(p, direction=direction)
            direction = None
        if not np.isfinite([pap, rz]).all() or min(pap, rz) <= 0:
            raise NonlinearConvergenceError('Fused linear CG lost positivity')
        update_args = () if y is None else ((y, weight),)
        new = (update_callback or fused_update)(p, ap, rz/pap, x, r, partial, *update_args)
        if not np.isfinite(new) or new < 0:
            raise NonlinearConvergenceError('Nonfinite fused linear residual')
        estimate = new if y is None else estimate
        if residual_smoothing:
            if y is None:
                # Spend an extra vector pass only in the final decade of
                # residual norms. The factor is independent of the material,
                # mesh, geometry and requested tolerance.
                if limit < new <= 100*limit:
                    y, weight = x.copy(), 0.
            else:
                weight = estimate/(estimate+new)
                estimate = weight*new
        if estimate <= limit:
            if y is not None:
                _finish_smoothing(x, y, weight)
                x = y
            r = rhs-(residual_action or action)(x)
            new = dot(r, r)
            if new <= limit:
                return x, i+1
            # Reset both sequences after replacing the recursive residual.
            # x owns the old y buffer; a later smoothing start makes a copy.
            y, estimate, weight = None, new, 0.
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            p = z.copy()
        else:
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            if action_and_dot is None:
                _cg_updates()[1](z, np.float64(next_rz/rz), p)
            else:
                direction = (z, next_rz/rz)
        rz = next_rz
    raise NonlinearConvergenceError('Fused linear CG failed true residual')
