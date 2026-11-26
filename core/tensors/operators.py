"""Linear operators on Tensor fields: DFTs and the projected material operator."""

import time
import numpy as np
from cufhom_core.trigpol import fft_form_default
from cufhom_core.general.cupy_capacity import library_capacity_required
from cufhom_core.tensors.objects import (
    SYM21_PAIRS,
    Tensor,
    TensorFuns,
    elasticity_direct_array,
    indexed_sym21_array,
)
from cufhom_core.tensors.fft import (
    cp, cupy_synchronize, fftn_cached, fftn_unscaled_cached, get_array_module, get_fft_backend,
    ifftn_cached, ifftn_unscaled_real_cached, irfftn_cached, irfftn_unscaled_cached,
    is_cupy_array, rfftn_cached, to_backend_array, CUPY_AVAILABLE,
)




def _unwrap_timed(op):
    return getattr(op, 'op', op)


def _tensor_val_on_cupy(tensor):
    base = _unwrap_timed(tensor)
    if isinstance(base, Tensor) and is_cupy_array(base.val):
        return base.val
    cache_owner = base if isinstance(base, Tensor) else tensor
    cached = getattr(cache_owner, '_cupy_val_cache', None)
    if cached is None:
        raw = base.val if isinstance(base, Tensor) else tensor.val
        cached = to_backend_array(raw, prefer_backend='cupy')
        setattr(cache_owner, '_cupy_val_cache', cached)
    return cached


def _rfft_unscaled_projection(tensor, prodN):
    """Cache a projection tensor whose scale is absorbed into raw C2R FFTs."""
    value = _tensor_val_on_cupy(tensor)
    key = (id(value), int(prodN), str(value.dtype))
    cached = getattr(tensor, '_cupy_rfft_unscaled_projection', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    # The normal C2R inverse returns C2R(x) / prodN.  The direct cuFFT path
    # returns C2R(x), hence scale hG once rather than divide every iteration.
    scaled = tensor.copy(val=value * (1.0 / float(prodN)))
    setattr(tensor, '_cupy_rfft_unscaled_projection', (key, scaled))
    return scaled


def _tensor_apply_cupy(tensor, value, out=None):
    """Material or projection action on a CuPy field, for the layouts the solver uses."""
    base = _unwrap_timed(tensor)
    multype = base.multype
    if multype in ('elasticity_hg1_direct', 'elasticity_g1_direct',
                   'elasticity_hg2_direct', 'elasticity_g2_direct'):
        direct = elasticity_direct_array(base, value, out=out)
        if direct is None:
            raise NotImplementedError("elasticity direct projection requiere CuPy.")
        return direct
    if multype in ('sym21_indexed', 'indexed_sym21'):
        indexed = indexed_sym21_array(base, value)
        if indexed is None:
            raise NotImplementedError("material sym21 indexado requiere CuPy float32.")
        return indexed

    xp = get_array_module(value)
    tval = _tensor_val_on_cupy(tensor)
    if multype in ('sym21', 'symmetric21'):
        out = xp.zeros((6,) + value.shape[1:], dtype=xp.result_type(tval.dtype, value.dtype))
        for kk, (ii, jj) in enumerate(SYM21_PAIRS):
            cval = tval[kk]
            out[ii] += cval * value[jj]
            if ii != jj:
                out[jj] += cval * value[ii]
        return out
    if multype in (21, '21'):
        if tval.ndim == value.ndim:  # stacked right-hand sides (D, L, *N)
            out = xp.zeros((tval.shape[0], value.shape[1]) + value.shape[2:],
                           dtype=xp.result_type(tval.dtype, value.dtype))
            for j in range(tval.shape[1]):
                out += tval[:, j, None, ...] * value[j, None, ...]
            return out
        out = xp.zeros((tval.shape[0],) + value.shape[1:], dtype=xp.result_type(tval.dtype, value.dtype))
        for j in range(tval.shape[1]):
            out += tval[:, j, ...] * value[j, ...]
        return out
    raise NotImplementedError(f"multype '{multype}' no soportado en backend CuPy.")


class DFT(TensorFuns):
    """(Inverse) discrete Fourier transform of Tensor fields.

    fft_form 0 is the standard numpy.fft.fftn layout (normalized by
    prod(N) on the forward transform); 'r' keeps the half spectrum of real
    data.
    """
    def __init__(self, inverse=False, N=None, fft_form=fft_form_default, **kwargs):
        self.__dict__.update(kwargs)
        if 'name' not in kwargs:
            self.name = 'iDFT' if inverse else 'DFT'
        self.N=np.array(N, dtype=np.int32)
        self._N_tuple=tuple(int(v) for v in self.N.tolist())
        self._prodN=int(np.prod(self._N_tuple))
        self._axes=tuple(range(-len(self._N_tuple), 0))
        self.inverse=inverse
        self._set_fft(fft_form)

    def __call__(self, x, *, overwrite_x=False):
        if not isinstance(x, Tensor):
            raise ValueError('DFT.__call__')
        assert(x.Fourier==self.inverse)
        if self.inverse:
            if self.fft_form == 0:
                val = ifftn_cached(x.val, self._N_tuple, self._prodN, self._axes)
            else:
                val = irfftn_cached(x.val, self._N_tuple, self._axes, overwrite_x=overwrite_x)
            return x.copy(name='iF({0})'.format(x.name[:10]), val=val.real, Fourier=not x.Fourier)
        assert(x.fft_form==self.fft_form)
        if self.fft_form == 0:
            val = fftn_cached(x.val, self._N_tuple, self._prodN, self._axes)
        else:
            val = rfftn_cached(x.val, self._N_tuple, self._axes)
        return x.copy(name='F({0})'.format(x.name[:10]), val=val, Fourier=not x.Fourier)


class Operator():
    """Product of linear maps applied right to left:
    Operator(mat=[[F, G, H]])(x) = F(G(H(x))); mat_rev lists them in call order.
    """
    def __init__(self, name='Operator', mat_rev=None, mat=None):
        self.name=name
        self.mat_rev = mat_rev if mat_rev is not None else [list(reversed(summand)) for summand in mat]
        if len(self.mat_rev) != 1:
            raise NotImplementedError('Operator supports a single product of maps.')
        self.no_summands = 1

    def __call__(self, x):
        res = self._call_cupy_fast_path(x)
        if res is None:
            res = x
            for i,matrix in enumerate(self.mat_rev[0]):
                # Intermediate spectra belong to this product; caller-owned
                # inputs and spectra retained by physical audits stay intact.
                consume=(i>0 and is_cupy_array(res.val) and res.val.dtype in (cp.complex64,cp.complex128)
                         and library_capacity_required(res.N,np.empty((),dtype=res.val.dtype).real.dtype))
                if (consume and type(matrix) is Tensor and matrix.multype in
                        ('elasticity_hg1_direct','elasticity_g1_direct','elasticity_hg2_direct','elasticity_g2_direct')):
                    res=res.copy(val=elasticity_direct_array(matrix,res.val,out=res.val))
                elif consume and type(matrix) is DFT and matrix.inverse:
                    res=matrix(res,overwrite_x=True)
                else:
                    res = matrix(res)
        res.name='{0}({1})'.format(self.name[:6], x.name[:10])
        return res

    def apply_material_result(self, stress, material, projection, *, spectrum=None):
        """Project an already evaluated material result for this same operator.

        The optional spectrum is the same stress's unscaled RFFT, supplied by
        the owning solve's true check. Return None for mismatched factors,
        incompatible FFT conventions, or an unavailable native GPU path.
        """
        if self.no_summands != 1 or len(self.mat_rev[0]) != 2:
            return None
        Aop, GNop = self.mat_rev[0]
        if (_unwrap_timed(Aop) is not _unwrap_timed(material)
                or _unwrap_timed(GNop) is not _unwrap_timed(projection)):
            return None
        return self._call_cupy_fast_path(stress, material_result=stress.val,
                                        material_spectrum=spectrum)

    def _call_cupy_fast_path(self, x, material_result=None, material_spectrum=None):
        if get_fft_backend() != 'cupy' or not CUPY_AVAILABLE:
            return None
        if not isinstance(x, Tensor):
            return None
        if self.no_summands != 1 or len(self.mat_rev[0]) != 2:
            return None

        Aop, GNop = self.mat_rev[0]
        Araw = _unwrap_timed(Aop)
        GNraw = _unwrap_timed(GNop)
        if (Aop is not Araw) or (GNop is not GNraw):
            return None
        if not isinstance(Araw, Tensor) or not isinstance(GNraw, Operator):
            return None
        if GNraw.no_summands != 1 or len(GNraw.mat_rev[0]) != 3:
            return None

        FNop, hGop, FiNop = GNraw.mat_rev[0]
        FNraw = _unwrap_timed(FNop)
        hGraw = _unwrap_timed(hGop)
        FiNraw = _unwrap_timed(FiNop)
        if (FNop is not FNraw) or (hGop is not hGraw) or (FiNop is not FiNraw):
            return None
        if not (isinstance(FNraw, DFT) and isinstance(hGraw, Tensor) and isinstance(FiNraw, DFT)):
            return None
        if FNraw.inverse or not FiNraw.inverse or FNraw.fft_form != FiNraw.fft_form \
                or FNraw.fft_form not in (0, 'r'):
            return None

        # Unscaled forward/inverse FFTs: the real pair folds 1/prod(N) into hG
        # once; the complex pair (fft_form 0) keeps hG as stored.
        rfft = FNraw.fft_form == 'r'
        if material_spectrum is not None:
            shape = FNraw._N_tuple
            expected = x.val.shape[:-len(shape)]+shape[:-1]+(shape[-1]//2+1,)
            if (not rfft or material_result is None
                    or not isinstance(x.val, cp.ndarray)
                    or tuple(a % x.val.ndim for a in FNraw._axes) != (1, 2, 3)
                    or not isinstance(material_spectrum, cp.ndarray)
                    or material_spectrum.shape != expected
                    or material_spectrum.real.dtype != x.val.dtype
                    or material_spectrum.dtype not in (cp.complex64, cp.complex128)
                    or not material_spectrum.flags.c_contiguous
                    or material_spectrum.device.id != x.val.device.id):
                return None
        if (getattr(self,'_capacity_inplace_fft',False)
                and material_result is None and material_spectrum is None):
            from cufhom_core.general.cupy_strain_capacity import apply
            return x.copy(val=apply(Araw,hGraw,x.val,FNraw._N_tuple))
        stats = getattr(self, '_component_profile', None)

        def measure(label, function, *args):
            if stats is None:
                return function(*args)
            cupy_synchronize()
            started = time.perf_counter()
            result = function(*args)
            cupy_synchronize(result)
            stats[label]['seconds'] += time.perf_counter()-started
            stats[label]['calls'] += 1
            return result

        x_val = to_backend_array(x.val, prefer_backend='cupy')
        Ax = measure('A', _tensor_apply_cupy, Araw, x_val) if material_result is None else material_result
        if not rfft:
            Fx = measure('FN', fftn_unscaled_cached, Ax, FNraw._N_tuple, FNraw._axes)
        elif material_spectrum is None:
            Fx = measure('FN', rfftn_cached, Ax, FNraw._N_tuple, FNraw._axes)
        else:
            Fx = material_spectrum
        consume=(material_spectrum is None and rfft and x_val.dtype in (cp.float32,cp.float64)
                 and library_capacity_required(FNraw._N_tuple,x_val.dtype))
        if consume:
            del Ax
        projection = _rfft_unscaled_projection(hGraw, FNraw._prodN) if rfft else hGraw
        Gx = measure('hG', _tensor_apply_cupy, projection, Fx, Fx if consume else None)
        inverse = irfftn_unscaled_cached if rfft else ifftn_unscaled_real_cached
        return x.copy(val=measure('FiN', inverse, Gx, FiNraw._N_tuple, FiNraw._axes))

