"""
This module contains classes and functions representing tensors
of trigonometric polynomials and relating operators.
"""

import numpy as np
from cufhom_core.trigpol import mean_index, fft_form_default
from cufhom_core.tensors.fft import (
    CUPY_AVAILABLE, get_array_module, get_fft_backend, is_array_type,
    is_cupy_array, to_backend_array, to_host_array, cp,
)
import re
from copy import copy


def _scalar_to_float(x):
    if CUPY_AVAILABLE and is_cupy_array(x):
        return float(x.item())
    if isinstance(x, np.generic):
        return float(x.item())
    return float(x)


SYM21_PAIRS = (
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
    (2, 2), (2, 3), (2, 4), (2, 5),
    (3, 3), (3, 4), (3, 5),
    (4, 4), (4, 5),
    (5, 5),
)

_CUPY_ELASTIC_DIRECT_KERNELS = {}
_CUPY_INDEXED_SYM21_KERNEL = None
_CUPY_INDEXED_SYM21_FP64_KERNEL = None


class TensorFuns:

    def mean_index(self):
        return mean_index(self.N, self.fft_form)

    def _copy(self, keys, **kwargs):
        # Explicit replacements already own their intended copy/alias semantics.
        # Do not allocate and discard a potentially large CPU/GPU field first.
        data={k:copy(self.__dict__[k]) for k in keys if k not in kwargs}
        data.update(kwargs)
        return self.__class__(**data)

    def copy(self, **kwargs):
        return self._copy(self.keys, **kwargs)

    def _set_fft(self, fft_form):
        assert(fft_form in ['c', 'r', 0])

        if fft_form in ['r']:
            self.N_fft=self.get_N_real(self.N)
            self.fft_coef=np.prod(self.N)
        else:
            self.N_fft=self.N
            self.fft_coef=1.

        self.fft_form=fft_form

    @staticmethod
    def get_N_real(N):
        N_rfft=np.copy(N)
        N_rfft[-1]=int(np.trunc(N[-1]/2)+1) # N[-1]//2+1
        return tuple(N_rfft)

class Tensor(TensorFuns):
    keys=('name','val','order','Y','N','multype','Fourier','fft_form','origin') # default keys

    def __init__(self, name='', val=None, order=None, shape=None, N=None, Y=None,
                 multype='scal', Fourier=False, fft_form=fft_form_default,
                 origin=0, dtype=None):

        self.name=name
        self.Fourier=Fourier
        self.origin=origin

        if is_array_type(val): # define: val + order
            self.val=val
            self.order=int(order)
            self.shape=self.val.shape[:order]
            if fft_form in ['r'] and Fourier:
                self.N=tuple(np.array(N, dtype=int))
            else:
                self.N=self.val.shape[order:]
            self._set_fft(fft_form)

        elif shape is not None and N is not None: # define: shape + N
            self.N=tuple(np.array(N, dtype=int))
            self._set_fft(fft_form)
            self.shape=tuple(np.array(shape, dtype=int))
            self.order=len(self.shape)

            backend = get_fft_backend()
            real_dtype = np.dtype(float if dtype is None else dtype)
            complex_dtype = np.complex64 if real_dtype == np.dtype(np.float32) else np.complex128
            if not self.Fourier:
                if backend == 'cupy' and CUPY_AVAILABLE:
                    import cupy as cp
                    self.val = cp.zeros(self.shape+self.N, dtype=real_dtype)
                else:
                    self.val=np.zeros(self.shape+self.N, dtype=real_dtype)
            else:
                if backend == 'cupy' and CUPY_AVAILABLE:
                    import cupy as cp
                    self.val = cp.zeros(self.shape+self.N_fft, dtype=complex_dtype)
                else:
                    self.val=np.zeros(self.shape+self.N_fft, dtype=complex_dtype)

        else:
            raise ValueError('Initialization of Tensor.')

        self.dim=self.N.__len__()
        if Y is None:
            self.Y=np.ones(self.dim, dtype=float)
        else:
            self.Y=np.array(Y, dtype=float)

        # definition of __mul__ operation
        self.multype=multype

    def set_fft_form(self, fft_form=fft_form_default, copy=False):
        if copy:
            R=self.copy()
        else:
            R=self

        if self.fft_form==fft_form:
            return R

        fft_form_orig = self.fft_form
        if R.Fourier:
            if fft_form_orig in ['r']:
                nval=np.flip(R.val[...,1:].conj(),axis=-1)
                for ax in self.axes[:-1]:
                    N,F = np.split(nval, [1], axis=ax)
                    nval=np.concatenate((N, np.flip(F, axis=ax)), axis=ax)
                if R.N[-1] % 2 == 0:
                    nval=nval[...,1:]
                val=np.concatenate((R.val,nval), axis=-1)
                R.val=1./np.prod(R.N)*val # fft_form=0
                if fft_form in ['c']:
                    R.val=np.fft.fftshift(R.val, axes=R.axes)
            elif fft_form_orig in ['c']:
                R.val=np.fft.ifftshift(R.val, axes=R.axes) # common for fft_form in [0,'r']
                if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
            elif fft_form_orig in [0]:
                if fft_form in ['c']:
                    R.val=np.fft.fftshift(R.val, axes=R.axes)
                else: # if fft_form in ['r']:
                    R.val=R.val[...,:self.get_N_real(self.N)[-1]]*np.prod(self.N)
        R._set_fft(fft_form)
        return R

    def __neg__(self):
        return self.copy(name='-'+self.name[:10], val=-self.val)

    def __add__(self, x):
        if isinstance(x, Tensor):
            assert(self.Fourier==x.Fourier)
            assert(self.val.shape==x.val.shape)
            name='({0}+{1})'.format(self.name[:10], x.name[:10])
            xp = get_array_module(self.val, x.val)
            if xp is np:
                val = self.val + x.val
            else:
                val = to_backend_array(self.val, prefer_backend='cupy') + to_backend_array(x.val, prefer_backend='cupy')
            return self.copy(name=name, val=val)

        elif isinstance(x, np.ndarray) or isinstance(x, float):
            xp = get_array_module(self.val)
            if xp is np:
                return self.copy(val=self.val+x)
            return self.copy(val=self.val + to_backend_array(x, prefer_backend='cupy'))
        else:
            raise ValueError('Tensor.__add__')

    def __iadd__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val += to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val += x.val
        else:
            if is_cupy_array(self.val):
                self.val += to_backend_array(x, prefer_backend='cupy')
            else:
                self.val += x
        return self

    def __isub__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val -= to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val -= x.val
        else:
            if is_cupy_array(self.val):
                self.val -= to_backend_array(x, prefer_backend='cupy')
            else:
                self.val -= x
        return self

    def __imul__(self, x):
        if isinstance(x, Tensor):
            if is_cupy_array(self.val) or is_cupy_array(x.val):
                if not is_cupy_array(self.val):
                    self.val = to_backend_array(self.val, prefer_backend='cupy')
                self.val *= to_backend_array(x.val, prefer_backend='cupy')
            else:
                self.val *= x.val # Element-wise! (Hadamard)
        else:
            if is_cupy_array(self.val):
                self.val *= x
            else:
                self.val *= x # Scalar
        return self

    def __sub__(self, x):
        return self.__add__(-x)

    def __rmul__(self, x):
        if np.size(x)==1 or isinstance(x, float):
            return self.copy(val=x*self.val)
        else:
            raise ValueError()

    def __call__(self, *args, **kwargs):
        return self.__mul__(*args, **kwargs)

    def __mul__(self, Y, multype=None, *args, **kwargs):
        if multype is None:
            multype=self.multype
        X=self
        assert(X.Fourier==Y.Fourier)
        assert(X.fft_form==Y.fft_form)
        if multype in ['scal', 'scalar']:
            return scalar_product(X, Y)
        elif multype in [21, '21']:
            return einsum('ij...,j...->i...', X, Y)
        elif multype in ['sym21', 'symmetric21']:
            return sym21_product(X, Y)
        elif multype in ['sym21_indexed', 'indexed_sym21']:
            return indexed_sym21_product(X, Y)
        elif str(multype).strip().lower() in [
                'elasticity_hg1_direct', 'elasticity_g1_direct',
                'elasticity_hg2_direct', 'elasticity_g2_direct']:
            return elasticity_direct_product(X, Y)
        elif multype in [42, '42']:
            return einsum('ijkl...,kl...->ij...', X, Y)
        elif multype in [00, 'elementwise', 'hadamard']:
            return einsum('...,...->...', X, Y)
        elif multype in ['grad']:
            return einsum('i...,...->i...', X, Y)
        elif multype in ['div']:
            return einsum('i...,i...->...', X, Y)
        else:
            try:
                return einsum(multype, X, Y)
            except Exception as error:
                raise ValueError() from error

    def mean(self):
        """
        Mean of trigonometric polynomial of shape of macroscopic vector.

        Physical FP16/FP32 fields use FP64 accumulation. Promoting only the
        returned scalar would preserve the rounding of a single-precision
        reduction, which can dominate averaged stresses near convergence.
        """
        mean=np.zeros(self.shape)
        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                value = to_host_array(self.val[di][ind])
                mean[di]=np.real(value)/self.fft_coef
        elif (CUPY_AVAILABLE and is_cupy_array(self.val) and self.val.dtype.kind == 'f'
              and self.val.flags.c_contiguous and mean.size > 0):
            # One launch for all components: CuPy's dtype-converting mean runs
            # on a single block (~5 GB/s, 60 ms per stress at 240^3).
            from cufhom_core.general.cupy_warp_reduction import component_means
            mean[...] = component_means(self.val.reshape(mean.size, -1)).reshape(self.shape)
        else:
            for di in np.ndindex(*self.shape):
                value = self.val[di]
                xp = get_array_module(value)
                accumulator = np.float64 if value.dtype.kind == 'f' and value.dtype.itemsize < 8 else None
                mean[di]=_scalar_to_float(xp.mean(value, dtype=accumulator))
        return mean

    def add_mean(self, mean):
        assert(self.shape==mean.shape)

        if self.Fourier:
            ind=self.mean_index()
            for di in np.ndindex(*self.shape):
                self.val[di+ind]=mean[di]*self.fft_coef
        elif (is_cupy_array(self.val) and self.val.dtype.kind == 'f'
              and self.val.shape[:len(self.shape)] == tuple(self.shape)):
            # One broadcast launch; each component still adds its value
            # rounded to the field dtype, as the per-component loop did.
            values = np.asarray(mean, dtype=np.float64).astype(self.val.dtype)
            extra = self.val.ndim - len(self.shape)
            self.val += cp.asarray(values).reshape(tuple(self.shape) + (1,)*extra)
        else:
            for di in np.ndindex(*self.shape):
                self.val[di]+=mean[di]
        return self

    def zeros_like(self, name=None):
        if name is None:
            name='zeros({})'.format(self.name[:10])
        return self.copy(name=name, val=get_array_module(self.val).zeros_like(self.val))

    @property
    def axes(self): # axes for Fourier transform
        return tuple(range(self.order, self.order+self.dim))

    def enlarge(self, M):
        """
        It enlarges a trigonometric polynomial by adding zeros to the Fourier
        coefficients with high frequencies.
        """
        assert(self.Fourier)
        if np.allclose(self.N, M):
            return self
        else:
            fft_form=self.fft_form
            self.set_fft_form(fft_form='c')

            val = self.val
            for ii,ax in enumerate(self.axes):
                if self.N[ii]%2==0:
                    N0,C=np.split(val, [1], axis=ax)
                    N2=np.copy(N0)
                    for jj, axc in enumerate(self.axes):
                        if ax==axc:
                            continue
                        elif N2.shape[axc]%2==0:
                            N20,N2C=np.split(N2, [1], axis=axc)
                            N2=np.concatenate((N20, np.flip(N2C, axis=axc)), axis=axc)
                        else:
                            N2=np.flip(N2, axis=axc)
                    val=np.concatenate((0.5*N0,C,0.5*N2.conj()), axis=ax)

            # enlarging the centered part with odd N
            M = np.array(M, dtype=float)
            N = np.array(val.shape[self.order:], dtype=float)

            ibeg = np.ceil((M-N)/2).astype(int)
            iend = np.ceil((M+N)/2).astype(int)

            slc=self.order*[slice(None)]+[slice(ibeg[i],iend[i],1) for i in range(N.size)]
            newval = np.zeros(self.shape+tuple(M.astype(int)), dtype=self.val.dtype)
            newval[tuple(slc)]=val

            R=self.copy(val=newval, N=M, fft_form='c')
            return R.set_fft_form(fft_form=fft_form)

# @staticmethod
def einsum(str_operator, x, y):
    assert(x.Fourier==y.Fourier)
    assert(np.all(x.N==y.N))
    xp = get_array_module(x.val, y.val)
    if xp is np:
        val=np.einsum(str_operator, x.val, y.val)
    else:
        val=xp.einsum(str_operator, to_backend_array(x.val, prefer_backend='cupy'),
                      to_backend_array(y.val, prefer_backend='cupy'))
    order=len(val.shape)-len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=val, order=order)


def sym21_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    assert(x.val.shape[0] == 21)
    assert(y.val.shape[0] == 6)
    xp = get_array_module(x.val, y.val)
    xval = x.val if xp is np else to_backend_array(x.val, prefer_backend='cupy')
    yval = y.val if xp is np else to_backend_array(y.val, prefer_backend='cupy')

    out = xp.zeros((6,) + yval.shape[1:], dtype=xp.result_type(xval.dtype, yval.dtype))
    for kk, (ii, jj) in enumerate(SYM21_PAIRS):
        cval = xval[kk]
        out[ii] += cval * yval[jj]
        if ii != jj:
            out[jj] += cval * yval[ii]
    order = len(out.shape) - len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)


def _get_cupy_indexed_sym21_kernel(dtype=np.float32):
    global _CUPY_INDEXED_SYM21_KERNEL, _CUPY_INDEXED_SYM21_FP64_KERNEL
    fp64 = np.dtype(dtype) == np.dtype('float64')
    cached = _CUPY_INDEXED_SYM21_FP64_KERNEL if fp64 else _CUPY_INDEXED_SYM21_KERNEL
    if cached is not None:
        return cached
    if not CUPY_AVAILABLE or cp is None:
        return None

    code = r'''
    extern "C" __global__
    void indexed_sym21_apply(const int* __restrict__ material_ids,
                             const float* __restrict__ table,
                             const float* __restrict__ X,
                             float* __restrict__ Y,
                             const long batch,
                             const long spatial,
                             const long total,
                             const long materials) {
        const long idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= total) return;

        const long s = idx % spatial;
        const long b = idx / spatial;
        const long base = b * spatial + s;
        const long stride = batch * spatial;
        const int material_id = material_ids[s];
        if (material_id < 0 || material_id >= materials) {
            for (int c=0; c<6; c++) Y[c*stride+base] = nanf("");
            return;
        }
        const float* A = table + ((long)material_id) * 21;

        const float x0 = X[0 * stride + base];
        const float x1 = X[1 * stride + base];
        const float x2 = X[2 * stride + base];
        const float x3 = X[3 * stride + base];
        const float x4 = X[4 * stride + base];
        const float x5 = X[5 * stride + base];

        const float a00 = A[ 0];
        const float a01 = A[ 1];
        const float a02 = A[ 2];
        const float a03 = A[ 3];
        const float a04 = A[ 4];
        const float a05 = A[ 5];
        const float a11 = A[ 6];
        const float a12 = A[ 7];
        const float a13 = A[ 8];
        const float a14 = A[ 9];
        const float a15 = A[10];
        const float a22 = A[11];
        const float a23 = A[12];
        const float a24 = A[13];
        const float a25 = A[14];
        const float a33 = A[15];
        const float a34 = A[16];
        const float a35 = A[17];
        const float a44 = A[18];
        const float a45 = A[19];
        const float a55 = A[20];

        Y[0 * stride + base] = a00*x0 + a01*x1 + a02*x2 + a03*x3 + a04*x4 + a05*x5;
        Y[1 * stride + base] = a01*x0 + a11*x1 + a12*x2 + a13*x3 + a14*x4 + a15*x5;
        Y[2 * stride + base] = a02*x0 + a12*x1 + a22*x2 + a23*x3 + a24*x4 + a25*x5;
        Y[3 * stride + base] = a03*x0 + a13*x1 + a23*x2 + a33*x3 + a34*x4 + a35*x5;
        Y[4 * stride + base] = a04*x0 + a14*x1 + a24*x2 + a34*x3 + a44*x4 + a45*x5;
        Y[5 * stride + base] = a05*x0 + a15*x1 + a25*x2 + a35*x3 + a45*x4 + a55*x5;
    }

    '''
    try:
        if fp64:
            code = code.replace("float", "double").replace("nanf", "nan")
        kernel = cp.RawKernel(code, "indexed_sym21_apply")
        if fp64:
            _CUPY_INDEXED_SYM21_FP64_KERNEL = kernel
        else:
            _CUPY_INDEXED_SYM21_KERNEL = kernel
    except Exception:
        return None
    return kernel


def indexed_sym21_array(x, yval):
    if not (CUPY_AVAILABLE and cp is not None and is_cupy_array(yval)):
        return None
    material_ids = getattr(x, "val", None)
    table = getattr(x, "material_table", None)
    if not (is_cupy_array(material_ids) and is_cupy_array(table)):
        return None
    if yval.dtype not in (cp.float32, cp.float64) or table.dtype != yval.dtype:
        return None
    if (table.ndim != 2 or table.shape[0] == 0 or table.shape[1] != 21
            or material_ids.shape not in (tuple(x.N), (1,)+tuple(x.N))):
        raise ValueError('Invalid indexed sym21 material layout')
    if material_ids.dtype != cp.int32 or yval.shape[0] != 6:
        return None

    if yval.ndim == len(x.N) + 2:
        batch = int(yval.shape[1])
        spatial_shape = tuple(int(v) for v in yval.shape[2:])
        out_shape = (6, batch) + spatial_shape
    elif yval.ndim == len(x.N) + 1:
        batch = 1
        spatial_shape = tuple(int(v) for v in yval.shape[1:])
        yval = yval[:, None, ...]
        out_shape = (6,) + spatial_shape
    else:
        return None
    if spatial_shape != tuple(int(v) for v in x.N):
        return None

    ids = material_ids.reshape(-1)
    if not ids.flags.c_contiguous or not table.flags.c_contiguous:
        return None
    if not yval.flags.c_contiguous:
        yval = cp.ascontiguousarray(yval)

    spatial = int(np.prod(spatial_shape))
    total = int(batch * spatial)
    out_raw = cp.empty((6, batch) + spatial_shape, dtype=yval.dtype)
    kernel = _get_cupy_indexed_sym21_kernel(yval.dtype)
    if kernel is None:
        return None
    threads = 256
    blocks = (total + threads - 1) // threads
    try:
        kernel(
            (blocks,),
            (threads,),
            (
                ids,
                table,
                yval,
                out_raw,
                np.int64(batch),
                np.int64(spatial),
                np.int64(total),
                np.int64(table.shape[0]),
            ),
        )
    except Exception:
        return None
    return out_raw if len(out_shape) == len(spatial_shape) + 2 else out_raw[:, 0, ...]


def indexed_sym21_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    value = y.val
    # A caller may request host solutions while the indexed material stays on
    # CUDA. Postprocessing must upload that field before the GPU-only product.
    if cp is not None and isinstance(value, np.ndarray) and value.dtype in (np.float32, np.float64):
        value = cp.asarray(value)
    out = indexed_sym21_array(x, value)
    if out is None:
        raise NotImplementedError("indexed_sym21_product requiere campos y coeficientes CuPy de igual precision FP32/FP64.")
    order = len(out.shape) - len(x.N)
    return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)


def _get_cupy_elastic_direct_kernel(kind, projection, inplace=False):
    key = (kind, projection, bool(inplace))
    kernel = _CUPY_ELASTIC_DIRECT_KERNELS.get(key)
    if kernel is not None:
        return kernel

    if kind in {"f32_r32", "f64_r64"}:
        x_type = "float"
        y_type = "float"
        load_x = """
            const float x0r = X[0 * stride + base];
            const float x1r = X[1 * stride + base];
            const float x2r = X[2 * stride + base];
            const float x3r = X[3 * stride + base];
            const float x4r = X[4 * stride + base];
            const float x5r = X[5 * stride + base];
        """
        imag_defs = """
            const float x0i = 0.0f;
            const float x1i = 0.0f;
            const float x2i = 0.0f;
            const float x3i = 0.0f;
            const float x4i = 0.0f;
            const float x5i = 0.0f;
        """
        write_y = """
            Y[0 * stride + base] = y0r;
            Y[1 * stride + base] = y1r;
            Y[2 * stride + base] = y2r;
            Y[3 * stride + base] = y3r;
            Y[4 * stride + base] = y4r;
            Y[5 * stride + base] = y5r;
        """
    elif kind in {"f32_c64", "f64_c128"}:
        x_type = "float2"
        y_type = "float2"
        load_x = """
            const float2 x0 = X[0 * stride + base];
            const float2 x1 = X[1 * stride + base];
            const float2 x2 = X[2 * stride + base];
            const float2 x3 = X[3 * stride + base];
            const float2 x4 = X[4 * stride + base];
            const float2 x5 = X[5 * stride + base];
            const float x0r = x0.x; const float x0i = x0.y;
            const float x1r = x1.x; const float x1i = x1.y;
            const float x2r = x2.x; const float x2i = x2.y;
            const float x3r = x3.x; const float x3i = x3.y;
            const float x4r = x4.x; const float x4i = x4.y;
            const float x5r = x5.x; const float x5i = x5.y;
        """
        imag_defs = ""
        write_y = """
            Y[0 * stride + base] = make_float2(y0r, y0i);
            Y[1 * stride + base] = make_float2(y1r, y1i);
            Y[2 * stride + base] = make_float2(y2r, y2i);
            Y[3 * stride + base] = make_float2(y3r, y3i);
            Y[4 * stride + base] = make_float2(y4r, y4i);
            Y[5 * stride + base] = make_float2(y5r, y5i);
        """
    else:
        return None

    g2_expr = ""
    if projection == "g2":
        g2_expr = """
            y0r = x0r - y0r; y0i = x0i - y0i;
            y1r = x1r - y1r; y1i = x1i - y1i;
            y2r = x2r - y2r; y2i = x2i - y2i;
            y3r = x3r - y3r; y3i = x3i - y3i;
            y4r = x4r - y4r; y4i = x4i - y4i;
            y5r = x5r - y5r; y5i = x5i - y5i;
        """

    code = r'''
    extern "C" __global__
    void elasticity_direct_apply(const XTYPE* __restrict__ X,
                                 YTYPE* __restrict__ Y,
                                 const long batch,
                                 const long n0,
                                 const long n1,
                                 const long n2_data,
                                 const long n2_full,
                                 const float inv_y0,
                                 const float inv_y1,
                                 const float inv_y2,
                                 const long total) {
        const long idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= total) return;

        const long spatial = n0 * n1 * n2_data;
        const long s = idx % spatial;
        const long b = idx / spatial;
        const long i2 = s % n2_data;
        const long tmp = s / n2_data;
        const long i1 = tmp % n1;
        const long i0 = tmp / n1;
        const long base = b * spatial + s;
        const long stride = batch * spatial;

        LOAD_X
        IMAG_DEFS

        // Even-grid Nyquist indices differentiate the real trigonometric
        // interpolant: zero wavenumber on that axis. The mean has no
        // fluctuation; a vanishing wavevector elsewhere has G1 = 0 (G2 = I).
        const bool nyq0 = ((n0 % 2) == 0) && (i0 == n0 / 2);
        const bool nyq1 = ((n1 % 2) == 0) && (i1 == n1 / 2);
        const bool nyq2 = ((n2_full % 2) == 0) && (i2 == n2_full / 2);
        if (i0 == 0 && i1 == 0 && i2 == 0) {
            const float y0r = 0.0f; const float y0i = 0.0f;
            const float y1r = 0.0f; const float y1i = 0.0f;
            const float y2r = 0.0f; const float y2i = 0.0f;
            const float y3r = 0.0f; const float y3i = 0.0f;
            const float y4r = 0.0f; const float y4i = 0.0f;
            const float y5r = 0.0f; const float y5i = 0.0f;
            WRITE_Y
            return;
        }

        const bool is_rfft = n2_data != n2_full;
        const float k0_raw = nyq0 ? 0.0f : ((i0 <= n0 / 2) ? (float)i0 : (float)(i0 - n0));
        const float k1_raw = nyq1 ? 0.0f : ((i1 <= n1 / 2) ? (float)i1 : (float)(i1 - n1));
        const float k2_raw = nyq2 ? 0.0f : ((is_rfft || i2 <= n2_full / 2) ? (float)i2 : (float)(i2 - n2_full));
        const float k0 = k0_raw * inv_y0;
        const float k1 = k1_raw * inv_y1;
        const float k2 = k2_raw * inv_y2;

        const float q00 = k0 * k0;
        const float q11 = k1 * k1;
        const float q22 = k2 * k2;
        const float q01 = k0 * k1;
        const float q02 = k0 * k2;
        const float q12 = k1 * k2;
        if (q00 + q11 + q22 == 0.0f) {
            float y0r = 0.0f; float y0i = 0.0f;
            float y1r = 0.0f; float y1i = 0.0f;
            float y2r = 0.0f; float y2i = 0.0f;
            float y3r = 0.0f; float y3i = 0.0f;
            float y4r = 0.0f; float y4i = 0.0f;
            float y5r = 0.0f; float y5i = 0.0f;
            G2_EXPR
            WRITE_Y
            return;
        }
        const float invn2 = 1.0f / (q00 + q11 + q22);
        const float invn4 = invn2 * invn2;
        const float rt2 = 1.4142135623730951f;

        const float a00 = 2.0f*q00*invn2 - q00*q00*invn4;
        const float a01 = -q00*q11*invn4;
        const float a02 = -q00*q22*invn4;
        const float a03 = -rt2*q00*q12*invn4;
        const float a04 = rt2*q02*invn2 - rt2*q00*q02*invn4;
        const float a05 = rt2*q01*invn2 - rt2*q00*q01*invn4;

        const float a11 = 2.0f*q11*invn2 - q11*q11*invn4;
        const float a12 = -q11*q22*invn4;
        const float a13 = rt2*q12*invn2 - rt2*q11*q12*invn4;
        const float a14 = -rt2*q11*q02*invn4;
        const float a15 = rt2*q01*invn2 - rt2*q11*q01*invn4;

        const float a22 = 2.0f*q22*invn2 - q22*q22*invn4;
        const float a23 = rt2*q12*invn2 - rt2*q22*q12*invn4;
        const float a24 = rt2*q02*invn2 - rt2*q22*q02*invn4;
        const float a25 = -rt2*q22*q01*invn4;

        const float a33 = (1.0f - q00*invn2) - 2.0f*q12*q12*invn4;
        const float a34 = q01*invn2 - 2.0f*q12*q02*invn4;
        const float a35 = q02*invn2 - 2.0f*q12*q01*invn4;
        const float a44 = (1.0f - q11*invn2) - 2.0f*q02*q02*invn4;
        const float a45 = q12*invn2 - 2.0f*q02*q01*invn4;
        const float a55 = (1.0f - q22*invn2) - 2.0f*q01*q01*invn4;

        float y0r = a00*x0r + a01*x1r + a02*x2r + a03*x3r + a04*x4r + a05*x5r;
        float y1r = a01*x0r + a11*x1r + a12*x2r + a13*x3r + a14*x4r + a15*x5r;
        float y2r = a02*x0r + a12*x1r + a22*x2r + a23*x3r + a24*x4r + a25*x5r;
        float y3r = a03*x0r + a13*x1r + a23*x2r + a33*x3r + a34*x4r + a35*x5r;
        float y4r = a04*x0r + a14*x1r + a24*x2r + a34*x3r + a44*x4r + a45*x5r;
        float y5r = a05*x0r + a15*x1r + a25*x2r + a35*x3r + a45*x4r + a55*x5r;

        float y0i = a00*x0i + a01*x1i + a02*x2i + a03*x3i + a04*x4i + a05*x5i;
        float y1i = a01*x0i + a11*x1i + a12*x2i + a13*x3i + a14*x4i + a15*x5i;
        float y2i = a02*x0i + a12*x1i + a22*x2i + a23*x3i + a24*x4i + a25*x5i;
        float y3i = a03*x0i + a13*x1i + a23*x2i + a33*x3i + a34*x4i + a35*x5i;
        float y4i = a04*x0i + a14*x1i + a24*x2i + a34*x3i + a44*x4i + a45*x5i;
        float y5i = a05*x0i + a15*x1i + a25*x2i + a35*x3i + a45*x4i + a55*x5i;

        G2_EXPR
        WRITE_Y
    }
    '''
    code = (
        code.replace("XTYPE", x_type)
            .replace("YTYPE", y_type)
            .replace("LOAD_X", load_x)
            .replace("IMAG_DEFS", imag_defs)
            .replace("WRITE_Y", write_y)
            .replace("G2_EXPR", g2_expr)
    )
    if kind.startswith('f64_'):
        # Share the formula, but not rounded float literals, with the FP32 path.
        code = code.replace('float', 'double')
        code = re.sub(r'(?<=\d)f\b', '', code)
    if inplace:
        # Each thread reads all six components of its own frequency before
        # writing them. Aliasing is safe only without the restrict contract.
        code = code.replace('__restrict__', '')

    try:
        kernel = cp.RawKernel(code, "elasticity_direct_apply")
    except Exception:
        return None
    _CUPY_ELASTIC_DIRECT_KERNELS[key] = kernel
    return kernel


def elasticity_direct_array(x, yval, *, out=None):
    if not (CUPY_AVAILABLE and cp is not None and is_cupy_array(yval)):
        return None
    if yval.shape[0] != 6:
        return None
    if yval.dtype == cp.float32:
        kind = "f32_r32"
        out_dtype = cp.float32
    elif yval.dtype == cp.complex64:
        kind = "f32_c64"
        out_dtype = cp.complex64
    elif yval.dtype == cp.float64:
        kind = "f64_r64"
        out_dtype = cp.float64
    elif yval.dtype == cp.complex128:
        kind = "f64_c128"
        out_dtype = cp.complex128
    else:
        return None

    multype = str(getattr(x, "multype", "")).strip().lower()
    if multype in {"elasticity_hg1_direct", "elasticity_g1_direct"}:
        projection = "g1"
    elif multype in {"elasticity_hg2_direct", "elasticity_g2_direct"}:
        projection = "g2"
    else:
        return None

    if yval.ndim == len(x.N) + 2:
        batch = int(yval.shape[1])
        spatial_shape = tuple(int(v) for v in yval.shape[2:])
        out_shape = (6, batch) + spatial_shape
    elif yval.ndim == len(x.N) + 1:
        batch = 1
        spatial_shape = tuple(int(v) for v in yval.shape[1:])
        yval = yval[:, None, ...]
        out_shape = (6,) + spatial_shape
    else:
        return None

    full_shape = tuple(int(v) for v in x.N)
    if getattr(x, "fft_form", None) == 'r':
        expected_shape = full_shape[:-1] + (int(full_shape[-1] // 2 + 1),)
    else:
        expected_shape = full_shape
    if spatial_shape != expected_shape:
        return None
    if len(spatial_shape) != 3:
        return None
    if not yval.flags.c_contiguous:
        yval = cp.ascontiguousarray(yval)

    inplace = out is not None
    if inplace and (not is_cupy_array(out) or out.dtype != out_dtype
                    or out.shape != out_shape or not out.flags.c_contiguous
                    or out.data.ptr != yval.data.ptr):
        raise ValueError('Direct projection out must be the contiguous input array')
    kernel = _get_cupy_elastic_direct_kernel(kind, projection, inplace=inplace)
    if kernel is None:
        return None

    n0, n1, n2_data = spatial_shape
    n2_full = int(full_shape[2])
    spatial = int(n0 * n1 * n2_data)
    out_raw = out.reshape((6, batch)+spatial_shape) if inplace else cp.empty((6, batch)+spatial_shape, dtype=out_dtype)
    total = int(batch * spatial)
    threads = 256
    blocks = (total + threads - 1) // threads
    inv_y = [float(1.0 / yy) for yy in getattr(x, "Y", np.ones(3, dtype=float))]
    scalar_dtype = np.float64 if kind.startswith('f64_') else np.float32

    try:
        kernel(
            (blocks,),
            (threads,),
            (
                yval,
                out_raw,
                np.int64(batch),
                np.int64(n0),
                np.int64(n1),
                np.int64(n2_data),
                np.int64(n2_full),
                scalar_dtype(inv_y[0]),
                scalar_dtype(inv_y[1]),
                scalar_dtype(inv_y[2]),
                np.int64(total),
            ),
        )
    except Exception:
        return None
    return out_raw if len(out_shape) == len(spatial_shape) + 2 else out_raw[:, 0, ...]


def elasticity_direct_product(x, y):
    assert(x.Fourier == y.Fourier)
    assert(np.all(x.N == y.N))
    assert(y.val.shape[0] == 6)
    xp = get_array_module(y.val)
    if xp is not np:
        yval = to_backend_array(y.val, prefer_backend='cupy')
        out = elasticity_direct_array(x, yval)
        if out is not None:
            order = len(out.shape) - len(x.N)
            return y.copy(name='{0}({1})'.format(x.name, y.name), val=out, order=order)
    raise NotImplementedError("elasticity_direct_product requiere CuPy y arreglos GPU.")

def scalar_product(y, x):
    assert(isinstance(x, Tensor))
    assert(y.val.shape==x.val.shape)
    assert(y.fft_form==x.fft_form)

    xp = get_array_module(y.val, x.val)
    if xp is np:
        yval = y.val
        xval = x.val
    else:
        yval = to_backend_array(y.val, prefer_backend='cupy')
        xval = to_backend_array(x.val, prefer_backend='cupy')

    if y.Fourier:
        if x.fft_form in ['r']:
            if x.N[-1] % 2 == 1:
                if xp is np:
                    scal=(np.sum(yval[...,0]*np.conj(xval[...,0])).real +
                          2*np.sum(yval[...,1:]*np.conj(xval[...,1:])).real)/np.prod(y.N)**2
                else:
                    scal=(xp.sum(yval[...,0]*xp.conj(xval[...,0])).real +
                          2*xp.sum(yval[...,1:]*xp.conj(xval[...,1:])).real)/np.prod(y.N)**2
            else:
                if xp is np:
                    scal=(np.sum(yval[...,0]*np.conj(xval[...,0])).real +
                          np.sum(yval[...,-1]*np.conj(xval[...,-1])).real +
                          2*np.sum(yval[...,1:-1]*np.conj(xval[...,1:-1])).real)/np.prod(y.N)**2
                else:
                    scal=(xp.sum(yval[...,0]*xp.conj(xval[...,0])).real +
                          xp.sum(yval[...,-1]*xp.conj(xval[...,-1])).real +
                          2*xp.sum(yval[...,1:-1]*xp.conj(xval[...,1:-1])).real)/np.prod(y.N)**2
        else:
            if xp is np:
                scal=np.sum(yval[:]*np.conj(xval[:])).real
            else:
                scal=xp.sum(yval[:]*xp.conj(xval[:])).real
    else:
        if xp is np:
            scal=np.sum(yval[:]*xval[:])/np.prod(y.N)
        else:
            scal=xp.sum(yval[:]*xval[:])/np.prod(y.N)
    return _scalar_to_float(scal)
