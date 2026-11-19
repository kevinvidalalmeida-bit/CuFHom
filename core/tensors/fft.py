from __future__ import annotations

import numpy as np
import scipy.fft as scipy_fft

try:
    import pyfftw.interfaces.scipy_fft as pyfftw_fft
    import pyfftw.interfaces.cache as pyfftw_cache

    pyfftw_cache.enable()
    PYFFTW_AVAILABLE = True
except ImportError:
    pyfftw_fft = None
    pyfftw_cache = None
    PYFFTW_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.fft as cupy_fft
    from cupyx.scipy.fft import get_fft_plan as cupy_get_fft_plan

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cupy_fft = None
    cupy_get_fft_plan = None
    CUPY_AVAILABLE = False


FFT_WORKERS = 1
FFT_BACKEND = "scipy"
CUPY_PLAN_MODE = "auto"  # auto | manual | none
_CUPY_PLAN_CACHE = {}


def set_fft_backend(name):
    global FFT_BACKEND
    if name not in {"scipy", "pyfftw", "auto", "cupy"}:
        raise ValueError("FFT backend debe ser 'scipy', 'pyfftw', 'cupy' o 'auto'.")
    FFT_BACKEND = name


def set_cupy_plan_mode(mode, *, preserve_cache=False):
    """Set CuPy FFT planning mode, optionally retaining compatible plans.

    A fixed-grid sequence of GPU solves repeatedly requests ``manual``.  It
    is safe to retain the existing plans in that case, but never across a
    mode change.  The default preserves the historical clearing behaviour.
    """
    global CUPY_PLAN_MODE
    if mode not in {"auto", "manual", "none"}:
        raise ValueError("CUPY_PLAN_MODE debe ser 'auto', 'manual' o 'none'.")
    if preserve_cache and mode == CUPY_PLAN_MODE:
        if CUPY_AVAILABLE:
            cp.fft.config.enable_nd_planning = mode != "none"
        return
    CUPY_PLAN_MODE = mode
    _CUPY_PLAN_CACHE.clear()
    if CUPY_AVAILABLE:
        cp.fft.config.enable_nd_planning = mode != "none"


def get_fft_backend():
    if FFT_BACKEND == "auto":
        if PYFFTW_AVAILABLE:
            return "pyfftw"
        return "scipy"
    if FFT_BACKEND == "pyfftw" and not PYFFTW_AVAILABLE:
        raise RuntimeError("FFT_BACKEND='pyfftw' pero pyfftw no esta instalado.")
    if FFT_BACKEND == "cupy" and not CUPY_AVAILABLE:
        raise RuntimeError("FFT_BACKEND='cupy' pero cupy no esta instalado.")
    return FFT_BACKEND


def get_fft_backend_status():
    return {
        "requested": FFT_BACKEND,
        "resolved": get_fft_backend(),
        "pyfftw_available": PYFFTW_AVAILABLE,
        "cupy_available": CUPY_AVAILABLE,
        "cupy_plan_mode": CUPY_PLAN_MODE,
    }


def is_cupy_array(x) -> bool:
    return bool(CUPY_AVAILABLE and isinstance(x, cp.ndarray))


def is_array_type(x) -> bool:
    if isinstance(x, np.ndarray):
        return True
    return is_cupy_array(x)


def _has_cupy_payload(obj) -> bool:
    if is_cupy_array(obj):
        return True
    return is_cupy_array(getattr(obj, "val", None))


def cupy_synchronize(*objs) -> bool:
    if not CUPY_AVAILABLE:
        return False

    if objs:
        needs_sync = any(_has_cupy_payload(obj) for obj in objs)
    else:
        try:
            needs_sync = get_fft_backend() == "cupy"
        except Exception:
            needs_sync = False

    if not needs_sync:
        return False

    try:
        cp.cuda.get_current_stream().synchronize()
    except Exception:
        cp.cuda.Stream.null.synchronize()
    return True


def get_array_module(*xs):
    for x in xs:
        if is_cupy_array(x):
            return cp
    return np


def to_backend_array(x, prefer_backend: str | None = None, copy: bool = False):
    if prefer_backend is None:
        prefer_backend = get_fft_backend()

    if prefer_backend == "cupy":
        if not CUPY_AVAILABLE:
            raise RuntimeError("Backend CuPy solicitado pero cupy no esta disponible.")
        if is_cupy_array(x):
            return cp.array(x, copy=copy) if copy else x
        return cp.array(x, copy=copy) if copy else cp.asarray(x)

    if is_cupy_array(x):
        arr = cp.asnumpy(x)
        return np.array(arr, copy=copy) if copy else arr
    return np.array(x, copy=copy) if copy else np.asarray(x)


def to_host_array(x, copy: bool = False):
    if is_cupy_array(x):
        arr = cp.asnumpy(x)
        return np.array(arr, copy=copy) if copy else arr
    return np.array(x, copy=copy) if copy else np.asarray(x)


def _cupy_fft_plan_key(x, axes, value_type, shape=None):
    input_shape = tuple(int(v) for v in x.shape)
    transform_shape = None if shape is None else tuple(int(v) for v in shape)
    # Plans and their workspaces belong to one CUDA device.
    return (int(x.device.id), input_shape, str(x.dtype), tuple(axes), value_type, transform_shape)


def _maybe_cupy_plan(x, axes, value_type, shape=None):
    if not is_cupy_array(x) or CUPY_PLAN_MODE != "manual":
        return None

    key = _cupy_fft_plan_key(x, axes, value_type, shape=shape)
    plan = _CUPY_PLAN_CACHE.get(key)
    if plan is None:
        plan = cupy_get_fft_plan(
            x,
            shape=shape,
            axes=axes,
            value_type=value_type,
        )
        _CUPY_PLAN_CACHE[key] = plan
    return plan


def _fft_module_for_value(x):
    backend = get_fft_backend()
    if backend == "cupy" or is_cupy_array(x):
        if not CUPY_AVAILABLE:
            raise RuntimeError("Backend CuPy solicitado pero cupy no esta disponible.")
        return cupy_fft
    if backend == "pyfftw":
        return pyfftw_fft
    return scipy_fft


def _cupy_complex_dtype(x):
    if not is_cupy_array(x):
        return None
    return cp.complex64 if x.dtype in (cp.float32, cp.complex64) else cp.complex128


def fftn_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * out


def fftn_unscaled_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        return fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    return fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def ifftn_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x.astype(_cupy_complex_dtype(x), copy=False), axes, "C2C")
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(x, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real * prodN


def ifftn_unscaled_real_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        plan = _maybe_cupy_plan(x.astype(_cupy_complex_dtype(x), copy=False), axes, "C2C")
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(x, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real


def rfftn_cached(x, N_tuple, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        if _component_fft_required(x,N_tuple,axes):
            return _component_real_fft(x,N_tuple,inverse=False)
        plan = _maybe_cupy_plan(x, axes, "R2C")
        return fft_mod.rfftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.rfftn(x, s=N_tuple, axes=axes)
    return fft_mod.rfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def irfftn_cached(x, N_tuple, axes, *, overwrite_x=False):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        if _component_fft_required(x,N_tuple,axes):
            return _component_real_fft(x,N_tuple,inverse=True,normalized=True,
                                       preserve_input=not overwrite_x)
        plan = _maybe_cupy_plan(x, axes, "C2R", shape=N_tuple)
        return fft_mod.irfftn(x, s=N_tuple, axes=axes, plan=plan, overwrite_x=overwrite_x) if plan else fft_mod.irfftn(x, s=N_tuple, axes=axes, overwrite_x=overwrite_x)
    return fft_mod.irfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def irfftn_unscaled_cached(x, N_tuple, axes):
    """Execute a CuPy C2R plan without its per-call ``1/prod(N)`` scaling.

    ``cupy.fft.irfftn`` implements the normalized mathematical inverse with a
    full-array divide kernel.  In the projected operator that scale can be
    absorbed once into the (constant) Fourier projection tensor, so invoking
    the native cuFFT plan directly removes the divide from every CG matvec.
    The fallback preserves the ordinary CuPy result multiplied by ``prodN``.
    """
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        if _component_fft_required(x,N_tuple,axes):
            return _component_real_fft(x,N_tuple,inverse=True)
        plan = _maybe_cupy_plan(x, axes, "C2R", shape=N_tuple)
        if plan is not None and hasattr(plan, "fft"):
            out_shape = list(x.shape)
            for axis, size in zip(axes, N_tuple):
                out_shape[axis] = int(size)
            real_dtype = cp.float32 if x.dtype == cp.complex64 else cp.float64
            out = cp.empty(tuple(out_shape), dtype=real_dtype)
            try:
                plan.fft(x, out, cp.cuda.cufft.CUFFT_INVERSE)
                return out
            except Exception:
                # Some older CuPy plan variants do not expose a compatible
                # low-level C2R execution path. Keep correctness first.
                pass
        normalized = (
            fft_mod.irfftn(x, s=N_tuple, axes=axes, plan=plan)
            if plan else fft_mod.irfftn(x, s=N_tuple, axes=axes)
        )
        return normalized * int(np.prod(N_tuple))
    return fft_mod.irfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS) * int(np.prod(N_tuple))


def _component_fft_required(x,shape,axes):
    """Bound cuFFT workspaces for six large contiguous Mandel fields."""
    if (x.ndim!=4 or x.shape[0]!=6 or len(shape)!=3
            or tuple(a%x.ndim for a in axes)!=(1,2,3)
            or x.dtype not in (cp.float32,cp.complex64,cp.float64,cp.complex128) or not x.flags.c_contiguous):
        return False
    from cufhom_core.general.cupy_capacity import library_capacity_required
    real_dtype=np.empty((),dtype=x.dtype).real.dtype
    return library_capacity_required(shape,real_dtype)


def _component_real_fft(x,shape,*,inverse,normalized=False,preserve_input=False):
    """Same 3-D library FFTs with one-component plans, not a generated FFT.

    The output is still a full six-component field. Only cuFFT batch/workspace
    storage changes. Normalized inverses preserve caller inputs by copying
    one component at a time when overwrite permission was not given.
    """
    shape=tuple(int(n) for n in shape)
    axes=(0,1,2)
    target=(6,)+ (shape if inverse else shape[:-1]+(shape[-1]//2+1,))
    real_dtype=cp.float32 if x.dtype in (cp.float32,cp.complex64) else cp.float64
    complex_dtype=cp.complex64 if real_dtype==cp.float32 else cp.complex128
    out=cp.empty(target,dtype=real_dtype if inverse else complex_dtype)
    for c in range(6):
        row=x[c].copy() if inverse and preserve_input else x[c]
        kind='C2R' if inverse else 'R2C'
        plan_shape=shape if inverse else None
        key=_cupy_fft_plan_key(row,axes,kind,shape=plan_shape)
        if CUPY_PLAN_MODE=='manual' and key not in _CUPY_PLAN_CACHE:
            # A small persistent workspace must not split a reusable multi-GiB
            # field block in the pool. Allocate plan-owned workspace directly;
            # the cached plan retains it and repeated transforms reuse it.
            cp.cuda.get_current_stream().synchronize()
            from cufhom_core.general.cupy_capacity import current_memory_pool
            current_memory_pool().free_all_blocks()
            with cp.cuda.using_allocator():
                plan=_maybe_cupy_plan(row,axes,kind,shape=plan_shape)
        else:
            plan=_maybe_cupy_plan(row,axes,kind,shape=plan_shape)
        if plan is not None and hasattr(plan,'fft'):
            plan.fft(row,out[c],cp.cuda.cufft.CUFFT_INVERSE if inverse else cp.cuda.cufft.CUFFT_FORWARD)
            if inverse and normalized:
                out[c]/=int(np.prod(shape))
        elif inverse:
            value=cupy_fft.irfftn(row,s=shape,axes=axes,plan=plan,overwrite_x=not preserve_input)
            out[c]=value if normalized else value*int(np.prod(shape))
        else:
            out[c]=cupy_fft.rfftn(row,s=shape,axes=axes,plan=plan)
    return out
