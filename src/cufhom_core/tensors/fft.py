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
CUPY_RFFT_BATCH_MODE = "auto"  # auto | full | component
_CUPY_PLAN_CACHE = {}
_COMPONENT_FFT_DEVICES = {}


def set_cupy_rfft_batch_mode(mode):
    """Select execution batching, never change the transform/discretization.

    Auto uses the measured component policy for six float32 240^3 fields on
    an RTX 5070 Ti. Other configurations retain full batches. Component mode
    permits explicit benchmarking on other compatible grids.
    """
    global CUPY_RFFT_BATCH_MODE
    if mode not in {"auto", "full", "component"}:
        raise ValueError("cupy_rfft_batch_mode must be auto, full or component")
    CUPY_RFFT_BATCH_MODE = mode


def _component_fft_device_supported():
    device = cp.cuda.runtime.getDevice()
    if device not in _COMPONENT_FFT_DEVICES:
        name = cp.cuda.runtime.getDeviceProperties(device)['name']
        if isinstance(name, bytes):
            name = name.decode()
        _COMPONENT_FFT_DEVICES[device] = name == 'NVIDIA GeForce RTX 5070 Ti'
    return _COMPONENT_FFT_DEVICES[device]


def _use_component_fft(x, shape, axes, inverse):
    # Called only for CuPy arrays. Odd-volume real component offsets do not
    # satisfy cuFFT complex alignment, even when the full array is contiguous.
    if CUPY_RFFT_BATCH_MODE == 'full' or CUPY_PLAN_MODE != 'manual':
        return False
    if len(shape) != 3 or x.ndim != 4 or x.shape[0] != 6 or not x.flags.c_contiguous:
        return False
    if tuple(a % 4 for a in axes) != (1, 2, 3):
        return False
    shape = tuple(shape)
    expected = shape[:-1]+(shape[-1]//2+1,) if inverse else shape
    allowed = (cp.complex64, cp.complex128) if inverse else (cp.float32, cp.float64)
    if x.shape[1:] != expected or x.dtype not in allowed:
        return False
    alignment = x.dtype.itemsize if inverse else 2*x.dtype.itemsize
    if (shape[0]*shape[1]*shape[2]) % 2 or x.data.ptr % alignment:
        return False
    if CUPY_RFFT_BATCH_MODE == 'auto':
        single = cp.complex64 if inverse else cp.float32
        return shape == (240, 240, 240) and x.dtype == single and _component_fft_device_supported()
    return True


def _component_fft(x, shape, inverse):
    """Same I_6 tensor F transform, six sequential launches of one cached plan.

    Each output owns its storage. As for the existing raw C2R path, inverse
    input is consumable. No copy/normalization pass is introduced. All launches
    use the current stream; no concurrent use of a shared plan/workspace.
    """
    shape = tuple(shape)
    dtype = x.real.dtype if inverse else (cp.complex64 if x.dtype == cp.float32 else cp.complex128)
    out_shape = shape if inverse else shape[:-1]+(shape[-1]//2+1,)
    out = cp.empty((6,)+out_shape, dtype=dtype)
    plan = _maybe_cupy_plan(x[:1], (1, 2, 3), 'C2R' if inverse else 'R2C',
                            shape=shape if inverse else None)
    for component in range(6):
        plan.fft(x[component:component+1], out[component:component+1],
                 cp.cuda.cufft.CUFFT_INVERSE if inverse else cp.cuda.cufft.CUFFT_FORWARD)
    return out


def set_fft_workers(n):
    global FFT_WORKERS
    FFT_WORKERS = max(1, int(n))


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
        "cupy_rfft_batch_mode": CUPY_RFFT_BATCH_MODE,
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


def _array_module_shift(x):
    return cp.fft if is_cupy_array(x) else np.fft


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


def _normalize_fft_args(N):
    N_tuple = tuple(int(v) for v in np.asarray(N, dtype=np.int32).tolist())
    prodN = int(np.prod(N_tuple))
    axes = tuple(range(-len(N_tuple), 0))
    return N_tuple, prodN, axes


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


def _fftshift_like(x, arr, axes):
    shift_mod = _array_module_shift(x)
    return shift_mod.fftshift(arr, axes=axes)


def _ifftshift_like(x, arr, axes):
    shift_mod = _array_module_shift(x)
    return shift_mod.ifftshift(arr, axes=axes)


def _cupy_complex_dtype(x):
    if not is_cupy_array(x):
        return None
    return cp.complex64 if x.dtype in (cp.float32, cp.complex64) else cp.complex128


def cfftnc_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    shifted = _ifftshift_like(x, x, axes)
    if is_cupy_array(shifted):
        x_fft = shifted.astype(_cupy_complex_dtype(shifted), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * _fftshift_like(x, out, axes)


def icfftnc_cached(Fx, N_tuple, prodN, axes):
    Fx = to_backend_array(Fx, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(Fx)
    shifted = _ifftshift_like(Fx, Fx, axes)
    if is_cupy_array(shifted):
        plan = _maybe_cupy_plan(shifted.astype(_cupy_complex_dtype(shifted), copy=False), axes, "C2C")
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(shifted, s=N_tuple, axes=axes)
        return _fftshift_like(Fx, out, axes).real * prodN
    out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return _fftshift_like(Fx, out, axes).real * prodN


def fftnc_cached(x, N_tuple, prodN, axes):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
        x_fft = x.astype(_cupy_complex_dtype(x), copy=False)
        plan = _maybe_cupy_plan(x_fft, axes, "C2C")
        out = fft_mod.fftn(x_fft, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.fftn(x_fft, s=N_tuple, axes=axes)
    else:
        out = fft_mod.fftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return (1.0 / prodN) * _fftshift_like(x, out, axes)


def icfftn_cached(Fx, N_tuple, prodN, axes):
    Fx = to_backend_array(Fx, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(Fx)
    shifted = _ifftshift_like(Fx, Fx, axes)
    if is_cupy_array(shifted):
        plan = _maybe_cupy_plan(shifted.astype(_cupy_complex_dtype(shifted), copy=False), axes, "C2C")
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.ifftn(shifted, s=N_tuple, axes=axes)
    else:
        out = fft_mod.ifftn(shifted, s=N_tuple, axes=axes, workers=FFT_WORKERS)
    return out.real * prodN


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
        if _use_component_fft(x, N_tuple, axes, False):
            return _component_fft(x, N_tuple, False)
        plan = _maybe_cupy_plan(x, axes, "R2C")
        return fft_mod.rfftn(x, s=N_tuple, axes=axes, plan=plan) if plan else fft_mod.rfftn(x, s=N_tuple, axes=axes)
    return fft_mod.rfftn(x, s=N_tuple, axes=axes, workers=FFT_WORKERS)


def irfftn_cached(x, N_tuple, axes, *, overwrite_x=False):
    x = to_backend_array(x, prefer_backend=get_fft_backend())
    fft_mod = _fft_module_for_value(x)
    if is_cupy_array(x):
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
        if _use_component_fft(x, N_tuple, axes, True):
            return _component_fft(x, N_tuple, True)
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


def cfftnc(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return cfftnc_cached(x, N_tuple, prodN, axes)


def icfftnc(Fx, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return icfftnc_cached(Fx, N_tuple, prodN, axes)


def fftnc(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return fftnc_cached(x, N_tuple, prodN, axes)


def icfftn(Fx, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return icfftn_cached(Fx, N_tuple, prodN, axes)


def fftn(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return fftn_cached(x, N_tuple, prodN, axes)


def ifftn(x, N):
    N_tuple, prodN, axes = _normalize_fft_args(N)
    return ifftn_cached(x, N_tuple, prodN, axes)


def rfftn(x, N):
    N_tuple, _, axes = _normalize_fft_args(N)
    return rfftn_cached(x, N_tuple, axes)


def irfftn(x, N):
    N_tuple, _, axes = _normalize_fft_args(N)
    return irfftn_cached(x, N_tuple, axes)
