"""Shared conjugate gradients with an audit of the true residual."""

from cufhom_core.tensors.fft import cp


class ConvergenceError(RuntimeError):
    """An iterative solve failed its convergence or acceptance contract."""


_CUPY_CG_UPDATE_KERNELS = None


def _cg_updates():
    global _CUPY_CG_UPDATE_KERNELS
    if _CUPY_CG_UPDATE_KERNELS is None:
        _CUPY_CG_UPDATE_KERNELS = (
            cp.ElementwiseKernel(
                "T p, T ap, float64 alpha",
                "T x, T r",
                "const T a=(T)alpha; x += a*p; r -= a*ap;",
                "cg_update_xr",
            ),
            cp.ElementwiseKernel(
                "T r, float64 beta",
                "T p",
                "p = r + (T)beta*p;",
                "cg_update_p",
            ),
        )
    return _CUPY_CG_UPDATE_KERNELS
