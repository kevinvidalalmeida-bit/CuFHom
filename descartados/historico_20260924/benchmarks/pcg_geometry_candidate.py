"""Benchmark-only constant-grid specialization of the exact fused PCG kernels."""
from contextlib import contextmanager
from unittest.mock import patch
from profile_g09_fft import cp
from pipeline.nonlinear_krylov import FusedCoordinatePCG, _CODE

_INIT = FusedCoordinatePCG.__init__
_MODULES = {}


def specialized_init(engine, coordinates):
    _INIT(engine, coordinates)
    key = coordinates.hshape
    if key not in _MODULES:
        code = _CODE.replace('long long n0,long long n1,long long nz',
            'long long unused_n0,long long unused_n1,long long unused_nz')
        declarations = '\n'.join(f'constexpr long long {name}={n};'
            for name,n in zip(('n0','n1','nz'), key))
        code = code.replace(' {\n', ' {\n'+declarations+'\n')
        _MODULES[key] = cp.RawModule(code=code)
    engine.update_kernel = _MODULES[key].get_function('update_energy')
    engine.direction_kernel = _MODULES[key].get_function('direction')


@contextmanager
def installed():
    with patch.object(FusedCoordinatePCG, '__init__', specialized_init):
        yield
