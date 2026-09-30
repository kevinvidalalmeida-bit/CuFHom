"""CG must remove work without changing warm starts or convergence semantics."""
from pathlib import Path
import sys

import numpy as np
import pytest

CORE = Path(__file__).resolve().parents[1] / 'FFT/ffthompy_core/ffthompy'
sys.path.insert(0, str(CORE))
from ffthompy.general.solver import CG, linear_solver
from ffthompy.tensors import Tensor


@pytest.mark.parametrize('initial', [None, 'zero', 'nonzero', 'solution'])
def test_counted_operator_and_warm_starts(initial):
    diagonal = np.array([1., 2., 4., 7.])
    rhs = np.array([1., -3., 2., 4.])
    expected = rhs / diagonal
    guess = {'zero': np.zeros(4), 'nonzero': np.ones(4),
             'solution': expected.copy()}.get(initial)
    calls = []
    def apply(x):
        calls.append(x.copy())
        return diagonal*x
    x, info = linear_solver(apply, rhs, x0=guess, solver='cg', par={
        'rtol': 1e-12, 'maxiter': 20, 'profile_cg_timing': True,
        'check_true_residual': True})
    np.testing.assert_allclose(x, expected, atol=1e-12)
    np.testing.assert_array_equal(rhs, [1., -3., 2., 4.])
    assert info['true_residual_converged']
    assert len(calls) == info['kit'] + (initial is not None) + 1
    assert info['cg_profile']['update_p']['calls'] == max(0, info['kit']-1)


def test_zero_rhs_never_applies_operator():
    def fail(x):
        raise AssertionError('A(0) is redundant')
    x, info = CG(fail, np.zeros(4), par={'rtol': 1e-8})
    assert info['kit'] == 0 and info['converged']
    np.testing.assert_array_equal(x, 0)


def test_initial_callback_mutation_is_respected():
    seen = []
    calls = []
    def callback(x):
        if not seen:
            x[:] = 0.25
        seen.append(x.copy())
    def apply(x):
        calls.append(x.copy())
        return x.copy()
    x, info = linear_solver(apply, np.ones(3), solver='cg', callback=callback,
                          par={'rtol': 1e-12})
    np.testing.assert_array_equal(calls[0], .25)
    np.testing.assert_allclose(x, 1.)
    assert len(seen) == info['kit']+1


def test_maxiter_skips_unused_direction_and_keeps_callback():
    seen = []
    x, info = CG(lambda x: np.array([1., 3.])*x, np.ones(2),
                 par={'rtol': 1e-14, 'maxiter': 1, 'profile_cg_timing': True},
                 callback=lambda x: seen.append(x.copy()))
    assert info['hit_maxiter'] and not info['converged']
    assert info['cg_profile']['update_p']['calls'] == 0
    assert len(seen) == 1
    np.testing.assert_array_equal(x, [.5, .5])


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('warm', [False, True])
def test_tensor_cpu_gpu_and_independent_rhs(backend, batched, warm):
    xp = np if backend == 'numpy' else pytest.importorskip('cupy')
    if backend == 'cupy':
        try:
            if xp.cuda.runtime.getDeviceCount() == 0:
                pytest.skip('CUDA device unavailable')
        except xp.cuda.runtime.CUDARuntimeError:
            pytest.skip('CUDA unavailable')
    shape = (2, 3, 4, 3) if batched else (2, 4, 3)
    values = xp.arange(np.prod(shape), dtype=xp.float64).reshape(shape)/10
    if batched:
        values[:, 1] = 0  # Already-converged RHS alongside active ones.
    rhs = Tensor(val=values.copy(), order=2 if batched else 1, fft_form='r')
    diagonal = xp.linspace(1., 3., 12).reshape(4, 3)
    calls = []
    def apply(x):
        calls.append(1)
        return x.copy(val=x.val*diagonal)
    guess = rhs.copy(val=xp.ones_like(values)*.2) if warm else None
    x, info = linear_solver(apply, rhs, x0=guess, solver='cg', par={
        'rtol': 1e-10, 'atol': 1e-13, 'batched_rhs': batched,
        'cupy_lazy_scalars': True, 'cupy_fused_cg_updates': True,
        'cupy_fused_dot': True, 'cupy_fused_xr_rr': True,
        'cupy_residual_check_every': 3, 'keep_solution_on_device': True})
    assert info['converged']
    assert len(calls) == info['kit']+int(warm)
    xp.testing.assert_allclose(x.val, values/diagonal, atol=1e-11, rtol=1e-10)
    xp.testing.assert_array_equal(rhs.val, values)
