"""Fused PCG must preserve free means, the PTC shift, and true residuals."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT'))
from pipeline.nonlinear_solver import MixedProjector, _cg, NonlinearConvergenceError, cp
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.nonlinear_reference import IsotropicReference
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from ffthompy.tensors import fft


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    old = fft.get_fft_backend_status()
    fft.set_fft_backend('cupy')
    yield
    fft.set_fft_backend(old['requested'])


def make_problem(shape, control, shift):
    projector = MixedProjector(shape, (1., 2., 3.), control, cp)
    coordinates = DisplacementCoordinates(projector)
    reference = IsotropicReference(projector, 4.7, 1.2)
    reference.set_shift(shift, np.dtype('float32'))
    return coordinates, reference, FusedCoordinatePCG(coordinates)


def dot(a, b):
    # Independent test reference accumulation, including the complex parts.
    return float(cp.sum(a.real.astype(cp.float64)*b.real + a.imag.astype(cp.float64)*b.imag))


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('control', [[True]*6, [True,False,False,False,False,False], [False]*6])
@pytest.mark.parametrize('shift', [0., 3.])
def test_fused_updates_against_independent_operations(shape, control, shift):
    Q, reference, engine = make_problem(shape, control, shift)
    rng = np.random.default_rng(457)
    x, r, p, ap = [Q.adjoint(cp.asarray(rng.normal(size=(6,)+shape), dtype=cp.float32)) for _ in range(4)]
    alpha, beta = np.float32(.137), np.float32(.19)
    expected_x, expected_r = x+alpha*p, r-alpha*ap
    expected_z = Q.precondition(expected_r, reference)
    expected_rr, expected_rz = dot(expected_r, expected_r), dot(expected_r, expected_z)
    rr, rz = engine.update(p, ap, alpha, x, r, reference)
    cp.testing.assert_allclose(x, expected_x, atol=5e-7, rtol=2e-6)
    cp.testing.assert_allclose(r, expected_r, atol=5e-7, rtol=2e-6)
    np.testing.assert_allclose([rr,rz], [expected_rr,expected_rz], rtol=3e-7, atol=1e-10)
    expected_p = Q.precondition(r, reference)+beta*p
    engine.direction(r, beta, p, reference)
    cp.testing.assert_allclose(p, expected_p, atol=5e-7, rtol=2e-6)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('shift', [0., 3.])
def test_cg_matches_original_constrained_problem(shape, shift):
    Q, reference, engine = make_problem(shape, [True,False,False,False,False,False], shift)
    rng = np.random.default_rng(804)
    rhs = Q.adjoint(cp.asarray(rng.normal(size=(6,)+shape), dtype=cp.float32))
    stiffness = cp.asarray(1+7*rng.random(shape), dtype=cp.float32)
    def action(z):
        return Q.adjoint(stiffness*Q.strain(z))+np.float32(shift)*z
    baseline, old_count = _cg(action, rhs, dot, 1e-5, 300, cp, fused=True,
                             precondition=lambda v: Q.precondition(v, reference))
    candidate, count = engine.solve(action, rhs, dot, 1e-5, 300, reference)
    r = rhs-action(candidate)
    assert np.sqrt(dot(r,r)/dot(rhs,rhs)) <= 1e-5
    cp.testing.assert_allclose(candidate, baseline, atol=5e-6, rtol=1e-4)
    assert abs(count-old_count) <= 2


def test_zero_rhs_positivity_and_nonfinite():
    Q, reference, engine = make_problem((5,5,5), [True]*6, 0.)
    rhs = Q.adjoint(cp.ones((6,5,5,5), dtype=cp.float32))
    def fail(_):
        raise AssertionError('Zero RHS must not apply A')
    x, count = engine.solve(fail, rhs*0, dot, 1e-5, 30, reference)
    assert count == 0 and not cp.any(x)
    rhs = Q.adjoint(cp.asarray(np.random.default_rng(99).normal(size=(6,5,5,5)), dtype=cp.float32))
    with pytest.raises(NonlinearConvergenceError, match='positivity'):
        engine.solve(lambda x:-x, rhs, dot, 1e-5, 30, reference)
    with pytest.raises(NonlinearConvergenceError, match='Nonfinite'):
        engine.solve(lambda x:x, rhs*cp.nan, dot, 1e-5, 30, reference)


def test_false_recursive_convergence_is_not_accepted(monkeypatch):
    Q, reference, engine = make_problem((5,5,5), [True,False,False,False,False,False], 0.)
    rng = np.random.default_rng(923)
    rhs = Q.adjoint(cp.asarray(rng.normal(size=(6,5,5,5)), dtype=cp.float32))
    coefficient = cp.asarray(1+rng.random((5,5,5)), dtype=cp.float32)
    original = engine.update
    def wrong_norm(*args):
        _, rz = original(*args)
        return 0., rz
    monkeypatch.setattr(engine, 'update', wrong_norm)
    with pytest.raises(NonlinearConvergenceError, match='true residual'):
        engine.solve(lambda z:Q.adjoint(coefficient*Q.strain(z)), rhs, dot, 1e-8, 1, reference)


def test_noncontiguous_rhs_is_rejected():
    Q, reference, engine = make_problem((5,5,5), [True]*6, 0.)
    rhs = cp.zeros(2*Q.size, dtype=cp.complex64)[::2]
    with pytest.raises(ValueError, match='contiguous'):
        engine.solve(lambda x:x, rhs, dot, 1e-5, 10, reference)
