"""Independent coordinate, CG restart and physical-solution checks."""
import numpy as np
import pytest

from cufhom.pipeline import linear_krylov as krylov
from cufhom.pipeline.linear_krylov import SymmetricCoordinates, fused_cg
from cufhom.pipeline.nonlinear_solver import MixedProjector, cp, NonlinearConvergenceError
from cufhom_core.tensors import fft


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    old_backend, old_plan = fft.FFT_BACKEND, fft.CUPY_PLAN_MODE
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    yield
    fft.set_fft_backend(old_backend); fft.set_cupy_plan_mode(old_plan)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8), (1, 3, 5)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_fused_maps_cpu_adjoint_and_owned_outputs(shape, dtype):
    lengths = (1., 1.7, .9)
    Q = SymmetricCoordinates(MixedProjector(shape, lengths, [True]*6, cp), 4.3, 1.6)
    cpu = SymmetricCoordinates(MixedProjector(shape, lengths, [True]*6, np), 4.3, 1.6)
    rng = np.random.default_rng(2909)
    field = rng.normal(size=(6,)+shape).astype(dtype)
    ctype = np.complex64 if dtype == np.float32 else np.complex128
    p = (rng.normal(size=Q.size)+1j*rng.normal(size=Q.size)).astype(ctype)
    r = (rng.normal(size=Q.size)+1j*rng.normal(size=Q.size)).astype(ctype)
    pg, rg, fg = cp.asarray(p), cp.asarray(r), cp.asarray(field)
    tol = 8e-7 if dtype == np.float32 else 2e-14
    result, dot = Q.adjoint_and_dot(fg, pg)
    expected = cpu._map(np.fft.rfftn(field, axes=(1, 2, 3)), True)
    assert np.linalg.norm(cp.asnumpy(result)-expected)/np.linalg.norm(expected) < tol
    separate = Q.adjoint(fg)
    # Inlining may change FMA contraction; require roundoff-level agreement,
    # rather than bit identity across the two separately compiled kernels.
    np.testing.assert_allclose(cp.asnumpy(result), cp.asnumpy(separate), rtol=tol, atol=tol)
    expected_dot = float(np.dot(p.view(dtype).astype(np.float64),
                                cp.asnumpy(result).view(dtype).astype(np.float64)))
    assert dot == pytest.approx(expected_dot, rel=2e-13, abs=1e-13)
    # Zero/mean/Nyquist and partial CUDA blocks are included in random input.
    beta = dtype(.37)
    strain = Q.strain_updated_direction(pg, rg, beta)
    updated = r+beta*p
    h = cpu._map(updated, False)
    expected_strain = np.fft.irfftn(h, s=shape, axes=(1, 2, 3))*np.prod(shape)
    assert np.linalg.norm(cp.asnumpy(strain)-expected_strain)/np.linalg.norm(expected_strain) < tol
    np.testing.assert_allclose(cp.asnumpy(pg), updated, rtol=tol, atol=tol)
    np.testing.assert_array_equal(cp.asnumpy(rg), r)
    np.testing.assert_array_equal(cp.asnumpy(fg), field)
    # Subsequent calls cannot overwrite an earlier result.
    saved_result, saved_strain = cp.asnumpy(result), cp.asnumpy(strain)
    Q.adjoint_and_dot(fg*2, pg)
    Q.strain_updated_direction(pg, rg, beta)
    np.testing.assert_array_equal(cp.asnumpy(result), saved_result)
    np.testing.assert_array_equal(cp.asnumpy(strain), saved_strain)


@pytest.mark.parametrize('smoothing', [False, True])
def test_cg_reliable_restart_with_deferred_direction(monkeypatch, smoothing):
    diagonal = cp.linspace(1., 10., 129, dtype=cp.float32)
    rhs = cp.ones(129, cp.complex64)
    def dot(a, b):
        return float(cp.sum(a.view(cp.float32).astype(cp.float64)*
                            b.view(cp.float32).astype(cp.float64)))
    action = lambda x: diagonal*x
    updates = []
    def action_and_dot(p, *, direction=None):
        updates.append(direction is not None)
        if direction is not None:
            r, beta = direction
            p[:] = r+cp.float32(beta)*p
        ap = action(p)
        return ap, dot(p, ap)
    original_update = krylov.fused_update
    count = 0
    def false_recursive_once(*a):
        nonlocal count
        count += 1
        true_value = original_update(*a)
        return 0. if count == 1 else true_value
    monkeypatch.setattr(krylov, 'fused_update', false_recursive_once)
    x, iterations = fused_cg(action, rhs, dot, 1e-5, 100, cp,
        action_and_dot=action_and_dot, residual_smoothing=smoothing)
    assert iterations > 2
    assert updates[:2] == [False, False]  # first action and reliable restart
    assert any(updates[2:])
    assert np.sqrt(dot(rhs-action(x), rhs-action(x))/dot(rhs, rhs)) <= 1e-5
    np.testing.assert_allclose(cp.asnumpy(x), 1/cp.asnumpy(diagonal), rtol=5e-5)
    np.testing.assert_array_equal(cp.asnumpy(rhs), np.ones(129))


@pytest.mark.parametrize('false_estimate', [False, True])
def test_late_smoothing_dense_spd_and_replacement(monkeypatch, false_estimate):
    rng = np.random.default_rng(3009)
    q, _ = np.linalg.qr(rng.normal(size=(129, 129)))
    matrix = (q*np.geomspace(1., 80., 129))@q.T
    b = rng.normal(size=129).astype(np.float32)
    a, rhs = cp.asarray(matrix, dtype=cp.float32), cp.asarray(b)
    def dot(x, y):
        return float(cp.sum(x.astype(cp.float64)*y.astype(cp.float64)))
    original = krylov.fused_update
    calls, corrupted = [], False
    def observed(*args):
        nonlocal corrupted
        norm = original(*args)
        active = len(args) == 7
        calls.append(active)
        # Corrupt the estimate AFTER smoothing starts, so the audit must
        # reset the delayed blend as well as the CG search direction.
        if active and false_estimate and not corrupted:
            corrupted = True
            return 0.
        return norm
    monkeypatch.setattr(krylov, 'fused_update', observed)
    x, _ = fused_cg(lambda v: a@v, rhs, dot, 1e-5, 250, cp, residual_smoothing=True)
    result = cp.asnumpy(x).astype(np.float64)
    assert not calls[0] and any(calls)
    if false_estimate:
        assert corrupted and not calls[calls.index(True)+1]
    # Independent CPU64 equation and solution, rather than the recurrence.
    assert np.linalg.norm(b-matrix@result)/np.linalg.norm(b) < 1.1e-5
    ref = np.linalg.solve(matrix, b)
    assert np.linalg.norm(result-ref)/np.linalg.norm(ref) < 8e-5
    np.testing.assert_array_equal(cp.asnumpy(rhs), b)


def test_smoothing_zero_rhs_budget_and_preconditioning():
    dot = lambda a, b: float(cp.vdot(a, b).real)
    rhs = cp.zeros(16, cp.complex64)
    x, iterations = fused_cg(lambda v: v, rhs, dot, 1e-5, 10, cp, residual_smoothing=True)
    assert iterations == 0 and not bool(cp.any(x))
    rhs[:] = 1
    with pytest.raises(NonlinearConvergenceError, match='true residual'):
        fused_cg(lambda v: cp.arange(1, 17)*v, rhs, dot, 1e-5, 1, cp, residual_smoothing=True)
    with pytest.raises(ValueError, match='symmetrically'):
        fused_cg(lambda v: v, rhs, dot, 1e-5, 10, cp,
            precondition=lambda v: v, residual_smoothing=True)


def test_fused_cg_rejects_nonpositive_product():
    rhs = cp.ones(8, cp.complex64)
    with pytest.raises(NonlinearConvergenceError, match='positivity'):
        fused_cg(lambda x: x, rhs, lambda a, b: float(cp.vdot(a, b).real), 1e-5, 10, cp,
                 action_and_dot=lambda p, **kw: (p.copy(), -1.))
