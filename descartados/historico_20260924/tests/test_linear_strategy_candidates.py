"""Small-grid verification before timing experimental linear strategies."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from profile_g09_fft import cp, solve_homogenization
from linear_strategy_candidates import (candidate, SymmetricCoordinates,
    AnisotropicCoordinates, fused_update)
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.nonlinear_reference import IsotropicReference
from test_linear_coordinates import case


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('No GPU')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('mode', ['symmetric', 'fused_updates', 'combined', 'anisotropic', 'polarization', 'polarization_macro'])
@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_candidate_cpu_reference(mode, shape):
    p = case(shape)
    reference = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, cfield_indexed=False, linear_formulation='strain'))
    with candidate(mode) as (infos, cache):
        result = solve_homogenization(p)
    assert len(infos) == 6
    assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)
    assert np.linalg.norm(result-reference)/np.linalg.norm(reference) < 2e-6


@pytest.mark.parametrize('mode', ['symmetric', 'fused_updates', 'combined', 'anisotropic', 'polarization', 'polarization_macro'])
def test_failure_and_patch_restoration(mode):
    from pipeline import linear_coordinates
    original = linear_coordinates.make_linear_solver
    with pytest.raises(RuntimeError):
        with candidate(mode):
            solve_homogenization(dict(case((6, 4, 8)), solver_maxiter=1))
    assert linear_coordinates.make_linear_solver is original


@pytest.mark.parametrize('mode', ['symmetric', 'fused_updates', 'combined', 'anisotropic', 'polarization', 'polarization_macro'])
def test_homogeneous_zero_rhs(mode):
    p = case((4, 4, 4)); p['phase_array'][:] = 0
    reference = solve_homogenization(dict(p, linear_formulation='strain'))
    with candidate(mode) as (infos, _):
        result = solve_homogenization(p)
    np.testing.assert_allclose(result, reference, atol=1e-6)
    assert all(i['coordinate_iterations'] == 0 and i['correction_iterations'] == 0 for i in infos)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_symmetric_coordinates_and_adjoint(shape):
    from ffthompy.tensors import fft
    fft.set_fft_backend('cupy')
    P = MixedProjector(shape, (1., 2., 3.), np.ones(6, bool), cp)
    Q = DisplacementCoordinates(P)
    F = SymmetricCoordinates(P, 2.5, 1.2)
    rng = np.random.default_rng(14)
    e = cp.asarray(rng.normal(size=(6,)+shape), dtype=cp.float32)
    v = Q.adjoint(e)
    ref = IsotropicReference(P, 2.5, 1.2); ref.set_shift(0., np.dtype('float32'))
    # R^2 equals the existing matrix inverse (on valid, zero-mean coordinates).
    cpuP = MixedProjector(shape, P.lengths, np.ones(6, bool), np)
    cpuF = SymmetricCoordinates(cpuP, 2.5, 1.2)
    rv = cp.asarray(cpuF.root(cp.asnumpy(v)), dtype=cp.complex64)
    np.testing.assert_allclose(cp.asnumpy(F.strain(v)), cp.asnumpy(Q.strain(rv)), rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(Q.precondition(v, ref)),
        cpuF.root(cpuF.root(cp.asnumpy(v))), rtol=5e-6, atol=2e-6)
    lhs = float(cp.vdot(F.strain(v).astype(cp.float64), e.astype(cp.float64)).real)
    rhs = float(cp.vdot(v.astype(cp.complex128), F.adjoint(e).astype(cp.complex128)).real)
    assert abs(lhs-rhs) < 2e-6*max(1., abs(lhs))
    assert lhs > 0


def test_fused_update_and_fp64_norm():
    rng = np.random.default_rng(31)
    arrays = [cp.asarray((rng.normal(size=5000)+1j*rng.normal(size=5000)).astype('complex64')) for _ in range(4)]
    p, ap, x, r = arrays
    expected_x = cp.asnumpy(x)+np.float32(.125)*cp.asnumpy(p)
    expected_r = cp.asnumpy(r)-np.float32(.125)*cp.asnumpy(ap)
    partial = cp.empty(40, cp.float64)
    norm = fused_update(p, ap, .125, x, r, partial)
    np.testing.assert_allclose(cp.asnumpy(x), expected_x, rtol=2e-6, atol=3e-7)
    np.testing.assert_allclose(cp.asnumpy(r), expected_r, rtol=2e-6, atol=3e-7)
    expected_norm = np.sum(cp.asnumpy(r).view(np.float32).astype(np.float64)**2)
    assert abs(norm-expected_norm)/expected_norm < 1e-12


def test_anisotropic_inverse_on_compatible_space():
    from ffthompy.tensors import fft
    fft.set_fft_backend('cupy')
    P = MixedProjector((6, 4, 8), (1., 2., 3.), np.ones(6, bool), cp)
    rng = np.random.default_rng(13)
    A = rng.normal(size=(6, 6)); C = A.T@A+3*np.eye(6)
    Q = AnisotropicCoordinates(P, C, {})
    v = Q.adjoint(cp.asarray(rng.normal(size=(6,)+P.shape), dtype=cp.float32))
    Hv = Q.adjoint(cp.einsum('ij,jxyz->ixyz', cp.asarray(C, dtype=cp.float32), Q.strain(v)))
    restored = Q.precondition(Hv, None)
    assert float(cp.linalg.norm(restored-v)/cp.linalg.norm(v)) < 1e-6
    assert float(cp.vdot(v, Q.precondition(v, None)).real) > 0


@pytest.mark.parametrize('ratio', [.3, .6, .9, 1.2])
def test_reference_ratio_preserves_physical_solution(ratio):
    p = case((6, 5, 8))
    reference = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, cfield_indexed=False, linear_formulation='strain'))
    with candidate('combined', reference_ratio=ratio) as (infos, _):
        result = solve_homogenization(p)
    assert np.linalg.norm(result-reference)/np.linalg.norm(reference) < 2e-6
    assert all(i['true_norm_res_rel'] <= 1e-5 for i in infos)


@pytest.mark.parametrize('ratio', [0., 1.5, float('nan')])
def test_reference_ratio_requires_spd(ratio):
    with pytest.raises(ValueError, match='Reference ratio'):
        with candidate('combined', reference_ratio=ratio):
            pass
