"""Algebra, RFFT admissibility and original-equation audits for isolated recycling."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from profile_g09_fft import cp, solve_homogenization
from linear_recycling_candidate import (BasisOps, Builder, CoarseSpace, candidate,
                                        energy_error_certificate)
from pipeline.nonlinear_solver import _cg, _table, MixedProjector
from pipeline.nonlinear_material import J2Material, elastic_action
from pipeline.linear_krylov import SymmetricCoordinates
from test_linear_coordinates import case


@pytest.fixture
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('method', ['spectral', 'balanced'])
def test_preconditioner_spd_and_exact_coarse_action(method):
    rng = np.random.default_rng(145)
    a = rng.normal(size=(20, 20)); a = a.T@a+np.eye(20)
    u = rng.normal(size=(4, 20)); coarse = CoarseSpace(u, u@a, BasisOps(np))
    b = np.column_stack([getattr(coarse, method)(v) for v in np.eye(20)])
    np.testing.assert_allclose(b, b.T, atol=2e-14)
    assert np.linalg.eigvalsh(b).min() > 0
    if method == 'balanced':
        np.testing.assert_allclose(b@a@u.T, u.T, atol=5e-14)
    rhs = rng.normal(size=20); x, _ = _cg(lambda v:a@v, rhs, np.dot, 1e-11, 200, np,
                                        precondition=getattr(coarse, method))
    assert np.linalg.norm(rhs-a@x)/np.linalg.norm(rhs) < 1e-11


@pytest.mark.parametrize('which', ['low', 'high'])
@pytest.mark.parametrize('rolling', [False, True])
def test_bounded_builder_and_true_au(which, rolling):
    rng = np.random.default_rng(32); a = rng.normal(size=(24, 24)); a = a.T@a+np.eye(24)
    b = rng.normal(size=24); builder = Builder(8, 3, which, rolling=rolling)
    def action(v):
        av = a@v; builder.capture(v, av, np.dot); return av
    x, k = _cg(action, b, np.dot, 1e-10, 200, np)
    assert builder.count <= 8 and builder.rows.shape == (8, 24)
    coarse = builder.finish(lambda v:a@v, k)
    assert builder.rows is None and len(coarse.rows) == 3
    np.testing.assert_allclose(coarse.arows, coarse.rows@a, atol=2e-13)
    np.testing.assert_allclose(coarse.G, np.eye(3), atol=2e-13)
    assert np.linalg.norm(b-a@x)/np.linalg.norm(b) < 1e-10
    assert builder.restarts > 0 if rolling else builder.restarts == 0


def test_rank_deficiency_and_zero_history():
    builder = Builder(4, 4)
    v = np.array([1., 2., 3.])
    for _ in range(4):
        builder.capture(v, 2*v, np.dot)
    coarse = builder.finish(lambda x:2*x, 4)
    assert len(coarse.rows) == 1
    assert Builder(3, 1).finish(lambda x:x, 0) is None


@pytest.mark.parametrize('shape', [(5, 5, 5), (6, 4, 8)])
def test_energy_identity_bounds_and_galerkin_correction(shape):
    from ffthompy.tensors import fft
    fft.set_fft_backend('scipy')
    rng = np.random.default_rng(248)
    phase = (rng.random(shape)<.25).astype(np.uint8)
    ori = rng.normal(size=shape+(3,)); ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    phase.flags.writeable = ori.flags.writeable = False
    law = J2Material(E=3., nu=.3)
    ids, table = _table(phase, ori, law, dict(E_L=70., E_T=8., G_LT=5., nu_LT=.25, nu_TT=.3))
    table = table.astype(np.float64)
    P = MixedProjector(shape, (1., 2., 3.), np.ones(6, bool), np)
    Q = SymmetricCoordinates(P, law.K, law.mu)
    def C(v): return elastic_action(v, ids, table)
    def A(v): return Q.adjoint(C(Q.strain(v)))
    def dot(x, y): return float(np.vdot(x, y).real)
    w, v = np.linalg.eigh(table[0]); invsqrt = (v*(1/np.sqrt(w)))@v.T
    eig = np.linalg.eigvalsh(np.einsum('ij,njk,kl->nil', invsqrt, table, invsqrt))
    alpha, beta = float(eig.min())*(1-1e-12), float(eig.max())*(1+1e-12)
    fields, residuals, deltas, zs, Es, bs = [], [], [], [], [], []
    builder = Builder(8, 3)
    for j in range(6):
        E = np.zeros((6,)+shape); E[j] = 1
        b = -Q.adjoint(C(E))
        def capture(v):
            av = A(v); builder.capture(v, av, dot); return av
        z, k = _cg(capture if j == 0 else A, b, dot, 1e-2, 1000, np)
        if j == 0:
            coarse = builder.finish(A, k)
        zstar, _ = _cg(A, b, dot, 1e-12, 1000, np)
        fields.append((E+Q.strain(z), E+Q.strain(zstar)))
        residuals.append(b-A(z)); deltas.append(z-zstar); zs.append(z); Es.append(E); bs.append(b)
    N = int(np.prod(shape))
    def tensor(es):
        stresses = [C(e) for e in es]
        return np.array([[np.sum(e*s)/N for s in stresses] for e in es])
    approximate = tensor([f[0] for f in fields]); exact = tensor([f[1] for f in fields])
    error = approximate-exact
    identity = np.array([[dot(x, A(y))/N for y in deltas] for x in deltas])
    np.testing.assert_allclose(error, identity, atol=3e-11)
    upper, trace = energy_error_certificate(np.column_stack(residuals), alpha, N)
    lower = upper*(alpha/beta)
    assert np.linalg.eigvalsh(upper-error).min() > -3e-11
    assert np.linalg.eigvalsh(error-lower).min() > -3e-11
    assert np.linalg.norm(error, 2) <= trace+3e-11
    d = np.column_stack([coarse.ops.project(coarse.rows, r) for r in residuals])
    corrected_prediction = approximate-d.T@coarse.Hinv@d/N
    corrected_fields = [E+Q.strain(z+coarse.seed(r)) for E, z, r in zip(Es, zs, residuals)]
    np.testing.assert_allclose(tensor(corrected_fields), corrected_prediction, atol=3e-11)
    assert np.linalg.eigvalsh(corrected_prediction-exact).min() > -3e-11
    for E, e in zip(Es, corrected_fields):
        np.testing.assert_allclose(P(e-E), e-E, atol=2e-12)
        np.testing.assert_allclose(np.mean(e-E, axis=(1, 2, 3)), 0., atol=1e-13)


@pytest.mark.parametrize('alpha', [0., -1., float('nan')])
def test_bad_certificate_bound_rejected(alpha):
    with pytest.raises(ValueError):
        energy_error_certificate(np.eye(6), alpha, 1)


@pytest.mark.parametrize('volume', [0., -1., float('nan'), float('inf')])
def test_bad_certificate_volume_rejected(volume):
    with pytest.raises(ValueError):
        energy_error_certificate(np.eye(6), 1., volume)


def test_energy_benchmark_is_serializable():
    import json
    from benchmark_linear_energy_certificate import experiment
    material = dict(Em=3., nu_m=.3, Ef_L=70., Ef_T=8., G_LT=5., nu_LT=.25, nu_TT=.3)
    report = experiment((5, 5, 5), material, 1e-4)
    json.dumps(report, allow_nan=False)
    assert report['modes']['physical_1e-5']['physical_1e_5_passed']
    assert report['modes']['tensor_only']['certified_target_passed']


def test_gpu_basis_products_and_no_input_mutation(gpu):
    rng = np.random.default_rng(121)
    rows = (rng.normal(size=(4, 5101))+1j*rng.normal(size=(4, 5101))).astype('complex64')
    x = rows[0].copy(); c = rng.normal(size=4)
    ops = BasisOps(cp); U = cp.asarray(rows); v = cp.asarray(x)
    reference = (rows.astype('complex128').conj()@x.astype('complex128')).real
    np.testing.assert_allclose(ops.project(U, v), reference, rtol=2e-13, atol=2e-12)
    actual = ops.combine(U, c, v)
    np.testing.assert_allclose(cp.asnumpy(actual), x+c@rows.astype('complex128'), rtol=2e-6, atol=4e-7)
    np.testing.assert_array_equal(cp.asnumpy(U), rows)
    np.testing.assert_array_equal(cp.asnumpy(v), x)


@pytest.mark.parametrize('mode', ['spectral', 'balanced', 'seed'])
@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('rolling', [False, True])
def test_six_loads_original_audit_against_cpu64(gpu, mode, shape, rolling):
    p = case(shape); phase = p['phase_array'].copy(); ori = p['ori_array'].copy()
    ref = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, cfield_indexed=False, linear_formulation='strain'))
    with candidate(mode, window=8, rank=3, rolling=rolling) as (infos, state):
        actual = solve_homogenization(p)
    assert len(infos) == 6 and len(state['loads']) == 6
    assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)
    assert np.linalg.norm(actual-ref)/np.linalg.norm(ref) < 2e-6
    assert state['basis']['rank'] <= 3
    np.testing.assert_array_equal(phase, p['phase_array'])
    np.testing.assert_array_equal(ori, p['ori_array'])


@pytest.mark.parametrize('mode', ['baseline', 'spectral', 'balanced', 'seed'])
def test_failure_restores_patches(gpu, mode):
    from pipeline import linear_coordinates as linear
    original = linear.make_linear_solver, linear.fused_cg
    with pytest.raises(RuntimeError):
        with candidate(mode, window=4, rank=2):
            solve_homogenization(dict(case((6, 4, 8)), solver_maxiter=1))
    assert (linear.make_linear_solver, linear.fused_cg) == original


@pytest.mark.parametrize('mode', ['baseline', 'spectral', 'balanced', 'seed'])
def test_zero_rhs_and_basis_not_reused_between_solves(gpu, mode):
    p = case((4, 4, 4)); p['phase_array'][:] = 0
    with candidate(mode, window=4, rank=2) as (infos, state):
        actual = solve_homogenization(p)
    assert all(i['coordinate_iterations'] == 0 for i in infos)
    if mode != 'baseline':
        assert state['basis']['rank'] == 0
    with candidate(mode, window=4, rank=2) as (infos, _):
        solve_homogenization(case((5, 7, 5)))
    assert len(infos) == 6
    assert np.isfinite(actual).all()


def test_same_context_rejects_new_operator(gpu):
    with candidate('spectral', window=4, rank=2):
        solve_homogenization(case((4, 4, 4)))
        with pytest.raises(RuntimeError, match='Cannot reuse a basis across different operators'):
            solve_homogenization(case((5, 7, 5)))
