"""Learned preconditioner invariants; no claim of neural solution accuracy."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from profile_g09_fft import cp, solve_homogenization
from learned_spectral_candidate import (NBINS, fit_spd, make_groups, LearnedCoordinates,
    OperatorSamples, collect_samples, learned_candidate)
from pipeline.linear_krylov import SymmetricCoordinates
from pipeline.nonlinear_solver import MixedProjector
from ffthompy.tensors import fft
from test_linear_coordinates import case


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    fft.set_fft_backend('cupy')


def test_symmetric_fit_recovers_exact_inverse_and_identity_without_data():
    rng = np.random.default_rng(421)
    R = rng.normal(size=(3, 1000)); B = np.array([[.8,.1,.05],[.1,.6,-.03],[.05,-.03,.4]])
    E = B@R
    blocks = fit_spd(np.array([R@R.T,np.zeros((3,3))]),
                     np.array([E@R.T,np.zeros((3,3))]), ridge=1e-10)
    np.testing.assert_allclose(blocks[0], B, atol=1e-9)
    np.testing.assert_array_equal(blocks[1], np.eye(3))
    with pytest.raises(ValueError, match='Nonfinite'):
        fit_spd(np.full((1,3,3), np.nan), np.zeros((1,3,3)))


@pytest.mark.parametrize('shape', [(5,7,5),(6,4,8)])
def test_groups_preserve_reality_and_excluded_modes(shape):
    g = cp.asnumpy(make_groups(shape, (1.,2.,3.))).reshape(shape[:-1]+(shape[-1]//2+1,))
    assert g[0,0,0] == -1
    for i in range(shape[0]):
        for j in range(shape[1]):
            assert g[i,j,0] == g[-i%shape[0],-j%shape[1],0]
    for axis, n in enumerate(shape):
        if n%2 == 0:
            assert np.all(np.take(g, n//2, axis=axis) == -1)


@pytest.mark.parametrize('shape', [(5,7,5),(6,4,8)])
def test_fused_learned_map_matches_explicit_composition(shape):
    P = MixedProjector(shape, (1.,2.,3.), np.ones(6,bool), cp)
    Q = SymmetricCoordinates(P, 2.7, 1.2)
    rng = np.random.default_rng(19)
    a = rng.normal(size=(NBINS,3,3)); B = a@a.transpose(0,2,1)+.4*np.eye(3)
    model = dict(shape=shape, lengths=P.lengths, blocks=B)
    F = LearnedCoordinates(P, 2.7, 1.2, model, {})
    e = cp.asarray(rng.normal(size=(6,)+shape), dtype=cp.float32)
    z = Q.adjoint(e)
    groups = cp.maximum(F.groups, 0)
    def root(x):
        return cp.einsum('nij,jn->in', F.roots[groups], x.reshape(3,-1)).ravel()
    np.testing.assert_allclose(cp.asnumpy(F.strain(z)), cp.asnumpy(Q.strain(root(z))), rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(cp.asnumpy(F.adjoint(e)), cp.asnumpy(root(Q.adjoint(e))), rtol=3e-5, atol=3e-6)
    a = float(cp.vdot(F.strain(z).astype(cp.float64),e.astype(cp.float64)))
    b = float(cp.vdot(z.astype(cp.complex128),F.adjoint(e).astype(cp.complex128)).real)
    assert abs(a-b) < 3e-6*max(1.,abs(a),abs(b))


@pytest.mark.parametrize('shape', [(5,7,5),(6,4,8)])
def test_trained_candidate_cpu_reference_and_restoration(shape):
    from pipeline import linear_coordinates as linear
    original_factory, original_cg, original_map = linear.make_linear_solver, linear.fused_cg, linear.SymmetricCoordinates
    p = case(shape)
    reference = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-11, linear_formulation='strain', cfield_indexed=False))
    samples = OperatorSamples(samples_per_bin=4, stride=2, last_action=12)
    with collect_samples(samples):
        solve_homogenization(dict(p, active_load_ids=[0,1,2], partial_load_output=True))
    model = samples.model()
    assert model['pairs'] > 0
    assert np.linalg.eigvalsh(model['blocks']).min() > 0
    with learned_candidate(model, .5) as (infos, _):
        actual = solve_homogenization(p)
    assert len(infos)==6 and all(i['true_norm_res_rel']<=1e-5 for i in infos)
    assert np.linalg.norm(actual-reference)/np.linalg.norm(reference) < 2e-6
    assert (linear.make_linear_solver, linear.fused_cg, linear.SymmetricCoordinates)==(original_factory, original_cg, original_map)
    with pytest.raises(RuntimeError):
        with learned_candidate(model):
            solve_homogenization(dict(p, solver_maxiter=1))
    assert linear.SymmetricCoordinates is original_map


def test_bad_spd_model_is_rejected():
    P = MixedProjector((4,4,4), (1.,1.,1.), np.ones(6,bool), cp)
    model = dict(shape=P.shape,lengths=P.lengths,blocks=np.tile(-np.eye(3),(NBINS,1,1)))
    with pytest.raises(ValueError, match='SPD'):
        LearnedCoordinates(P, 2.7, 1.2, model, {})
