"""GPU polarization: algebra, local material equivalence and physical audits."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from profile_g09_fft import cp, MATERIAL, solve_homogenization
from linear_polarization_candidate import FiberPolarization
from linear_strategy_candidates import candidate, fused_update
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_reference import IsotropicReference
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


def engine_case(shape, negative=False):
    rng = np.random.default_rng(126)
    K, mu = 3., 1.2
    Cm = 2*mu*np.eye(6); Cm[:3, :3] += K-2*mu/3
    F = rng.normal(size=(6, 6)); delta = F.T@F+np.eye(6)
    C = np.array([Cm, Cm+(-delta if negative else delta), Cm+2*delta])
    i, j = np.triu_indices(6)
    ids = rng.integers(0, 3, size=shape, dtype=np.int32)
    A = SimpleNamespace(N=shape, val=cp.asarray(ids), material_table=cp.asarray(C[:, i, j], dtype=cp.float32))
    return A, K, mu


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('local', [False, True])
def test_action_cpu_equivalence_symmetry_and_reuse(shape, local):
    A, K, mu = engine_case(shape)
    lengths = (1., 2., 3.)
    engine = FiberPolarization(A, lengths, K, mu, local=local)
    rng = np.random.default_rng(12)
    u, v = [cp.asarray(rng.normal(size=6*engine.nf), dtype=cp.float32) for _ in range(2)]
    au = engine.action(u)
    av = engine.action(v)
    lhs = float(cp.sum(u.astype(cp.float64)*av))
    rhs = float(cp.sum(v.astype(cp.float64)*au))
    # Normalize adjoint error by operand norms, not an inner product that may
    # nearly cancel. The CPU action check below is independent of this test.
    scale = float(cp.linalg.norm(u.astype(cp.float64))*cp.linalg.norm(av.astype(cp.float64))
                  +cp.linalg.norm(v.astype(cp.float64))*cp.linalg.norm(au.astype(cp.float64)))
    assert abs(lhs-rhs) < 3e-7*scale
    assert float(cp.sum(u.astype(cp.float64)*au)) > 0
    np.testing.assert_array_equal(cp.asnumpy(au), cp.asnumpy(engine.action(u)))

    ids = cp.asnumpy(engine.ids).ravel(); selected = cp.asnumpy(engine.selected)
    D = cp.asnumpy(engine.roots).astype(np.float64)[ids[selected]]
    vec = cp.asnumpy(u).astype(np.float64).reshape(6, -1)
    full = np.zeros((6, engine.n))
    full[:, selected] = np.einsum('nij,jn->in', D, vec)
    P = MixedProjector(shape, lengths, np.ones(6, bool), np)
    Gamma = IsotropicReference(P, K, mu); Gamma.set_shift(0., np.dtype('float64'))
    fft.set_fft_backend('scipy')
    g = Gamma(full.reshape((6,)+shape)).reshape(6, -1)
    base = cp.asnumpy(engine.base).astype(np.float64)[ids[selected]]
    expected = (np.einsum('nij,jn->in', base, vec)
                +np.einsum('nji,jn->in', D, g[:, selected])).ravel()
    np.testing.assert_allclose(cp.asnumpy(au), expected, rtol=2e-5, atol=3e-6)


def test_negative_contrast_is_rejected():
    A, K, mu = engine_case((4, 4, 4), negative=True)
    with pytest.raises(ValueError, match='positive material contrast'):
        FiberPolarization(A, (1., 1., 1.), K, mu)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('mode', ['polarization', 'polarization_macro', 'polarization_local'])
def test_random_g09_material_cpu64_reference(shape, mode):
    rng = np.random.default_rng(31)
    p = case(shape); m = json.loads(MATERIAL.read_text())
    p.update({k:m[k] for k in ('Em', 'nu_m', 'Ef_L', 'Ef_T', 'nu_LT', 'nu_TT', 'G_LT')})
    p['phase_array'] = (rng.random(shape) < .25).astype(np.uint8)
    ori = rng.normal(size=shape+(3,)); ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    p['ori_array'] = ori
    ref = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-11, linear_formulation='strain', cfield_indexed=False))
    with candidate(mode) as (infos, _):
        actual = solve_homogenization(p)
    assert np.linalg.norm(actual-ref)/np.linalg.norm(ref) < 2e-6
    assert all(i['true_norm_res_rel'] <= 1e-5 for i in infos)


def test_false_inner_convergence_requires_physical_correction(monkeypatch):
    import linear_strategy_candidates as strategies
    from ffthompy import applications
    original = applications.solve_load_elasticity
    p = case((5, 7, 5))
    ref = solve_homogenization(p)
    monkeypatch.setattr(strategies, 'fused_cg', lambda action, rhs, *a, **kw: (cp.zeros_like(rhs), 1))
    with candidate('polarization_macro') as (infos, _):
        actual = solve_homogenization(p)
    assert applications.solve_load_elasticity is original
    assert all(i['correction_iterations'] > 0 for i in infos)
    np.testing.assert_allclose(actual, ref, atol=3e-6, rtol=2e-6)


def test_real_fused_update_has_correct_extent_and_fp64_sum():
    rng = np.random.default_rng(75)
    p, ap, x, r = [cp.asarray(rng.normal(size=1003), dtype=cp.float32) for _ in range(4)]
    rx = cp.asnumpy(x)+np.float32(.125)*cp.asnumpy(p)
    rr = cp.asnumpy(r)-np.float32(.125)*cp.asnumpy(ap)
    norm = fused_update(p, ap, .125, x, r, cp.empty(17, cp.float64))
    np.testing.assert_allclose(cp.asnumpy(x), rx, atol=3e-7, rtol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(r), rr, atol=3e-7, rtol=1e-6)
    assert abs(norm-np.sum(cp.asnumpy(r).astype(np.float64)**2)) < 1e-10
