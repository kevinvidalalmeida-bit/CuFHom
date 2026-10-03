"""Material, mixed-control, restart and CPU/GPU verification of nonlinear FFT."""
import numpy as np
import pytest

from cufhom.materials.plasticity import (J2Material, initial_state, update_cpu,
                                         tangent_cpu, commit_state, CuPyMaterial)
from cufhom.fft.nonlinear.solver import (MixedProjector, solve_nonlinear_homogenization,
                                       NonlinearConvergenceError, _cg)
from cufhom.fft.linear.solver import cp, solve_homogenization
from cufhom_core.tensors import fft as core_fft


def case(n=5, composite=False):
    phase = np.zeros((n, n, n), dtype=np.uint8)
    if composite:
        phase[n//2, :, :] = 1
    ori = np.zeros(phase.shape+(3,))
    ori[..., 0] = 1
    return dict(phase_array=phase, ori_array=ori, path=[0, .01, 0],
                initial_increment=.001, backend='scipy', return_fields=True)


def test_material_tangent_and_immutable_state():
    m = J2Material()
    state = initial_state((2, 2, 2))
    ids = np.zeros((2, 2, 2), dtype=np.int32)
    table = np.zeros((1, 6, 6))
    e = np.zeros((6, 2, 2, 2)); e[0] = .013; e[3] = .004
    response = update_cpu(e, state, m, ids, table)
    v = np.random.default_rng(42).normal(size=e.shape)
    h = 1e-7
    fd = (update_cpu(e+h*v, state, m, ids, table).stress-
          update_cpu(e-h*v, state, m, ids, table).stress)/(2*h)
    np.testing.assert_allclose(tangent_cpu(v, response, m, ids, table), fd, rtol=2e-8, atol=2e-8)
    assert not np.any(state.plastic_strain) and not np.any(state.equivalent)
    committed = commit_state(state, response)
    assert np.all(committed.equivalent > 0)
    assert not np.shares_memory(committed.plastic_strain, state.plastic_strain)
    np.testing.assert_allclose(committed.plastic_strain[:3].sum(axis=0), 0, atol=1e-15)


def test_default_path_is_monotonic_elastoplastic_loading():
    p = case(3)
    p.pop('path')
    result = solve_nonlinear_homogenization(p)
    assert result['path'] == [0., .01]
    assert result['cg_projection'] == 'galerkin'
    assert result['fused_cg'] is True
    assert result['inner_forcing'] == 'ew'
    assert len(result['history']) == 11
    axial = np.array([r['strain'][0] for r in result['history']])
    assert np.all(np.diff(axial) > 0)
    assert abs(axial[-1]-.01) < 1e-12
    assert not result['rejected_steps']
    assert result['history'][-1]['equivalent_plastic_mean'] > 0
    assert result['history'][-1]['stress'][0] > .015
    # With the same strain increments, subsequent unloading cannot change
    # the already confirmed loading branch.
    cycle = solve_nonlinear_homogenization(dict(p, path=[0., .01, 0.]))
    np.testing.assert_allclose([r['stress'] for r in result['history']],
                               [r['stress'] for r in cycle['history'][:11]], atol=1e-13)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_projector_orthogonal_and_mixed_means(shape):
    core_fft.set_fft_backend('scipy')
    P = MixedProjector(shape, (1., 2., 3.), [True, False, False, False, False, False])
    rng = np.random.default_rng(10)
    a, b = rng.normal(size=(2, 6)+shape)
    pa, pb = P(a), P(b)
    np.testing.assert_allclose(P(pa), pa, atol=2e-15)
    np.testing.assert_allclose(np.sum(a*pb), np.sum(pa*b), atol=1e-12)
    means = pa.mean(axis=(1, 2, 3))
    np.testing.assert_allclose(means[0], 0, atol=1e-15)
    np.testing.assert_allclose(means[1:], a.mean(axis=(1, 2, 3))[1:], atol=1e-15)


def test_homogeneous_analytic_curve_and_unloading():
    result = solve_nonlinear_homogenization(case())
    m = J2Material()
    for r in result['history'][:11]:
        e = r['strain'][0]
        expected = m.E*e if e <= m.yield_stress/m.E else (
            m.yield_stress + m.E*m.hardening/(m.E+m.hardening)*(e-m.yield_stress/m.E))
        assert abs(r['stress'][0]-expected) < 1e-10
        assert np.linalg.norm(r['stress'][1:]) < 1e-10
        assert r['residual'] <= 1e-8
    peak = result['history'][10]
    last = result['history'][-1]
    assert abs((peak['stress'][0]-last['stress'][0])/.01-m.E) < 1e-8
    assert last['equivalent_plastic_mean'] > 0
    assert last['stress'][0] < 0  # Unloading to zero strain is not zero stress.
    assert max(r['hill_mandel_error'] for r in result['history']) < 1e-10


def test_restart_equivalence_and_mismatch(tmp_path):
    p = case(3)
    ck = tmp_path/'state.npz'
    full = solve_nonlinear_homogenization(p)
    solve_nonlinear_homogenization(dict(p, path=[0, .01], checkpoint_path=ck))
    resumed = solve_nonlinear_homogenization(dict(p, path=[.01, 0], restart_from=ck))
    np.testing.assert_allclose(resumed['fields']['strain'], full['fields']['strain'], atol=1e-13)
    np.testing.assert_allclose(resumed['state'].plastic_strain, full['state'].plastic_strain, atol=1e-13)
    with pytest.raises(ValueError, match='mismatch'):
        solve_nonlinear_homogenization(dict(p, matrix=dict(E=4.), path=[.01, 0], restart_from=ck))


def test_failed_increment_does_not_commit(tmp_path):
    ck = tmp_path/'failed.npz'
    p = case(composite=True)
    with pytest.raises(NonlinearConvergenceError):
        solve_nonlinear_homogenization(dict(p, max_cg=1, min_increment=.001, checkpoint_path=ck))
    with np.load(ck, allow_pickle=False) as state:
        assert not np.any(state['equivalent'])
        assert not np.any(state['strain'])


def test_adaptive_cutback_and_checkpoint_export(tmp_path, monkeypatch):
    import cufhom.fft.nonlinear.solver as module
    original = module._cg
    failures = [True]
    def once(*args, **kwargs):
        if failures.pop() if failures else False:
            raise NonlinearConvergenceError('injected failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_cg', once)
    p = case(3)
    r = solve_nonlinear_homogenization(dict(p, path=[0, .001], output_dir=tmp_path, globalization='newton'))
    assert len(r['rejected_steps']) == 1
    assert abs(r['history'][-1]['stress'][0]-.003) < 1e-10
    assert (tmp_path/'curve.csv').exists() and (tmp_path/'curve.png').exists()


@pytest.mark.parametrize('n', [5, 6])
def test_linear_limit_matches_existing_solver(n):
    p = case(n, composite=True)
    linear = solve_homogenization(dict(input_dir='.', phase_array=p['phase_array'],
        ori_array=p['ori_array'], Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25,
        nu_TT=.3, G_LT=5., fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, solver_maxiter=1000, require_convergence=True, cfield_storage='sym21'))
    r = solve_nonlinear_homogenization(dict(p, path=[0, .0001], initial_increment=.0001))
    expected = .0001/np.linalg.inv(linear)[0, 0]
    assert abs(r['history'][-1]['stress'][0]-expected)/expected < 1e-6
    assert not np.any(r['state'].equivalent)


def test_small_composite_increment_convergence():
    p = case(5, composite=True)
    coarse = solve_nonlinear_homogenization(p)
    fine = solve_nonlinear_homogenization(dict(p, initial_increment=.0005))
    c = np.array([r['stress'] for r in coarse['history']])
    f = np.array([r['stress'] for r in fine['history']])[::2]
    assert np.max(np.abs(c-f))/.015 < .001
    assert max(r['hill_mandel_error'] for r in fine['history']) < 1e-7
    assert all(r['plastic_dissipation'] >= 0 for r in fine['history'])


def require_gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_gpu_curve_matches_cpu(dtype):
    require_gpu()
    p = case(5, composite=True)
    cpu = solve_nonlinear_homogenization(p)
    gpu = solve_nonlinear_homogenization(dict(p, backend='cupy', dtype=dtype))
    c = np.array([r['stress'] for r in cpu['history']])
    g = np.array([r['stress'] for r in gpu['history']])
    assert c.shape == g.shape
    assert np.max(np.abs(c-g))/.015 < .001
    assert max(r['residual'] for r in gpu['history']) <= gpu['rtol']
    np.testing.assert_allclose(gpu['state'].equivalent, cpu['state'].equivalent, atol=2e-6)


def test_cg_breakdown_and_exact_zero():
    b = np.ones((6, 2, 2, 2))
    dot = lambda a, b: float(np.sum(a*b))
    with pytest.raises(NonlinearConvergenceError, match='curvature'):
        _cg(lambda x: -x, b, dot, 1e-8, 10, np)
    x, iterations = _cg(lambda x: x, b*0, dot, 1e-8, 10, np)
    assert iterations == 0 and not np.any(x)


def test_variational_cg_matches_constrained_dense_solution():
    # C does not commute with P: the test must not accidentally be scalar C.
    rng = np.random.default_rng(90)
    Q, _ = np.linalg.qr(rng.normal(size=(19, 7)))
    P = Q@Q.T
    a = rng.normal(size=(19, 19))
    C = a.T@a + np.eye(19)*.1
    rhs = P@rng.normal(size=19)
    dot = lambda a, b: float(a@b)
    reference = Q@np.linalg.solve(Q.T@C@Q, Q.T@rhs)
    for action in (lambda v: C@v, lambda v: P@C@v):
        result, _ = _cg(action, rhs, dot, 1e-11, 100, np, project=lambda v: P@v)
        np.testing.assert_allclose(result, reference, atol=2e-12)
        assert np.linalg.norm(P@(rhs-C@result)) < 1e-11


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_fused_spectral_projection(dtype, shape, axis):
    require_gpu()
    core_fft.set_fft_backend('cupy')
    core_fft.set_cupy_plan_mode('manual', preserve_cache=True)
    control = np.zeros(6, dtype=bool); control[axis] = True
    lengths = (1., 2., 3.)
    reference = MixedProjector(shape, lengths, control, cp)
    candidate = MixedProjector(shape, lengths, control, cp, spectral_fusion=True)
    rng = np.random.default_rng(70)
    x = cp.asarray(rng.normal(size=(6,)+shape).astype(dtype))
    y = cp.asarray(rng.normal(size=(6,)+shape).astype(dtype))
    before = x.copy()
    px, py = candidate(x), candidate(y)
    tol = 2e-6 if dtype == 'float32' else 2e-14
    np.testing.assert_allclose(cp.asnumpy(px), cp.asnumpy(reference(x)), atol=tol, rtol=tol)
    np.testing.assert_allclose(cp.asnumpy(candidate(px)), cp.asnumpy(px), atol=tol, rtol=tol)
    assert float(cp.abs(cp.sum(x*py)-cp.sum(px*y))) < tol*x.size
    np.testing.assert_array_equal(cp.asnumpy(x), cp.asnumpy(before))
    constant = cp.ones_like(x)
    projected = candidate(constant)
    constant[axis] = 0
    np.testing.assert_allclose(cp.asnumpy(projected), cp.asnumpy(constant), atol=tol)


@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
def test_fused_spectral_curve_and_fields(globalization):
    require_gpu()
    p = dict(case(6, composite=True), backend='cupy', globalization=globalization)
    ref = solve_nonlinear_homogenization(dict(p, spectral_fusion=False))
    result = solve_nonlinear_homogenization(dict(p, spectral_fusion=True))
    assert result['spectral_fusion'] and not result['rejected_steps']
    a, b = [np.array([r['stress'] for r in run['history']]) for run in (ref, result)]
    assert np.max(np.abs(a-b))/.015 < .001
    np.testing.assert_allclose(result['fields']['strain'], ref['fields']['strain'], atol=2e-6)
    np.testing.assert_allclose(result['state'].plastic_strain, ref['state'].plastic_strain, atol=2e-6)
    assert max(r['residual'] for r in result['history']) <= result['rtol']


@pytest.mark.parametrize('backend', ['scipy', 'cupy'])
def test_ew_forcing_preserves_curve_restart_and_final_acceptance(backend, tmp_path):
    if backend == 'cupy':
        require_gpu()
    p = dict(case(6, composite=True), backend=backend, spectral_fusion=True)
    ref = solve_nonlinear_homogenization(dict(p, inner_forcing='legacy'))
    ew = solve_nonlinear_homogenization(dict(p, inner_forcing='ew'))
    a, b = [np.array([r['stress'] for r in run['history']]) for run in (ref, ew)]
    assert a.shape == b.shape and np.max(np.abs(a-b))/.015 < .001
    assert max(r['residual'] for r in ew['history']) <= ew['rtol']
    assert not ew['rejected_steps']
    np.testing.assert_allclose(ew['fields']['strain'], ref['fields']['strain'], atol=2e-6)
    np.testing.assert_allclose(ew['state'].equivalent, ref['state'].equivalent, atol=2e-6)
    firsts = [r['linear_solves'][0]['relative_tolerance'] for r in ew['history'][1:] if r['linear_solves']]
    assert firsts and all(t == .1 for t in firsts)
    ck = tmp_path/'ew.npz'
    solve_nonlinear_homogenization(dict(p, inner_forcing='ew', path=[0, .01], checkpoint_path=ck))
    restarted = solve_nonlinear_homogenization(dict(p, inner_forcing='ew', path=[.01, 0], restart_from=ck))
    np.testing.assert_allclose(restarted['fields']['strain'], ew['fields']['strain'], atol=2e-6)


def test_ew_does_not_change_ptc_forcing():
    p = dict(case(5, composite=True), globalization='ptc')
    a = solve_nonlinear_homogenization(dict(p, inner_forcing='legacy'))
    b = solve_nonlinear_homogenization(dict(p, inner_forcing='ew'))
    np.testing.assert_array_equal(a['fields']['strain'], b['fields']['strain'])


def test_ew_bounds():
    from cufhom.fft.nonlinear.solver import _ew_tolerance
    assert _ew_tolerance(1., None, None, 1e-5) == .1
    assert _ew_tolerance(1e-10, 1., .1, 1e-5) == 1e-5
    assert _ew_tolerance(2., 1., .1, 1e-5) == .3
    with pytest.raises(ValueError, match='inner_forcing'):
        solve_nonlinear_homogenization(dict(case(), inner_forcing='invalid'))


def test_fused_spectral_fallback_without_manual_plan():
    require_gpu()
    old = core_fft.get_fft_backend_status()
    try:
        core_fft.set_fft_backend('cupy')
        core_fft.set_cupy_plan_mode('auto', preserve_cache=True)
        shape = (5, 7, 5)
        P = MixedProjector(shape, (1., 2., 3.), [True, False, False, False, False, False],
                           cp, spectral_fusion=True)
        reference = MixedProjector(shape, (1., 2., 3.), [True, False, False, False, False, False], cp)
        x = cp.asarray(np.random.default_rng(80).normal(size=(6,)+shape), dtype=cp.float64)
        np.testing.assert_allclose(cp.asnumpy(P(x)), cp.asnumpy(reference(x)), atol=2e-14)
    finally:
        core_fft.set_fft_backend(old['requested'])
        core_fft.set_cupy_plan_mode(old['cupy_plan_mode'], preserve_cache=True)


@pytest.mark.parametrize('backend', ['scipy', 'cupy'])
@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
def test_galerkin_curve_preserves_double_projection(backend, globalization):
    if backend == 'cupy':
        require_gpu()
    p = dict(case(6, composite=True), backend=backend, globalization=globalization)
    reference = solve_nonlinear_homogenization(dict(p, cg_projection='double', fused_cg=False))
    candidate = solve_nonlinear_homogenization(dict(p, cg_projection='galerkin', fused_cg=True))
    assert not candidate['rejected_steps']
    assert candidate['timing_calls']['fft_projection'] < reference['timing_calls']['fft_projection']
    stress_error = np.max(np.abs(np.array([r['stress'] for r in reference['history']])-
                                np.array([r['stress'] for r in candidate['history']])))
    assert stress_error/.015 < .001
    assert max(r['residual'] for r in candidate['history']) <= candidate['rtol']
    np.testing.assert_allclose(candidate['fields']['strain'], reference['fields']['strain'], atol=2e-6)
    np.testing.assert_allclose(candidate['state'].equivalent, reference['state'].equivalent, atol=2e-6)


def test_pseudotime_is_monotone_and_does_not_change_physics(tmp_path):
    p = case(3)
    strain_control = solve_nonlinear_homogenization(p)
    p.pop('initial_increment')
    pseudo_control = solve_nonlinear_homogenization(dict(p, initial_pseudo_step=.1))
    np.testing.assert_allclose([r['stress'] for r in strain_control['history']],
                               [r['stress'] for r in pseudo_control['history']], atol=1e-13)
    times = [r['pseudo_time'] for r in pseudo_control['history']]
    assert np.all(np.diff(times) > 0) and abs(times[-1]-2.) < 1e-13
    assert pseudo_control['history'][-1]['load_factor'] == 0.
    ck = tmp_path/'pseudo.npz'
    solve_nonlinear_homogenization(dict(p, path=[0, .01], checkpoint_path=ck, initial_pseudo_step=.1))
    resumed = solve_nonlinear_homogenization(dict(p, path=[.01, 0], restart_from=ck, initial_pseudo_step=.1))
    np.testing.assert_allclose([r['pseudo_time'] for r in resumed['history']], times, atol=1e-13)


@pytest.mark.parametrize('backend', ['scipy', 'cupy'])
def test_ptc_matches_physical_newton_curve(backend):
    if backend == 'cupy':
        require_gpu()
    p = dict(case(5, composite=True), backend=backend)
    newton = solve_nonlinear_homogenization(dict(p, globalization='newton'))
    ptc = solve_nonlinear_homogenization(dict(p, globalization='ptc'))
    assert any(r['ptc_iterations'] > 0 for r in ptc['history'])
    assert not ptc['rejected_steps']
    error = np.max(np.abs(np.array([r['stress'] for r in newton['history']])-
                         np.array([r['stress'] for r in ptc['history']])))
    assert error/.015 < .001
    assert max(r['residual'] for r in ptc['history']) <= ptc['rtol']
    np.testing.assert_allclose(ptc['state'].equivalent, newton['state'].equivalent, atol=2e-6)


def test_ptc_auto_fallback_preserves_increment(monkeypatch):
    import cufhom.fft.nonlinear.solver as module
    original = module._cg
    calls = [0]
    def fail_once(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise NonlinearConvergenceError('Injected Newton breakdown')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, '_cg', fail_once)
    result = solve_nonlinear_homogenization(dict(case(3), path=[0, .001], globalization='auto'))
    assert not result['rejected_steps'] and len(result['history']) == 2
    assert result['history'][-1]['ptc_iterations'] > 0
    assert 'Injected' in result['history'][-1]['ptc_fallback_reason']
    assert abs(result['history'][-1]['stress'][0]-.003) < 1e-10


def test_ptc_small_pseudostep_is_not_false_convergence(tmp_path):
    checkpoint = tmp_path/'failed_ptc.npz'
    with pytest.raises(NonlinearConvergenceError):
        solve_nonlinear_homogenization(dict(case(3), path=[0, .01], globalization='ptc',
            ptc_initial_dt=1e-8, max_ptc=1, initial_increment=.01, min_increment=.01,
            checkpoint_path=checkpoint))
    with np.load(checkpoint, allow_pickle=False) as state:
        assert not np.any(state['equivalent']) and not np.any(state['strain'])


@pytest.mark.parametrize('backend', ['scipy', 'cupy'])
def test_explicit_warmup_then_implicit_ptc_preserves_curve(backend):
    if backend == 'cupy':
        require_gpu()
    p = dict(case(5, composite=True), backend=backend)
    reference = solve_nonlinear_homogenization(dict(p, globalization='newton'))
    hybrid = solve_nonlinear_homogenization(dict(p, globalization='ptc', ptc_explicit_steps=3))
    assert not hybrid['rejected_steps']
    assert sum(r.get('explicit_steps', 0) for r in hybrid['history']) > 0
    assert max(r['residual'] for r in hybrid['history']) <= hybrid['rtol']
    error = np.max(np.abs(np.array([r['stress'] for r in reference['history']])-
                         np.array([r['stress'] for r in hybrid['history']])))
    assert error/.015 < .001
    np.testing.assert_allclose(hybrid['state'].equivalent, reference['state'].equivalent, atol=2e-6)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_gpu_direct_projection_matches_reference(dtype, shape):
    require_gpu()
    from cufhom_core.tensors.objects import elasticity_direct_array
    control = [True, False, False, False, False, False]
    p = MixedProjector(shape, (1., 2., 3.), control, cp)
    a = np.random.default_rng(6).normal(size=(6,)+shape).astype(dtype)
    core_fft.set_fft_backend('cupy')
    h = core_fft.rfftn_cached(cp.asarray(a), shape, (1, 2, 3))
    direct = elasticity_direct_array(p.descriptor, h)
    assert direct is not None  # Do not silently exercise the generic fallback.
    cp.testing.assert_allclose(direct, p._formula(h), rtol=1e-5 if dtype == 'float32' else 1e-12,
                               atol=1e-5 if dtype == 'float32' else 1e-12)
    actual = cp.asnumpy(p(cp.asarray(a)))
    core_fft.set_fft_backend('scipy')
    expected = MixedProjector(shape, (1., 2., 3.), control)(a.astype('float64'))
    np.testing.assert_allclose(actual, expected, atol=1e-6 if dtype == 'float32' else 1e-14)


def test_gpu_material_update_and_tangent_match_cpu():
    require_gpu()
    m = J2Material()
    shape = (4, 5, 6)
    state = initial_state(shape)
    ids = np.zeros(shape, dtype=np.int32)
    ids[0] = 1
    table = np.stack([np.eye(6), np.diag([9., 5., 3., 1., 1., 2.])])
    rng = np.random.default_rng(15)
    e = rng.normal(size=(6,)+shape)*.01
    v = rng.normal(size=e.shape)
    cpu = update_cpu(e, state, m, ids, table)
    for dtype, tol in [(np.float32, 2e-6), (np.float64, 1e-13)]:
        gpu = CuPyMaterial(cp, m, cp.asarray(ids), cp.asarray(table, dtype=dtype), dtype)
        response = gpu.update(cp.asarray(e, dtype=dtype), initial_state(shape, cp, dtype))
        for key in ('stress', 'direction', 'a', 'b', 'dp'):
            np.testing.assert_allclose(cp.asnumpy(getattr(response, key)), getattr(cpu, key), atol=tol, rtol=tol)
        np.testing.assert_allclose(cp.asnumpy(gpu.tangent(cp.asarray(v, dtype=dtype), response)),
                                   tangent_cpu(v, cpu, m, ids, table), atol=tol*5, rtol=tol)


@pytest.mark.parametrize('parameters', [dict(matrix=dict(hardening=0)), dict(axis=6),
    dict(path=[.001, .01]), dict(initial_increment=-1), dict(fiber_fraction_array=1),
    dict(dtype='float16')])
def test_invalid_inputs(parameters):
    with pytest.raises(ValueError):
        solve_nonlinear_homogenization(dict(case(), **parameters))
