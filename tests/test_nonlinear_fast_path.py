"""The retained nonlinear FFT fast path preserves the constrained problem."""
from pathlib import Path
import sys
import numpy as np
import pytest

from cufhom.pipeline.nonlinear_solver import MixedProjector, cp, solve_nonlinear_homogenization
from cufhom.pipeline.nonlinear_reference import IsotropicReference
from cufhom.pipeline.nonlinear_coordinates import DisplacementCoordinates
from cufhom_core.tensors import fft as core_fft


def _case():
    phase = np.zeros((6, 6, 6), dtype=np.uint8)
    phase[2:4, 2:4, :] = 1
    orientation = np.zeros(phase.shape + (3,))
    orientation[..., 2] = 1
    return dict(phase_array=phase, ori_array=orientation, path=[0., .006, .004],
                initial_increment=.001, return_fields=True)


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_displacement_coordinates_are_an_isometry(shape):
    core_fft.set_fft_backend('scipy')
    control = np.array([True, False, False, False, False, False])
    projector = MixedProjector(shape, (1., 2., 3.), control)
    coordinates = DisplacementCoordinates(projector)
    a, b = np.random.default_rng(4).normal(size=(2, 6) + shape)
    z, w = coordinates.adjoint(a), coordinates.adjoint(b)
    np.testing.assert_allclose(coordinates.strain(z), projector(a), atol=2e-12)
    np.testing.assert_allclose(coordinates.adjoint(coordinates.strain(z)), z, atol=2e-12)
    np.testing.assert_allclose(np.vdot(z, w).real, np.sum(projector(a) * projector(b)), atol=2e-12)


@pytest.mark.parametrize('shift', [0., 3.])
def test_matrix_reference_is_exact_for_its_homogeneous_operator(shift):
    core_fft.set_fft_backend('scipy')
    projector = MixedProjector((5, 7, 5), (1., 2., 3.), [True, False, False, False, False, False])
    reference = IsotropicReference(projector, 2.5, 1.2)
    reference.set_shift(shift, np.dtype('float64'))
    stiffness = np.eye(6) * (2 * 1.2 + shift)
    stiffness[:3, :3] += 2.5 - 2 * 1.2 / 3
    field = projector(np.random.default_rng(5).normal(size=(6, 5, 7, 5)))
    rhs = projector(np.einsum('ij,j...->i...', stiffness, field))
    np.testing.assert_allclose(reference(rhs), field, atol=1e-12)


def test_default_fast_path_matches_strain_reference():
    fast = solve_nonlinear_homogenization(_case())
    reference = solve_nonlinear_homogenization(
        dict(_case(), formulation='strain', preconditioner='none'))
    assert fast['formulation'] == 'displacement'
    assert fast['preconditioner'] == 'matrix'
    np.testing.assert_allclose([row['stress'] for row in fast['history']],
                               [row['stress'] for row in reference['history']], atol=1e-9)
    np.testing.assert_allclose(fast['fields']['strain'], reference['fields']['strain'], atol=1e-9)
    np.testing.assert_allclose(fast['state'].equivalent, reference['state'].equivalent, atol=1e-9)


@pytest.mark.skipif(cp is None, reason='CuPy unavailable')
def test_gpu_fast_path_matches_cpu():
    cpu = solve_nonlinear_homogenization(_case())
    gpu = solve_nonlinear_homogenization(dict(_case(), backend='cupy'))
    assert gpu['coordinate_pcg'] == 'fused' and gpu['reuse_trial_response']
    assert gpu['inner_forcing'] == 'ew_target'
    assert cpu['coordinate_pcg'] == 'classic' and cpu['inner_forcing'] == 'ew'
    np.testing.assert_allclose([row['stress'] for row in gpu['history']],
                               [row['stress'] for row in cpu['history']], atol=2e-6)
    assert max(row['residual'] for row in gpu['history']) <= gpu['rtol']


@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
def test_reuse_accepted_equilibrium_is_exact_and_saves_evaluations(globalization, monkeypatch):
    from cufhom.pipeline import nonlinear_solver as solver
    original = solver.update_cpu
    calls = []
    def counted(e, state, *args):
        before = state.plastic_strain.copy(), state.equivalent.copy()
        answer = original(e, state, *args)
        np.testing.assert_array_equal(state.plastic_strain, before[0])
        np.testing.assert_array_equal(state.equivalent, before[1])
        calls.append(1)
        return answer
    monkeypatch.setattr(solver, 'update_cpu', counted)
    p = dict(_case(), globalization=globalization)
    old = solve_nonlinear_homogenization(dict(p, reuse_trial_response=False))
    old_calls = len(calls); calls.clear()
    new = solve_nonlinear_homogenization(dict(p, reuse_trial_response=True))
    assert old_calls-len(calls) == sum(r['newton_iterations'] for r in new['history'])
    for name in ('stress', 'strain', 'residual', 'cg_iterations'):
        np.testing.assert_array_equal([r[name] for r in new['history']],
                                      [r[name] for r in old['history']])
    np.testing.assert_array_equal(new['state'].plastic_strain, old['state'].plastic_strain)
    np.testing.assert_array_equal(new['state'].equivalent, old['state'].equivalent)


@pytest.mark.skipif(cp is None, reason='CuPy unavailable')
@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
def test_fused_coordinate_curve_and_restart(globalization, tmp_path):
    p = dict(_case(), backend='cupy', globalization=globalization)
    old = solve_nonlinear_homogenization(dict(p, coordinate_pcg='classic', reuse_trial_response=False))
    new = solve_nonlinear_homogenization(dict(p, coordinate_pcg='fused', reuse_trial_response=True))
    assert new['coordinate_pcg'] == 'fused' and not new['rejected_steps']
    for name in ('stress', 'strain'):
        np.testing.assert_allclose([r[name] for r in new['history']],
                                   [r[name] for r in old['history']], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(new['state'].equivalent, old['state'].equivalent, atol=2e-7)
    checkpoint = tmp_path/'confirmed.npz'
    options = dict(p, coordinate_pcg='fused', reuse_trial_response=True)
    solve_nonlinear_homogenization(dict(options, path=[0., .006], checkpoint_path=checkpoint))
    restarted = solve_nonlinear_homogenization(dict(options, path=[.006, .004], restart_from=checkpoint))
    np.testing.assert_allclose(restarted['state'].equivalent, new['state'].equivalent, atol=2e-7)
    assert max(r['residual'] for r in new['history']) <= new['rtol']


def test_fused_coordinate_policy_cpu_and_invalid_options():
    result = solve_nonlinear_homogenization(dict(_case(), coordinate_pcg='auto'))
    assert result['coordinate_pcg'] == 'classic'
    with pytest.raises(ValueError, match='requires CuPy'):
        solve_nonlinear_homogenization(dict(_case(), coordinate_pcg='fused'))
    with pytest.raises(ValueError, match='coordinate_pcg must'):
        solve_nonlinear_homogenization(dict(_case(), coordinate_pcg='typo'))


def test_cached_response_does_not_survive_rejected_attempt(monkeypatch):
    from cufhom.pipeline import nonlinear_solver as solver
    original = solver._cg
    counter = [0]
    def fail_second(*args, **kwargs):
        counter[0] += 1
        if counter[0] == 2:
            raise solver.NonlinearConvergenceError('injected after an accepted Newton update')
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, '_cg', fail_second)
    p = dict(_case(), globalization='newton')
    old = solve_nonlinear_homogenization(dict(p, reuse_trial_response=False))
    counter[0] = 0
    new = solve_nonlinear_homogenization(dict(p, reuse_trial_response=True))
    assert len(new['rejected_steps']) == len(old['rejected_steps']) == 1
    np.testing.assert_array_equal(new['fields']['strain'], old['fields']['strain'])
    np.testing.assert_array_equal(new['state'].equivalent, old['state'].equivalent)


@pytest.mark.skipif(cp is None, reason='CuPy unavailable')
def test_fused_failed_increment_retains_confirmed_checkpoint(tmp_path):
    from cufhom.pipeline.nonlinear_solver import NonlinearConvergenceError
    checkpoint = tmp_path/'failed.npz'
    with pytest.raises(NonlinearConvergenceError):
        solve_nonlinear_homogenization(dict(_case(), backend='cupy', coordinate_pcg='fused',
            reuse_trial_response=True, max_cg=1, globalization='newton',
            min_increment=.001, checkpoint_path=checkpoint))
    with np.load(checkpoint, allow_pickle=False) as saved:
        assert not np.any(saved['strain']) and not np.any(saved['equivalent'])
        assert not np.any(saved['plastic_strain'])


@pytest.mark.parametrize('backend', ['scipy', 'cupy'])
@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
def test_target_forcing_preserves_outer_acceptance_and_state(backend, globalization):
    if backend == 'cupy' and cp is None:
        pytest.skip('CuPy unavailable')
    p = dict(_case(), backend=backend, globalization=globalization,
             coordinate_pcg='auto', reuse_trial_response=True)
    old = solve_nonlinear_homogenization(dict(p, inner_forcing='ew'))
    new = solve_nonlinear_homogenization(dict(p, inner_forcing='ew_target'))
    assert not new['rejected_steps'] and len(new['history']) == len(old['history'])
    error = np.max(np.abs(np.array([r['stress'] for r in new['history']])-
                         np.array([r['stress'] for r in old['history']])))
    assert error / .015 <= .001
    assert max(r['residual'] for r in new['history']) <= new['rtol']
    np.testing.assert_allclose(new['state'].equivalent, old['state'].equivalent, atol=2e-7)
    if globalization == 'ptc':
        np.testing.assert_array_equal(new['fields']['strain'], old['fields']['strain'])


def test_target_forcing_does_not_relax_final_acceptance():
    from cufhom.pipeline.nonlinear_solver import _ew_tolerance
    # Near the goal, avoid solving to 0.008 when 0.3 is already sufficient.
    assert _ew_tolerance(1.5e-5, 3.5e-4, .04, 1e-5, target=1e-5) == .3
    assert _ew_tolerance(.1, None, None, 1e-5, target=1e-5) == .1
    for error in np.geomspace(1.001e-5, 1., 100):
        eta = _ew_tolerance(error, 2*error, .1, 1e-5, target=1e-5)
        assert 1e-5 <= eta <= .3
