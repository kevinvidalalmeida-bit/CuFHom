"""Material-to-material recycling must preserve the original FFT equations."""
import json

import numpy as np
import pytest

from test_linear_coordinates import case, cp, gpu, solve_homogenization


def solve_fields(parameters, timing):
    p = dict(parameters, solver_timing_path=str(timing),
             solution_field_return_in_memory=True)
    ceff = solve_homogenization(p)
    report = json.loads(timing.read_text())
    infos = report['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
    assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= p['solver_rtol']
               for i in infos)
    return ceff, p['_solution_fields_result'], infos, report


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('preconditioning', ['matrix', 'symmetric_fused'])
def test_changed_material_warm_start_against_cpu(shape, preconditioning, tmp_path):
    p = dict(case(shape), linear_formulation='auto', linear_preconditioning=preconditioning)
    _, fields, _, _ = solve_fields(p, tmp_path/'first.json')
    original = [f.copy() for f in fields]
    new = dict(p, Em=4.1, nu_m=.36, Ef_L=85., Ef_T=9., G_LT=5.5)
    cold, _, cold_infos, _ = solve_fields(new, tmp_path/'cold.json')
    warm, _, infos, report = solve_fields(dict(new, initial_solution_fields=fields), tmp_path/'warm.json')
    reference = solve_homogenization(dict(new, fft_backend='scipy', solver_real_dtype='float64',
        cfield_indexed=False, solver_rtol=1e-10, linear_formulation='strain'))
    np.testing.assert_allclose(warm, reference, atol=3e-6, rtol=2e-6)
    np.testing.assert_allclose(warm, cold, atol=3e-6, rtol=2e-6)
    assert report['linear_formulation'] == 'displacement'
    assert all(i['warm_start_accepted'] for i in infos)
    # Tolerance remains based on the full new-material RHS, not b-A*x0.
    np.testing.assert_allclose([i['rhs_norm'] for i in infos], [i['rhs_norm'] for i in cold_infos])
    np.testing.assert_allclose([i['threshold'] for i in infos], [i['threshold'] for i in cold_infos])
    for a, b in zip(fields, original):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('gpu_fields', [False, True])
@pytest.mark.parametrize('scale', [1., 2.7])
def test_repeated_or_common_scaled_material_early_exit(gpu_fields, scale, tmp_path):
    p = case((5, 7, 5))
    ceff, fields, _, _ = solve_fields(dict(p, solver_rtol=2e-6), tmp_path/'first.json')
    changed = dict(p, **{k: scale*p[k] for k in ('Em', 'Ef_L', 'Ef_T', 'G_LT')})
    if gpu_fields:
        fields = tuple(cp.asarray(f) for f in fields)
    actual, _, infos, _ = solve_fields(dict(changed, initial_solution_fields=fields), tmp_path/'next.json')
    np.testing.assert_allclose(actual, ceff*scale, atol=3e-6, rtol=2e-6)
    assert all(i['cg_iterations'] == 0 and i['warm_start_accepted'] for i in infos)


@pytest.mark.parametrize('project_requested', [False, True])
def test_projection_removes_mean_and_incompatible_modes(project_requested, tmp_path):
    from pipeline.nonlinear_solver import MixedProjector
    p = case((6, 4, 8))
    expected, fields, _, _ = solve_fields(dict(p, solver_rtol=2e-6), tmp_path/'cold.json')
    rng = np.random.default_rng(231)
    projector_gpu = MixedProjector((6, 4, 8), (1., 1., 1.), np.ones(6, bool), cp)
    def projector(v):
        return cp.asnumpy(projector_gpu(cp.asarray(v, dtype=cp.float32)))
    # G*(I-G)=0, so this contamination must have no effect on the solution.
    contaminated = []
    for f in fields:
        v = rng.normal(size=f.shape)
        contaminated.append((f+v-projector(v)+.7).astype(np.float32))
    before = [f.copy() for f in contaminated]
    actual, result_fields, infos, _ = solve_fields(dict(p, initial_solution_fields=contaminated,
        project_initial_solution_fields=project_requested), tmp_path/'warm.json')
    np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=2e-6)
    for value in result_fields:
        np.testing.assert_allclose(value, projector(value), atol=2e-6)
        np.testing.assert_allclose(value.mean(axis=(1, 2, 3)), 0, atol=2e-7)
    for a, b in zip(contaminated, before):
        np.testing.assert_array_equal(a, b)
    assert all(i['warm_start_projected'] for i in infos)


def test_worse_guess_reverts_to_cold_without_changing_sample(tmp_path):
    p = case((5, 7, 5))
    cold, fields, cold_infos, _ = solve_fields(p, tmp_path/'cold.json')
    bad = [f*100 for f in fields]
    actual, _, infos, _ = solve_fields(dict(p, initial_solution_fields=bad), tmp_path/'bad.json')
    np.testing.assert_array_equal(actual, cold)
    assert all(i['warm_start_requested'] and not i['warm_start_accepted'] for i in infos)
    assert [i['cg_iterations'] for i in infos] == [i['cg_iterations'] for i in cold_infos]


@pytest.mark.parametrize('value', [np.nan, np.inf])
def test_nonfinite_snapshot_is_rejected(value):
    p = case((4, 4, 4))
    fields = np.zeros((6, 6, 4, 4, 4), np.float32)
    fields[0, 0, 1, 0, 0] = value
    with pytest.raises(RuntimeError, match='finite'):
        solve_homogenization(dict(p, initial_solution_fields=fields))


def test_wrong_snapshot_size_is_rejected():
    with pytest.raises(RuntimeError, match='entries; expected'):
        solve_homogenization(dict(case((4, 4, 4)), initial_solution_fields=np.zeros((6, 7))))


def test_homogeneous_material_rejects_nonzero_guess(tmp_path):
    p = case((4, 4, 4)); p['phase_array'][:] = 0
    fields = np.random.default_rng(911).normal(size=(6, 6, 4, 4, 4)).astype(np.float32)
    _, _, infos, _ = solve_fields(dict(p, initial_solution_fields=fields), tmp_path/'zero.json')
    assert all(i['cg_iterations'] == 0 and not i['warm_start_accepted'] for i in infos)


def test_warm_start_does_not_hide_a_failed_coordinate_solve(monkeypatch, tmp_path):
    from pipeline import linear_coordinates
    p = case((5, 7, 5))
    _, fields, _, _ = solve_fields(p, tmp_path/'first.json')
    target = dict(p, Em=4.1, Ef_L=82.)
    expected, _, _, _ = solve_fields(target, tmp_path/'cold.json')
    monkeypatch.setattr(linear_coordinates, 'fused_cg',
                        lambda action, rhs, *a, **kw: (cp.zeros_like(rhs), 1))
    actual, _, infos, _ = solve_fields(dict(target, initial_solution_fields=fields), tmp_path/'warm.json')
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-6)
    assert all(i['correction_iterations'] > 0 for i in infos)
