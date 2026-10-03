"""Linear displacement PCG: reference accuracy, physical audits and failures."""
import json
import numpy as np
import pytest

from cufhom.fft.linear.solver import solve_homogenization, cp


def case(shape):
    phase = np.zeros(shape, np.uint8)
    phase[1:3, 1:3, :] = 1
    ori = np.zeros(shape+(3,), np.float32)
    ori[:] = np.array([1., 2., 3.])/np.sqrt(14)
    return dict(input_dir='.', phase_array=phase, ori_array=ori,
                Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.,
                fft_backend='cupy', solver_real_dtype='float32', solver_rtol=1e-5,
                solver_fft_form='r', projection_storage='direct', cfield_storage='sym21',
                cfield_indexed=True, cupy_unscaled_fft_pair=True,
                keep_solution_on_device=True, postprocess_batch_size=1,
                postprocess_assembly='scalar', linear_formulation='displacement')


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('No CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
def test_six_load_cpu_reference_and_true_residual(shape, tmp_path):
    p = case(shape)
    reference = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, cfield_indexed=False, linear_formulation='strain'))
    timing = tmp_path/'timing.json'
    candidate = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    assert np.linalg.norm(candidate-reference)/np.linalg.norm(reference) < 2e-6
    report = json.loads(timing.read_text())
    assert report['linear_formulation'] == 'displacement'
    infos = report['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert len(infos) == 6
    assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)


def test_nonconvergence_is_not_accepted():
    with pytest.raises(RuntimeError):
        solve_homogenization(dict(case((6, 4, 8)), solver_maxiter=1))


def test_homogeneous_zero_rhs_and_host_result():
    p = case((4, 4, 4)); p['phase_array'][:] = 0
    candidate = solve_homogenization(dict(p, keep_solution_on_device=False))
    reference = solve_homogenization(dict(p, linear_formulation='strain'))
    np.testing.assert_allclose(candidate, reference, atol=1e-6)


def test_incompatible_configuration_is_explicit():
    with pytest.raises(ValueError, match='displacement requires'):
        solve_homogenization(dict(case((4, 4, 4)), cupy_unscaled_fft_pair=False))


@pytest.mark.parametrize('selection', [None, 'auto', 'strain'])
@pytest.mark.parametrize('backend', ['cupy', 'scipy', 'scaled_cupy'])
def test_default_selection_and_legacy_override(selection, backend, tmp_path):
    p = case((4, 4, 4))
    p.pop('linear_formulation')
    if selection is not None:
        p['linear_formulation'] = selection
    if backend == 'scipy':
        p.update(fft_backend='scipy', solver_real_dtype='float64', cfield_indexed=False)
    elif backend == 'scaled_cupy':
        p['cupy_unscaled_fft_pair'] = False
    expected = 'displacement' if backend == 'cupy' and selection != 'strain' else 'strain'
    timing = tmp_path/'selection.json'
    candidate = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    metadata = json.loads(timing.read_text())
    assert metadata['linear_formulation'] == expected
    assert metadata['linear_formulation_requested'] == (selection or 'auto')
    reference = solve_homogenization(dict(p, linear_formulation=expected))
    np.testing.assert_allclose(candidate, reference, atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize('preconditioning', ['matrix', 'symmetric_fused'])
def test_audit_corrects_inaccurate_coordinate_iterate(monkeypatch, preconditioning):
    import cufhom.fft.linear.coordinates as linear_coordinates
    p = case((5, 7, 5))
    p['linear_preconditioning'] = preconditioning
    reference = solve_homogenization(p)
    # Simulate a false inner convergence; acceptance must use the physical
    # residual and recover through the original equations.
    monkeypatch.setattr(linear_coordinates, '_cg' if preconditioning == 'matrix' else 'fused_cg',
                        lambda action, rhs, *a, **kw: (cp.zeros_like(rhs), 1))
    corrected = solve_homogenization(p)
    np.testing.assert_allclose(corrected, reference, atol=3e-6, rtol=2e-6)


@pytest.mark.parametrize('selection', [None, 'matrix', 'symmetric_fused'])
def test_linear_preconditioning_selection(selection, tmp_path):
    p = case((5, 6, 4))
    if selection is not None:
        p['linear_preconditioning'] = selection
    timing = tmp_path/'preconditioning.json'
    actual = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    metadata = json.loads(timing.read_text())
    assert metadata['linear_preconditioning'] == (selection or 'symmetric_fused')
    ref = solve_homogenization(dict(p, linear_preconditioning='matrix'))
    np.testing.assert_allclose(actual, ref, rtol=2e-6, atol=3e-6)


def test_unknown_linear_preconditioning_is_rejected():
    with pytest.raises(ValueError, match='linear_preconditioning'):
        solve_homogenization(dict(case((4, 4, 4)), linear_preconditioning='unknown'))


@pytest.mark.parametrize('smoothing', [False, True])
def test_smoothing_selection_and_audits(smoothing, tmp_path):
    p = dict(case((5, 7, 5)), linear_residual_smoothing=smoothing)
    timing = tmp_path/'smoothing.json'
    actual = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    reference = solve_homogenization(dict(p, fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, cfield_indexed=False, linear_formulation='strain'))
    np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=2e-6)
    report = json.loads(timing.read_text())
    assert report['linear_residual_smoothing'] is smoothing
    infos = report['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert all(i['linear_residual_smoothing'] is smoothing and
               i['true_norm_res_rel'] <= 1e-5 for i in infos)
