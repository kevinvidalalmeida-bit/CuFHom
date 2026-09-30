"""Default GPU solve must match CPU64 and respect incompatible overrides."""
import json
import numpy as np
import pytest

from cufhom.pipeline.linear_defaults import linear_defaults
from cufhom.pipeline.fft_solver import cp, solve_homogenization


def test_defaults_preserve_caller_and_cpu_profiles():
    p = dict(fft_backend='scipy', solver_rtol=1e-10)
    assert linear_defaults(p, use_gpu=False, profile={}) == p
    gpu = linear_defaults(p, use_gpu=True, profile={'solver_real_dtype': 'float64'})
    assert 'cfield_indexed' not in gpu
    assert 'solver_fft_form' not in gpu
    assert p == dict(fft_backend='scipy', solver_rtol=1e-10)


def test_explicit_opt_out_preserved():
    p = dict(cfield_indexed=False, cupy_fused_xr_rr=False, cupy_preserve_plans=False,
             cupy_reuse_inverse_input=False, keep_solution_on_device=False)
    resolved = linear_defaults(p, use_gpu=True, profile={})
    assert all(resolved[k] is False for k in p)


@pytest.mark.parametrize('old_option', [dict(cupy_matrix_reference=True), dict(cupy_reference_eta=2.)])
def test_retired_metric_is_explicit(old_option):
    with pytest.raises(ValueError, match='retired'):
        solve_homogenization(old_option)


@pytest.mark.parametrize('override', [dict(cfield_storage='full'),
    dict(projection_storage='full'), dict(solver_fft_form='c'),
    dict(force_disk_cfield=True), dict(use_cfield_material_fast_path=False),
    dict(cfield_origin='centered'), dict(Cfield_array=np.zeros(1)),
    dict(fiber_fraction_array=np.zeros(1))])
def test_indexing_not_selected_for_incompatible_input(override):
    p = linear_defaults(override, use_gpu=True, profile={})
    assert not p['cfield_indexed']
    for k, v in override.items():
        assert p[k] is v


@pytest.mark.parametrize('shape', [(5, 7, 5), (6, 4, 8)])
@pytest.mark.parametrize('geometry', ['block', 'laminate', 'inclusions'])
def test_minimal_gpu_six_loads_match_cpu64(shape, geometry, tmp_path):
    if cp is None:
        pytest.skip('CuPy unavailable')
    phase = np.zeros(shape, np.uint8)
    if geometry == 'block':
        phase[1:3, 1:3, :] = 1
    elif geometry == 'laminate':
        phase[:2] = 1
    else:
        # Disconnected compact inclusions; no fibre axis or aspect ratio.
        coordinates = np.indices(shape)
        for center in ((1, 1, 1), tuple(n-2 for n in shape)):
            phase[np.sum((coordinates-np.array(center)[:, None, None, None])**2, axis=0) <= 1] = 1
    ori = np.empty(shape+(3,), np.float32); ori[:] = np.array([1., 2., 3.])/np.sqrt(14)
    p = dict(input_dir='.', phase_array=phase, ori_array=ori,
        Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.)
    ref = solve_homogenization(dict(p, fft_backend='scipy', solver_rtol=1e-10))
    timing = tmp_path/'gpu.json'
    result = solve_homogenization(dict(p, fft_backend='cupy', solver_rtol=1e-5,
                                     solver_timing_path=str(timing)))
    assert np.linalg.norm(result-ref)/np.linalg.norm(ref) < 2e-6
    detail = json.loads(timing.read_text())
    assert detail['linear_formulation'] == 'displacement'
    assert detail['cfield_indexed']
    assert detail['cupy_preserve_plans'] and detail['keep_solution_on_device']
    assert detail['postprocess_batch_size'] == 1
    infos = detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert len(infos) == 6
    assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)


def test_general_anisotropic_cfield_retains_compatible_route(tmp_path):
    if cp is None:
        pytest.skip('CuPy unavailable')
    shape = (5, 4, 7)
    rng = np.random.default_rng(309)
    base = rng.normal(size=(6, 6))
    base = base.T @ base + 3*np.eye(6)
    # Smooth arbitrary SPD tensor field, independent of phase/orientation labels.
    scale = 1 + .3*np.sin(2*np.pi*np.indices(shape)[0]/shape[0])
    cfield = base[:, :, None, None, None]*scale
    saved = cfield.copy()
    p = dict(input_dir='.', phase_array=np.zeros(shape, np.uint8),
        ori_array=np.zeros(shape+(3,), np.float32), Cfield_array=cfield,
        Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.)
    ref = solve_homogenization(dict(p, fft_backend='scipy', solver_rtol=1e-10))
    timing = tmp_path/'general.json'
    result = solve_homogenization(dict(p, fft_backend='cupy', solver_rtol=1e-5,
                                     solver_timing_path=str(timing)))
    assert np.linalg.norm(result-ref)/np.linalg.norm(ref) < 2e-6
    np.testing.assert_array_equal(cfield, saved)
    detail = json.loads(timing.read_text())
    assert not detail['cfield_indexed']
    assert detail['linear_formulation'] == 'strain'
    assert detail['load_solver_summary']['all_converged']
