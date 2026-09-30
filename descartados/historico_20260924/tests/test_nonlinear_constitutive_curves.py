"""Six independent experiments, shear scaling, restart and exported CLI curves."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'FFT'))
from pipeline.fft_solver import cp, solve_homogenization
from pipeline.nonlinear_material import J2Material
from pipeline.nonlinear_solver import solve_nonlinear_homogenization
from pipeline.nonlinear_curves import (MANDEL_COMPONENTS, CONSTITUTIVE_COMPONENTS,
                                       constitutive_values)

RUNNER = ROOT/'FFT/benchmarks/run_nonlinear_fft.py'


def case(n=5, composite=False):
    phase = np.zeros((n, n, n), dtype=np.uint8)
    if composite:
        phase[n//2, :, :] = 1
    ori = np.zeros(phase.shape+(3,))
    ori[..., 0] = 1
    return dict(phase_array=phase, ori_array=ori, backend='scipy')


@pytest.mark.parametrize('axis', [0, 1, 2])
def test_all_normal_directions_match_uniaxial_solution(axis):
    result = solve_nonlinear_homogenization(dict(case(), axis=axis, path=[0, .012],
                                                initial_increment=.001))
    strain, stress = constitutive_values(result)
    m = J2Material()
    expected = np.where(strain <= m.yield_stress/m.E, m.E*strain,
                        m.yield_stress+m.E*m.hardening/(m.E+m.hardening)*
                        (strain-m.yield_stress/m.E))
    np.testing.assert_allclose(stress, expected, atol=1e-10)
    free = [i for i in range(6) if i != axis]
    np.testing.assert_allclose(np.asarray([r['stress'] for r in result['history']])[:, free],
                               0, atol=1e-10)


@pytest.mark.parametrize('axis', [3, 4, 5])
@pytest.mark.parametrize('formulation', ['strain', 'displacement'])
def test_shear_yield_hardening_and_unloading(axis, formulation):
    result = solve_nonlinear_homogenization(dict(case(), axis=axis,
        path=np.array([0, .02, 0])/np.sqrt(2.), initial_increment=.001/np.sqrt(2.),
        formulation=formulation))
    gamma, tau = constitutive_values(result)
    m = J2Material()
    tau_y = m.yield_stress/np.sqrt(3.)
    gamma_y = tau_y/m.mu
    slope = m.mu*m.hardening/(3*m.mu+m.hardening)
    expected = np.where(gamma[:21] <= gamma_y, m.mu*gamma[:21],
                        tau_y+slope*(gamma[:21]-gamma_y))
    np.testing.assert_allclose(tau[:21], expected, atol=1e-11)
    assert (tau[20]-tau[21])/(gamma[20]-gamma[21]) == pytest.approx(m.mu)
    assert tau[-1] < 0
    assert result['history'][-1]['equivalent_plastic_mean'] > result['history'][20]['equivalent_plastic_mean']
    free = [i for i in range(6) if i != axis]
    np.testing.assert_allclose(np.asarray([r['stress'] for r in result['history']])[:, free],
                               0, atol=1e-11)
    assert max(r['hill_mandel_error'] for r in result['history']) < 1e-10


@pytest.fixture(scope='module')
def elastic_composite():
    p = case(composite=True)
    stiffness = solve_homogenization(dict(input_dir='.', phase_array=p['phase_array'],
        ori_array=p['ori_array'], Em=3., nu_m=.3, Ef_L=70., Ef_T=8., nu_LT=.25,
        nu_TT=.3, G_LT=5., fft_backend='scipy', solver_real_dtype='float64',
        solver_rtol=1e-10, solver_maxiter=1000, require_convergence=True,
        cfield_storage='sym21'))
    return p, np.linalg.inv(stiffness)


@pytest.mark.parametrize('axis', range(6))
def test_six_initial_slopes_match_linear_compliance(axis, elastic_composite):
    p, compliance = elastic_composite
    scale = np.sqrt(2.) if axis >= 3 else 1.
    result = solve_nonlinear_homogenization(dict(p, axis=axis, path=[0, .0001/scale],
                                                initial_increment=.0001/scale))
    strain, stress = constitutive_values(result)
    expected_modulus = 1/(compliance[axis, axis]*scale**2)
    assert stress[-1]/strain[-1] == pytest.approx(expected_modulus, rel=1e-6)
    assert result['history'][-1]['equivalent_plastic_mean'] == 0.


@pytest.mark.parametrize('axis', [3, 4, 5])
def test_shear_checkpoint_continues_same_component_only(axis, tmp_path):
    p = dict(case(3), axis=axis, initial_increment=.001/np.sqrt(2.), return_fields=True)
    peak = .012/np.sqrt(2.)
    full = solve_nonlinear_homogenization(dict(p, path=[0, peak, 0]))
    checkpoint = tmp_path/'checkpoint.npz'
    solve_nonlinear_homogenization(dict(p, path=[0, peak], checkpoint_path=checkpoint))
    resumed = solve_nonlinear_homogenization(dict(p, path=[peak, 0], restart_from=checkpoint))
    np.testing.assert_allclose(resumed['fields']['strain'], full['fields']['strain'], atol=1e-13)
    np.testing.assert_allclose(resumed['state'].plastic_strain, full['state'].plastic_strain, atol=1e-13)
    with pytest.raises(ValueError, match='mismatch'):
        solve_nonlinear_homogenization(dict(p, axis=(axis+1)%6, path=[peak, 0],
                                            restart_from=checkpoint))


@pytest.mark.parametrize('axis', [3, 4, 5])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_composite_shear_gpu_matches_cpu(axis, dtype):
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA unavailable')
    except Exception:
        pytest.skip('CUDA unavailable')
    p = dict(case(composite=True), axis=axis, path=np.array([0, .012, 0])/np.sqrt(2.),
             initial_increment=.002/np.sqrt(2.))
    cpu = solve_nonlinear_homogenization(p)
    gpu = solve_nonlinear_homogenization(dict(p, backend='cupy', dtype=dtype))
    np.testing.assert_allclose(constitutive_values(gpu), constitutive_values(cpu), atol=2e-7)
    assert max(r['residual'] for r in gpu['history']) <= gpu['rtol']


def test_cli_six_curves_physical_units_fresh_states_and_outputs(tmp_path):
    output = tmp_path/'six'
    command = [sys.executable, str(RUNNER), '--component', 'all', '--maximum', '.01',
               '--increment', '.001', '--unload', '--checkpoint', '--output', str(output)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    with (output/'constitutive_curves.csv').open() as stream:
        combined = list(csv.DictReader(stream))
    assert {r['component'] for r in combined} == set(CONSTITUTIVE_COMPONENTS)
    for component in CONSTITUTIVE_COMPONENTS:
        axis = MANDEL_COMPONENTS.index(component)
        result = json.loads((output/component/'metadata.json').read_text())
        assert result['configuration']['axis'] == axis
        assert result['history'][0]['equivalent_plastic_mean'] == 0.
        assert result['history'][0]['stress'] == [0.]*6
        with (output/component/'constitutive.csv').open() as stream:
            physical = list(csv.DictReader(stream))
        e_key = ('gamma_' if axis >= 3 else 'epsilon_')+component
        s_key = ('tau_' if axis >= 3 else 'sigma_')+component+'_GPa'
        strain, stress = constitutive_values(result)
        np.testing.assert_allclose([float(r[e_key]) for r in physical], strain)
        np.testing.assert_allclose([float(r[s_key]) for r in physical], stress)
        assert max(strain) == pytest.approx(.01)
        assert strain[-1] == pytest.approx(0, abs=1e-12)
        assert result['history'][-1]['pseudo_time'] == pytest.approx(2.)
        assert (output/component/'checkpoint.npz').is_file()
        rows = [r for r in combined if r['component'] == component]
        np.testing.assert_allclose([float(r['stress_GPa']) for r in rows], stress)
    assert (output/'constitutive_curves.png').stat().st_size > 1000
    assert (output/'constitutive_curves.pdf').stat().st_size > 1000
    saved = (output/'11/metadata.json').read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and 'already contains a run' in refused.stderr
    (output/'metadata.json').unlink()
    refused = subprocess.run(command, capture_output=True, text=True)
    assert refused.returncode != 0 and 'already contains a run' in refused.stderr
    assert (output/'11/metadata.json').read_bytes() == saved


def test_cli_all_rejects_shared_restart(tmp_path):
    proc = subprocess.run([sys.executable, str(RUNNER), '--component', 'all',
                           '--restart', str(tmp_path/'state.npz')],
                          capture_output=True, text=True)
    assert proc.returncode != 0 and 'independent virgin states' in proc.stderr
