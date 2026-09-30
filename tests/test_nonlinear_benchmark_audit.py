"""The latency comparison must not pass changed physics or relaxed controls."""
import copy
import importlib.util
from pathlib import Path
import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'benchmarks/compare_nonlinear_curves.py'
spec = importlib.util.spec_from_file_location('nonlinear_curve_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def run_data():
    row = dict(strain=[.001, 0., 0., 0., 0., 0.], stress=[.003, 0., 0., 0., 0., 0.],
        seconds=.5, cg_iterations=5, newton_iterations=2, residual=1e-9,
        macro_residual=1e-10, hill_mandel_error=1e-10, plastic_dissipation=0.,
        yield_violation=0., equivalent_plastic_mean=0.)
    return dict(fingerprint='geometry/material', configuration=dict(axis=0, shape=[3]*3),
        backend='scipy', dtype='float64', rtol=1e-8, initial_increment=.001,
        min_increment=1e-7, history=[row], rejected_steps=[], wall_seconds=1.,
        stress_scale_GPa=.015)


def test_benchmark_audit_identical_and_roundoff():
    a = run_data()
    b = copy.deepcopy(a)
    b['min_increment'] = .01*1e-5
    b['wall_seconds'] = .6
    r = audit.compare_runs(a, b)
    assert r['passed'] and r['max_stress_error_over_scale'] == 0
    assert r['wall_seconds_reduction_percent'] == pytest.approx(40.)


@pytest.mark.parametrize('key,value', [('fingerprint', 'other'), ('rtol', 1e-5),
                                      ('initial_increment', .002)])
def test_benchmark_rejects_different_problem(key, value):
    a = run_data()
    b = copy.deepcopy(a)
    b[key] = value
    with pytest.raises(ValueError, match=key):
        audit.compare_runs(a, b)


def test_benchmark_rejects_different_path():
    a = run_data()
    b = copy.deepcopy(a)
    b['history'][0]['strain'][0] *= -1
    with pytest.raises(ValueError, match='paths'):
        audit.compare_runs(a, b)


@pytest.mark.parametrize('field,value', [('stress', [.004, 0., 0., 0., 0., 0.]),
    ('residual', 1e-3), ('hill_mandel_error', 1e-3), ('yield_violation', 1e-3),
    ('plastic_dissipation', -1e-9)])
def test_benchmark_fails_changed_solution_or_failed_diagnostics(field, value):
    a = run_data()
    b = copy.deepcopy(a)
    b['history'][0][field] = value
    assert not audit.compare_runs(a, b)['passed']
