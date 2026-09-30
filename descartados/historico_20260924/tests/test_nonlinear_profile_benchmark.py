"""Persisted references must compare by values, not tuple/list serialization."""
import copy
import importlib
import json
from pathlib import Path
import sys


def test_profile_accepts_same_configuration_loaded_from_json(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'FFT/benchmarks'))
    profile = importlib.import_module('profile_nonlinear_stages')
    row = dict(strain=[0.]*6, stress=[0.]*6, target_strain=0., seconds=0.,
        cg_iterations=0, newton_iterations=0, residual=0., macro_residual=0.,
        hill_mandel_error=0., plastic_dissipation=0., yield_violation=0.,
        equivalent_plastic_mean=0.)
    run = dict(fingerprint='same', configuration=dict(axis=0, shape=[3]*3,
        lengths=(1., 1., 1.)), backend='cupy', dtype='float32', rtol=1e-5,
        initial_increment=.00025, min_increment=1e-7, history=[row],
        rejected_steps=[], wall_seconds=1., stress_scale_GPa=.015)
    reference = tmp_path/'reference.json'
    reference.write_text(json.dumps(run))
    monkeypatch.setattr(profile, 'parameters', lambda: {})
    monkeypatch.setattr(profile, 'sync', lambda: None)
    monkeypatch.setattr(profile.solver, 'solve_nonlinear_homogenization', lambda p: copy.deepcopy(run))
    output = tmp_path/'output'
    monkeypatch.setattr(sys, 'argv', ['profile', '--output', str(output),
                        '--modes', 'plain_before', '--reference', str(reference)])
    profile.main()
    report = json.loads((output/'report.json').read_text())
    assert report['runs'][0]['comparison']['passed']
    assert (output/'plain_before/metadata.json').exists()
