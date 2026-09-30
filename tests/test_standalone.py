"""Checks imports and commands without the parent checkout or inherited paths."""
import os
from pathlib import Path
import subprocess
import sys


def test_imports_outside_checkout(tmp_path):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    command = '''
from pathlib import Path
import cufhom, cufhom_core
from cufhom.pipeline import sam_orientation, fft_solver, nonlinear_solver
assert 'geometry_compiled' not in cufhom.__file__
assert 'geometry_compiled' not in cufhom_core.__file__
print(cufhom.__file__)
'''
    subprocess.run([sys.executable, '-I', '-c', command], cwd=tmp_path, env=environment,
                   check=True, capture_output=True, text=True, timeout=30)


def test_sobol_help_does_not_require_cuda(tmp_path):
    result = subprocess.run([sys.executable, '-I', '-m', 'cufhom.cli.sobol', '--help'],
                            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30)
    assert '--smoke' in result.stdout


def test_geometry_help_outside_checkout(tmp_path):
    result = subprocess.run([sys.executable, '-I', '-m', 'cufhom.cli.geometry', '--help'],
                            cwd=tmp_path, check=True, capture_output=True, text=True, timeout=30)
    assert '--orientation-json' in result.stdout


def test_active_sources_do_not_depend_on_parent():
    root = Path(__file__).resolve().parents[1]
    for folder in ('src/cufhom', 'tests', 'benchmarks'):
        for source in (root/folder).rglob('*.py'):
            if source == Path(__file__).resolve():
                continue
            text = source.read_text()
            assert 'COMPUTATIONAL_WORKSPACE' not in text, source
            assert '/home/user/' not in text, source
            assert 'from geometry_runtime' not in text, source
            assert 'import main as campaign' not in text, source
