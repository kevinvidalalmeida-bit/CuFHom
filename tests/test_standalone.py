"""Checks imports and commands without the parent checkout or inherited paths."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_imports_outside_checkout(tmp_path):
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    command = """
from pathlib import Path
import cufhom, cufhom_core
from cufhom.geometry import GeometryRequest, generate_geometry
from cufhom.fft import solve_homogenization
from cufhom.materials import engineering_constants_from_Cmandel
assert all(callable(f) for f in (GeometryRequest, generate_geometry, solve_homogenization,
    engineering_constants_from_Cmandel))
assert 'geometry_compiled' not in cufhom.__file__
assert 'geometry_compiled' not in cufhom_core.__file__
print(cufhom.__file__)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_geometry_help_outside_checkout(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "cufhom.cli.geometry", "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "--orientation-json" in result.stdout


def test_geometry_help_exposes_defaults_and_separates_numerical_controls(tmp_path):
    def help_text(flag):
        return subprocess.run(
            [sys.executable, "-I", "-m", "cufhom.cli.geometry", flag],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout

    normal = help_text("--help")
    complete = help_text("--help-all")
    assert "--vf" in normal and "0.15" in normal
    assert "--help-all" in normal
    assert "--candidates" not in normal and "--relax-iterations" not in normal
    assert "--candidates" in complete and "--relax-iterations" in complete


def test_active_sources_do_not_depend_on_parent():
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "core", "tests", "examples"):
        for source in (root / folder).rglob("*.py"):
            if source == Path(__file__).resolve():
                continue
            text = source.read_text()
            assert "COMPUTATIONAL_WORKSPACE" not in text, source
            assert "/home/user/" not in text, source
            assert "from geometry_runtime" not in text, source
            assert "import main as campaign" not in text, source
