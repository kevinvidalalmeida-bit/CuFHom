"""The public snapshot must run independently and retain private work locally."""

import json
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    private = {
        "paper/MAIN.tex": "unpublished manuscript",
        "src/fft/nonlinear/solver.py": "unpublished solver",
        "src/materials/plasticity.py": "unpublished material",
        "src/cli/nonlinear.py": "unpublished command",
        "tests/fft/nonlinear/test_solver.py": "unpublished test",
    }
    files = private | {
        "src/fft/linear/__init__.py": '"""Public linear solver."""\n',
        "src/fft/__init__.py": "private exports replaced by the public API",
        "src/materials/__init__.py": "private exports replaced by the public API",
        "pyproject.toml": (ROOT / "pyproject.toml").read_text(),
        ".gitattributes": (ROOT / ".gitattributes").read_text(),
        "benchmarks/paper/inputs/case.json": json.dumps({"public": True}),
        "tools/export_public.py": (ROOT / "tools/export_public.py").read_text(),
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    commit(tmp_path)
    return tmp_path, private


def commit(path):
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Export test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Export fixture",
        ],
        cwd=path,
        check=True,
    )


def export(path, destination):
    return subprocess.run(
        [sys.executable, str(path / "tools/export_public.py"), str(destination)],
        cwd=path,
        capture_output=True,
        text=True,
    )


def test_snapshot_excludes_private_work_and_has_linear_package_metadata(checkout):
    root, private = checkout
    target = root / "public.tar.gz"
    result = export(root, target)
    assert result.returncode == 0, result.stderr
    with tarfile.open(target) as archive:
        names = archive.getnames()
        assert all(f"cufhom/{name}" not in names for name in private)
        assert "cufhom/.git" not in names
        public_input = archive.extractfile("cufhom/benchmarks/paper/inputs/case.json")
        assert json.load(public_input) == {"public": True}
        metadata = tomllib.loads(
            archive.extractfile("cufhom/pyproject.toml").read().decode()
        )
        assert "cufhom-nonlinear" not in metadata["project"]["scripts"]
        assert "cufhom.fft.nonlinear" not in metadata["tool"]["setuptools"]["packages"]
        for path in ("src/fft/__init__.py", "src/materials/__init__.py"):
            content = archive.extractfile(f"cufhom/{path}").read().decode()
            assert "nonlinear" not in content and "plasticity" not in content
            compile(content, path, "exec")
    for path, content in private.items():
        assert (root / path).read_text() == content
    before = target.read_bytes()
    assert export(root, target).returncode != 0
    assert target.read_bytes() == before


def test_missing_exclusion_aborts_without_writing_public_artifact(checkout):
    root, _ = checkout
    attributes = root / ".gitattributes"
    attributes.write_text(
        attributes.read_text().replace("/src/cli/nonlinear.py export-ignore", "")
    )
    subprocess.run(["git", "add", ".gitattributes"], cwd=root, check=True)
    commit(root)
    target = root / "leaked.tar.gz"
    result = export(root, target)
    assert result.returncode != 0
    assert "Private path" in result.stderr
    assert not target.exists()
