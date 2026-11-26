"""Public pushes must reject reserved files even when archives omit them."""

import io
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tools/check_public_tree.py"


@pytest.fixture
def repository(tmp_path):
    """Create an independent Git history for each publication check."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    return tmp_path


def commit(root, files):
    """Record fixture contents without using personal author settings."""
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    subprocess.run(["git", "add", "--all"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Publication test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Fixture",
        ],
        cwd=root,
        check=True,
    )


def check(root):
    """Execute the same checker used by CI and the local pre-push hook."""
    return subprocess.run(
        [sys.executable, str(CHECKER), "--all"],
        cwd=root,
        capture_output=True,
        text=True,
    )


def test_public_benchmark_drivers_and_inputs_are_allowed(repository):
    commit(
        repository,
        {
            "README.md": "Public software",
            "benchmarks/paper/run.py": "pass\n",
            "benchmarks/paper/inputs/materials.json": "{}\n",
        },
    )
    assert check(repository).returncode == 0


def test_export_ignore_does_not_make_private_commits_safe(repository):
    commit(
        repository,
        {
            ".gitattributes": "/paper export-ignore\n",
            "paper/MAIN.tex": "reserved",
            "README.md": "Public software",
        },
    )
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", "HEAD"], cwd=repository
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as snapshot:
        assert "paper/MAIN.tex" not in snapshot.getnames()
    result = check(repository)
    assert result.returncode == 1
    assert "paper/MAIN.tex" in result.stderr


def test_private_ancestor_is_rejected_even_after_file_deletion(repository):
    commit(repository, {"paper/MAIN.tex": "reserved", "README.md": "Public software"})
    (repository / "paper/MAIN.tex").unlink()
    commit(repository, {"README.md": "Public software updated"})
    assert "paper/MAIN.tex" not in subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repository, text=True
    )
    assert check(repository).returncode == 1


@pytest.mark.parametrize(
    "path",
    [
        "src/fft/nonlinear/solver.py",
        "src/materials/plasticity.py",
        "src/cli/nonlinear.py",
    ],
)
def test_nonlinear_implementation_is_rejected(repository, path):
    commit(
        repository, {path: "reserved implementation", "README.md": "Public software"}
    )
    result = check(repository)
    assert result.returncode == 1
    assert path in result.stderr
