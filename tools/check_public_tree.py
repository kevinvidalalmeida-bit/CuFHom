"""Reject reserved research files anywhere in a public commit history."""

import argparse
import subprocess
import sys
from pathlib import PurePosixPath

PUBLIC_ROOTS = {
    ".gitattributes",
    ".github",
    ".gitignore",
    "AGENTS.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "benchmarks",
    "core",
    "docs",
    "examples",
    "pyproject.toml",
    "src",
    "tests",
    "third_party",
    "tools",
}
PRIVATE_PARTS = {
    "paper",
    "research",
    "external",
    "results",
    "records",
    "archive",
    "archives",
    "reviews",
    "figuras_finales",
    "nonlinear",
    "CuFHom-NOLINEAL",
    ".venv",
    "build",
    "dist",
    "__pycache__",
}
PRIVATE_MATERIALS = {
    "double_float.py",
    "drucker_prager.py",
    "gradient_damage.py",
    "paraboloidal.py",
    "plasticity.py",
}
PRIVATE_SUFFIXES = {
    ".tex",
    ".pdf",
    ".bbl",
    ".aux",
    ".log",
    ".zip",
    ".gz",
    ".whl",
    ".pyc",
}


def reserved(path):
    """Allow benchmark inputs while rejecting manuscript and research paths."""
    parts = PurePosixPath(path).parts
    if not parts or parts[0] not in PUBLIC_ROOTS:
        return True
    for index, part in enumerate(parts):
        if part == "paper" and index == 1 and parts[0] == "benchmarks":
            continue
        if part in PRIVATE_PARTS or "nonlinear" in part.lower():
            return True
    return (
        parts[-1] in PRIVATE_MATERIALS or PurePosixPath(path).suffix in PRIVATE_SUFFIXES
    )


def git(*args):
    """Read the repository in the caller's current working directory."""
    return subprocess.check_output(["git", *args])


def main():
    """Validate current refs and optional additional tips before publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Inspect every local ref and its ancestors."
    )
    parser.add_argument(
        "revisions", nargs="*", help="Additional commit or tag tips to inspect."
    )
    args = parser.parse_args()
    tips = (["--all"] if args.all else []) + args.revisions
    commits = git("rev-list", *(tips or ["HEAD"])).decode().splitlines()
    trees = set()
    violations = []
    for commit in commits:
        tree = git("show", "-s", "--format=%T", commit).decode().strip()
        if tree in trees:
            continue
        trees.add(tree)
        paths = git("ls-tree", "-r", "-z", "--name-only", tree).decode().split("\0")
        violations.extend((commit, path) for path in paths if path and reserved(path))
    if violations:
        print(
            "Public push blocked: reserved paths exist in the proposed history.",
            file=sys.stderr,
        )
        for commit, path in violations[:20]:
            print(f"{commit[:12]} {path}", file=sys.stderr)
        return 1
    print(f"Public boundary verified: {len(commits)} commits, {len(trees)} file trees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
