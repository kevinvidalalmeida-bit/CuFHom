"""Export the committed public source without private paper files or Git history."""

import argparse
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PARTS = {
    "paper",
    "CuFHom-NOLINEAL",
    "nonlinear",
    "nonlinear.py",
    ".git",
    "external",
    "double_float.py",
    "drucker_prager.py",
    "gradient_damage.py",
    "paraboloidal.py",
    "plasticity.py",
    "test_nonlinear_models.py",
}


def public_content(name, content):
    """Limit package metadata and exports to the public linear API."""
    if name == "src/fft/__init__.py":
        return b'''"""Linear FFT homogenization of periodic elastic materials."""

from .grids import select_grid
from .krylov import ConvergenceError
from .linear.common import IndexedMaterial, SolveOptions
from .linear.framework import homogenize
from .linear.solver import solve_homogenization

__all__ = [
    "select_grid",
    "ConvergenceError",
    "IndexedMaterial",
    "SolveOptions",
    "homogenize",
    "solve_homogenization",
]
'''
    if name == "src/materials/__init__.py":
        return b'''"""Elastic materials in Mandel notation."""

from .elasticity import engineering_constants_from_Cmandel

__all__ = ["engineering_constants_from_Cmandel"]
'''
    if name == "pyproject.toml":
        text = content.decode().replace("linear/nonlinear", "linear")
        text = text.replace('cufhom-nonlinear = "cufhom.cli.nonlinear:main"\n', "")
        text = text.replace('    "cufhom.fft.nonlinear",\n', "")
        return text.encode()
    return content


def main():
    """Validate an archive of HEAD before writing a new public snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New .tar.gz archive of committed HEAD."
    )
    args = parser.parse_args()
    if not args.output.name.endswith(".tar.gz"):
        parser.error("Output must end in .tar.gz")
    if args.output.exists():
        parser.error("Output already exists; choose a new archive name.")
    archive = subprocess.run(
        ["git", "archive", "--format=tar.gz", "--prefix=cufhom/", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    result = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source,
        tarfile.open(fileobj=result, mode="w:gz") as destination,
    ):
        for member in source.getmembers():
            parts = Path(member.name).parts[1:]
            if (
                (parts and parts[0] in {"paper", "CuFHom-NOLINEAL", ".git", "external"})
                or (parts and parts[0] != "benchmarks" and set(parts) & PRIVATE_PARTS)
                or parts[:3] == ("benchmarks", "paper", "data")
            ):
                raise ValueError(f"Private path in public archive: {member.name}")
            if member.isfile():
                content = public_content(
                    "/".join(parts), source.extractfile(member).read()
                )
                member.size = len(content)
                destination.addfile(member, io.BytesIO(content))
            else:
                destination.addfile(member)
    archive = result.getvalue()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        target.write(archive)
    print(f"Public source from committed HEAD: {args.output} ({len(archive)} bytes)")


if __name__ == "__main__":
    main()
