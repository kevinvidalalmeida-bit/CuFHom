"""Hashin bulk-modulus check: voxel section, GaNi/HEX8 convergence and timings.

Run from the repository root after installing .[gpu,examples]. The analytical
reference applies to hydrostatic bulk response, not the full stiffness tensor.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def hashin_bulk(core, coating, coating_shear, fraction):
    """Exact bulk modulus; fraction is the core volume/coated-sphere volume."""
    if min(core, coating, coating_shear) <= 0 or not 0 <= fraction <= 1:
        raise ValueError("Positive moduli and a core fraction in [0, 1] required")
    contrast = core - coating
    scale = coating + 4 * coating_shear / 3
    return coating + fraction * contrast * scale / (scale + (1 - fraction) * contrast)


def main():
    """Choose the study, run the shared benchmark runner and draw its results."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cufhom.fft.grids import GRID_POLICIES

    from benchmarks.paper.cases import INPUTS
    from benchmarks.paper.run import plot_convergence, run_convergence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision", choices=("mixed", "fp32", "fp64"), default="mixed"
    )
    parser.add_argument("--backend", choices=("cupy", "numpy"), default="cupy")
    parser.add_argument(
        "--grid-policy",
        choices=GRID_POLICIES,
        default="auto",
        help="auto chooses odd-13 for GaNi and even-13 for HEX8",
    )
    parser.add_argument(
        "--grids",
        type=int,
        nargs="+",
        default=(16, 24, 32, 48, 64, 96, 128, 192),
        help="Requested sizes; the policy selects sizes at least this large",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    moduli = json.loads((INPUTS / "materials.json").read_text())["hashin"]["moduli"]
    exact = hashin_bulk(
        moduli["K"][2], moduli["K"][1], moduli["G"][1], moduli["f_core"]
    )
    np.testing.assert_allclose(exact, moduli["K"][0], rtol=1e-13)
    try:
        study, section, output = run_convergence(
            "hashin", reference_bulk=exact, **vars(args)
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    plot_convergence(
        study,
        section,
        output / "hashin_convergence.png",
        title="Hashin coated sphere: hydrostatic analytical check",
        phase_labels=("Matrix", "Coating", "Core"),
    )
    print(f"Figure and raw results: {output.resolve()}")
    for method, check in study["mesh_check"].items():
        print(
            f"{method}: 0.1% analytic + last-mesh target {'met' if check['target_met'] else 'not met'}"
        )


if __name__ == "__main__":
    main()
