"""Solve six unit Mandel loads for a small two-phase cell on CPU.

Run from an installed CuFHom environment with
``python examples/linear_elasticity.py``. This is a usage example, not a timing
benchmark or a comparison of continuum discretization errors.
"""

import numpy as np
from cufhom.fft import IndexedMaterial, homogenize


def main():
    """Build an inclusion, solve both routes and display their fresh checks."""
    labels = np.zeros((12, 12, 12), dtype=np.int32)
    labels[3:9, 3:9, 3:9] = 1
    matrices = []
    for young, poisson in ((3.0, 0.3), (30.0, 0.3)):
        shear = young / (2 * (1 + poisson))
        lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        matrix = 2 * shear * np.eye(6)
        matrix[:3, :3] += lame
        matrices.append(matrix)
    # Table columns follow the upper triangle row by row, in Mandel coordinates.
    table = np.stack([matrix[np.triu_indices(6)] for matrix in matrices])
    material = IndexedMaterial(labels, table, dtype="float64")

    for route in ("gani", "hex8"):
        result = homogenize(
            material,
            discretization=route,
            backend="numpy",
            working="float64",
            rtol=1e-6,
        )
        audits = result["audits"]
        if len(audits) != 6 or not all(audit["accepted"] for audit in audits):
            raise RuntimeError(f"{route}: all six loads must pass fresh acceptance")
        relative = [a["relative"] for a in audits if a["relative"] is not None]
        print(f"{route.upper()}: six accepted loads; updates {result['iterations']}")
        print(f"Fresh metric: {audits[0]['metric']}; max relative {max(relative):.2e}")
        with np.printoptions(precision=5, suppress=True):
            print("Effective Mandel stiffness (same stress units as E):")
            print(result["C"])
        print()


if __name__ == "__main__":
    main()
