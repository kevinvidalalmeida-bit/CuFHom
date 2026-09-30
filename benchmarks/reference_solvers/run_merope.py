#!/usr/bin/env python3
"""Time Mérope voxelization of the exact continuous SAM fibre rows."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import merope


HERE = Path(__file__).resolve().parent
CASE = HERE / "sfrp_vf15_ar10_random_g48"
OUT = CASE / "results" / "merope"


def cylinder(pt1, pt2, radius):
    axis = merope.geometry.Segment_3D([pt1, pt2])
    return merope.microInclusion.CylinderInclusions(merope.geometry.Cylinder(axis, radius))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fibres = np.loadtxt(CASE / "continuous_fibers_master.csv", delimiter=",", skiprows=1)
    inclusions = []
    for row in np.atleast_2d(fibres):
        center, direction, length, diameter = row[:3], row[3:6], row[6], row[7]
        inclusions.append(cylinder(center - 0.5 * length * direction, center + 0.5 * length * direction, 0.5 * diameter))
    cylinder_inclusions = merope.CylinderInclusions_3D()
    cylinder_inclusions.setLength([12.0, 12.0, 12.0])
    cylinder_inclusions.setInclusions(inclusions)
    multi = merope.MultiInclusions_3D()
    multi.setInclusions(cylinder_inclusions)
    identifiers = multi.getAllIdentifiers()
    multi.changePhase(identifiers, [1 for _ in identifiers])
    multi.setMatrixPhase(0)
    structure = merope.Structure_3D(multi)
    parameters = merope.vox.create_grid_parameters_N_L_3D([48, 48, 48], [12.0, 12.0, 12.0])
    results = {"n_fibres": int(len(inclusions)), "variants": {}}
    for name, rule in (("center", merope.vox.VoxelRule.Center), ("average_reuss", merope.vox.VoxelRule.Average)):
        started = time.perf_counter()
        grid = merope.vox.GridRepresentation_3D(structure, parameters, rule)
        if name == "average_reuss":
            grid.apply_homogRule(merope.HomogenizationRule.Reuss, [1, 2])
        elapsed = time.perf_counter() - started
        output = OUT / f"{name}.vtk"
        merope.vox.vtk_printer_3D().printVTK(grid, str(output))
        results["variants"][name] = {"voxelization_wall_s": elapsed, "output": str(output)}
        print(f"Merope {name}: {elapsed:.3f} s")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
