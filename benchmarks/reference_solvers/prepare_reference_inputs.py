#!/usr/bin/env python3
"""Create identical voxel inputs for the external FFT solver timing runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
CASE = HERE / "sfrp_vf15_ar10_random_g48"
INPUTS = CASE / "reference_inputs"


def write_amitex_vtk(phase: np.ndarray, size: float, target: Path) -> None:
    """Write AMITEX's legacy VTK material map: 1 matrix, 2 fibre."""
    from vtk import vtkImageData, vtkStructuredPointsWriter
    from vtk.util.numpy_support import numpy_to_vtk

    n = np.array(phase.shape, dtype=int)
    image = vtkImageData()
    image.SetDimensions(*(n + 1))
    image.SetOrigin(0.0, 0.0, 0.0)
    image.SetSpacing(*([size / n[0]] * 3))
    material = np.ascontiguousarray((phase.astype(np.int32) + 1).ravel(order="F"))
    values = numpy_to_vtk(material, deep=True)
    values.SetName("MaterialId")
    image.GetCellData().SetScalars(values)
    writer = vtkStructuredPointsWriter()
    writer.SetFileName(str(target))
    writer.SetInputData(image)
    writer.SetFileTypeToBinary()
    if not writer.Write():
        raise RuntimeError(f"Could not write {target}")


def main() -> None:
    geometry = np.load(CASE / "geometry.npz")
    phase = geometry["phase"].astype(np.int32)
    size = 12.0
    INPUTS.mkdir(parents=True, exist_ok=True)

    import damask

    damask.GeomGrid(phase, size=[size, size, size], comments="SAM SFRP Vf15 AR10 seed 20260929").save(
        INPUTS / "sfrp_vf15_ar10_random_g48.vti"
    )
    write_amitex_vtk(phase, size, INPUTS / "sfrp_vf15_ar10_random_g48_amitex.vtk")

    amitex_root = HERE.parents[1] / "external" / "solver_references" / "AMITEX_FFTP"
    umat = amitex_root / "libAmitex" / "src" / "materiaux" / "libUmatAmitex.so"
    (INPUTS / "amitex_material.xml").write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Materials>
  <Reference_Material Lambda0="1.00e10" Mu0="1.00e10"/>
  <Material numM="1" Lib="{umat}" Law="elasiso">
    <Coeff Index="1" Type="Constant" Value="2.5925925926e9"/>
    <Coeff Index="2" Type="Constant" Value="1.1111111111e9"/>
  </Material>
  <Material numM="2" Lib="{umat}" Law="elasiso">
    <Coeff Index="1" Type="Constant" Value="2.2540983607e10"/>
    <Coeff Index="2" Type="Constant" Value="2.8688524590e10"/>
  </Material>
</Materials>
''', encoding="utf-8")
    (INPUTS / "amitex_loading.xml").write_text('''<?xml version="1.0" encoding="UTF-8"?>
<Loading_Output>
  <Output><vtk_StressStrain Strain="0" Stress="0"/></Output>
  <Loading Tag="1">
    <Time_Discretization Discretization="Linear" Nincr="1" Tfinal="1"/>
    <xx Driving="Strain" Evolution="Linear" Value="0.001"/>
    <yy Driving="Stress" Evolution="Linear" Value="0"/>
    <zz Driving="Stress" Evolution="Linear" Value="0"/>
    <xy Driving="Stress" Evolution="Linear" Value="0"/>
    <xz Driving="Stress" Evolution="Linear" Value="0"/>
    <yz Driving="Stress" Evolution="Linear" Value="0"/>
  </Loading>
</Loading_Output>
''', encoding="utf-8")

    algorithms = {
        "basic_accelerated_hexa": ("true", "Default"),
        "basic_unaccelerated_hexa": ("false", "Default"),
        "basic_accelerated_octa": ("true", "Octa"),
    }
    for name, (acceleration, filter_name) in algorithms.items():
        (INPUTS / f"amitex_{name}.xml").write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Algorithm_Parameters>
  <Algorithm Type="Basic_Scheme">
    <Convergence_Criterion Value="1.e-5"/>
    <Convergence_Acceleration Value="{acceleration}"/>
  </Algorithm>
  <Mechanics><Filter Type="{filter_name}"/><Small_Perturbations Value="true"/></Mechanics>
</Algorithm_Parameters>
''', encoding="utf-8")

    material = '''homogenization:
  SX:
    N_constituents: 1
    mechanical: {type: pass}
material:
  - homogenization: SX
    constituents: [{phase: matrix, v: 1.0, O: [1.0, 0.0, 0.0, 0.0]}]
  - homogenization: SX
    constituents: [{phase: fiber_proxy, v: 1.0, O: [1.0, 0.0, 0.0, 0.0]}]
phase:
  matrix:
    lattice: cF
    mechanical:
      elastic: {type: Hooke, C_11: 5.0e9, C_12: 2.0e9, C_44: 1.5e9}
      plastic: {type: none}
  fiber_proxy:
    lattice: cF
    mechanical:
      elastic: {type: Hooke, C_11: 1.30e11, C_12: 4.8e10, C_44: 3.8e10}
      plastic: {type: none}
'''
    (INPUTS / "damask_material.yaml").write_text(material, encoding="utf-8")
    numerics = '''solver:
  grid:
    mechanical:
      N_iter_min: 1
      N_iter_max: 100
      eps_abs_div(P): 1.0e-4
      eps_rel_div(P): 1.0e-5
      eps_abs_P: 1.0e3
      eps_rel_P: 1.0e-4
'''
    (INPUTS / "damask_numerics.yaml").write_text(numerics, encoding="utf-8")
    for solver in ("spectral_basic", "spectral_polarization", "spectral_Galerkin"):
        (INPUTS / f"damask_{solver}.yaml").write_text(f'''solver:
  mechanical: {solver}
loadstep:
  - boundary_condition:
      mechanical:
        F_dot: [[1.0e-3, 0, 0], [0, x, 0], [0, 0, x]]
        P: [[x, x, x], [x, 0, x], [x, x, 0]]
    discretization: {{t: 1, N: 1}}
    f_out: 1
''', encoding="utf-8")
    (INPUTS / "case_metadata.json").write_text(json.dumps({
        "case": "SFRP periodic straight fibres",
        "box": [size, size, size], "grid": list(map(int, phase.shape)),
        "fiber_volume_fraction_voxel": float(phase.mean()),
        "fiber_aspect_ratio": 10.0, "orientation": "isotropic random",
        "load": "epsilon_xx = 0.001, lateral average stress zero",
        "external_material_note": "DAMASK and AMITEX use isotropic elastic two-phase timing proxies.",
    }, indent=2) + "\n", encoding="utf-8")
    print(INPUTS)


if __name__ == "__main__":
    main()
