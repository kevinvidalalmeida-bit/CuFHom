# Using CuFHom

The main linear API is `cufhom.fft.homogenize`. It takes an integer material image
and a stiffness table, solves six unit Mandel strain loads, and returns a tensor
and per-load acceptance records. Start with
[`examples/linear_elasticity.py`](../examples/linear_elasticity.py) for a complete
two-phase CPU example.

## Linear elasticity with indexed materials

Pack each symmetric 6 × 6 material matrix using the upper triangle, row by row.
For an isotropic homogeneous cell:

```python
import numpy as np
from cufhom.fft import IndexedMaterial, homogenize

E, nu = 3.0, 0.3
mu = E / (2 * (1 + nu))
lame = E * nu / ((1 + nu) * (1 - 2 * nu))
C = 2 * mu * np.eye(6)
C[:3, :3] += lame
labels = np.zeros((4, 4, 4), dtype=np.int32)
table = C[np.triu_indices(6)][None, :]
result = homogenize(
    IndexedMaterial(labels, table, dtype="float64"),
    discretization="hex8", backend="numpy", rtol=1e-8,
    working="float64",
)
np.testing.assert_allclose(result["C"], C, atol=1e-10, rtol=1e-10)
print(result["iterations"])
```

Mandel order is `(11, 22, 33, 23, 13, 12)` with √2 scaling on shear strain and
stress. Stiffness units follow the input moduli; these values can be interpreted
in GPa. Each label indexes a row of the material table. The material matrices
must be symmetric positive definite. The default cell lengths are `(1, 1, 1)`;
set `lengths=(Lx, Ly, Lz)` explicitly for a different periodic cell.

`discretization="gani"` selects Fourier–Galerkin with numerical integration;
`"hex8"` selects fully integrated trilinear hexahedral elements.
`result["C"]` is the residual-corrected effective stiffness,
`result["iterations"]` contains six update counts, and `result["audits"]`
contains the native metric, measured residual, threshold and acceptance for each
load. An unattainable target raises an error rather than returning a converged
status. The campaign runner saves such failures as records.

### CPU and CUDA precision

| Backend | Material dtype | `working` | Configuration |
| --- | --- | --- | --- |
| `numpy` | `float64` | `float64` | CPU FP64 example/reference path |
| `cupy` | `float64` | `float64` | GPU FP64 |
| `cupy` | `float64` | `float32` | GPU mixed: FP32 Krylov work, FP64 solution accumulation |
| `cupy` | `float32` | `float32` | GPU FP32-defined problem |

CuPy/CUDA is optional and installed separately. Use the CuPy distribution matching
your CUDA environment, then change `backend` to `cupy`. Specify `working`
explicitly when comparing precisions: an FP64 material on CUDA defaults to mixed
precision. For mixed and FP32 CUDA configurations, `evaluation="df64"` uses
compensated pairs of FP32 numbers for fresh checks; `evaluation="float64"`
requests IEEE FP64 evaluation. FP64 iteration uses FP64 evaluation.

The two routes use different residual norms: HEX8 uses nodal-force Linf and GaNi
projected-stress RMS. A shared tolerance does not establish equal continuum
accuracy. Fresh checks reuse native operators; they are not independent operator
implementations. Read [numerical conventions](NUMERICS.md) for residual
normalization, coefficient rounding, tensor references and memory policy.

## Fiber geometry generation

```bash
cufhom-generate --output results/geometry --shape cylinder \
  --vf .08 --ar 4 --diameter 1 --box 12 --grid 72 \
  --solver a4 --a2 .6 .3 --threads 2
```

Use consistent length units. `grid` is the voxel count per axis, `vf` the target
volume fraction, and `ar` the length/diameter ratio. Orientation targets use A2
or `--orientation-json`; `cufhom-generate --help` lists the options. The
[geometry illustration](../examples/plot_geometry_example.py) provides an
executable Python API example; install `.[plots]` to render it.

Generation and voxelization have separate acceptance checks. The output contains
configuration, acceptance summaries and geometry when accepted. Orientation or
penetration tolerances are not equilibrium tolerances. The fixed cells for the
linear studies are in [benchmark inputs](../benchmarks/reproduce/inputs/README.md)
rather than in the fiber-generator example.

## Additional nonlinear models

```bash
cufhom-nonlinear --help
python examples/smoke_cpu.py
```

`cufhom.fft.solve_nonlinear_homogenization` accepts material parameters and load
histories. Example JSON files in [examples/](../examples/) cover J2 plasticity,
Drucker–Prager, paraboloidal plasticity and damage. Material and loading units
must be consistent. The smoke example checks homogeneous linear and nonlinear
elastic limits on CPU; the linear benchmark protocols do not establish nonlinear
performance or validity for an arbitrary history.

## API documentation

Public classes and functions contain docstrings describing their purpose,
parameters, results and restrictions. They are available directly in Python:

```python
from cufhom.fft import IndexedMaterial, homogenize

help(IndexedMaterial)
help(homogenize)
```

The [entry-point source](../src/fft/linear/framework.py) documents grid,
precision, evaluation and verification options. Docstrings follow
[PEP 257](https://peps.python.org/pep-0257/); Ruff checks the configured rules.
This improves consistency and interactive help, but does not alter the numerical
method or generate a documentation website.
