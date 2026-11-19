# Numerical conventions and acceptance

CuFHom uses periodic small-strain elasticity on uniform Cartesian grids with
symmetric positive-definite constituent stiffnesses. The linear GaNi and HEX8
routes implement different discrete operators; sharing a voxel image or a
nominal tolerance does not make their discretization errors equal.

## Six loads and material representation

Mandel order is `(11, 22, 33, 23, 13, 12)`, with √2 scaling on shear strain and
stress. Six unit strains recover the effective stiffness in this same basis.
A packed material table stores its 21 upper-triangular entries row by row.
Crystal rotations preserve work in this orthonormal basis.

The synthetic benchmark definitions are sampled at voxel centers with periodic
inclusions/grains. The public image uses the original labels at 192³ and
`floor((i + 0.5) * 192 / n)` per axis after conversion to XYZ order. Resampling
can change phase fractions and interfaces; a larger array does not acquire new
geometric information. Material boundaries are perfectly bonded in the linear
model. Zero-stiffness phases are rejected by the positive-definiteness checks.

## Fresh equilibrium checks

The recursively updated PCG residual is inexpensive but can drift from the
residual of the stored solution through rounding. It triggers inspections; it
does not alone establish acceptance. A fresh evaluation applies the physical
discrete operator again to that solution.

On CUDA, HEX8 accepts the maximum nodal-force residual and GaNi accepts the RMS
projected stress. Each is compared with `max(atol, rtol * initial_measure)`.
All six loads must pass. A homogeneous zero-fluctuation problem can require
zero updates and have no defined relative residual; records preserve that as
`null`, not an artificial success ratio. Failed curvature, unsafe steps,
stagnation and iteration budgets remain explicit failures.

Native fresh checks reuse production kernels. They do not constitute an
independently implemented operator audit. The small-grid verification tests
compare against separately assembled/reference systems; they do not establish
experimental validity or continuum accuracy for every benchmark.

## Precision

- FP64: FP64 problem coefficients, iteration and solution.
- Mixed: nominal FP64 problem, FP32 iteration with FP64 solution accumulation
  and fresh evaluations in df64 or IEEE FP64.
- FP32: explicitly rounded coefficients and FP32 solution/iteration, with
  higher-precision fresh evaluation of the rounded problem.

Double-float `df64` uses a compensated pair of FP32 numbers and is not IEEE
FP64. Metadata identifies the actual evaluation route. Fresh residual values
and acceptance decisions are stored before display rounding.

## Tensor errors and assembly

A relative Frobenius difference uses the stated numerical reference tensor in
the denominator. It measures algebraic error for the same grid, formulation and
coefficients. A nominal-FP64 reference for an FP32-defined material also includes
constitutive-rounding effects. Neither quantity measures error against the
continuum or a physical experiment.

Direct and residual-corrected assembly use the same six solutions. The benchmark
clock includes corrected assembly; a direct tensor obtained from that call does
not have an independently measured direct-assembly time. Comparing their errors
can show the accuracy effect, but cannot establish isolated assembly speed-ups.
Near-roundoff differences are sensitive to the numerical reference.

## Reference and execution controls

The default reference uses the unweighted constitutive-table mean; GaNi takes
its isotropic projection. In contrast/reference protocols, the reference shear
modulus is the mean of the two phase shear moduli and stays fixed while reference
Poisson ratio varies at a given contrast. Changing a preconditioner preserves
the discrete equilibrium problem but can change iteration counts.

A cold process/cache, a cached first call and warm repeated calls are distinct.
Inspect the recorded boundary and initialization state before taking time ratios.
Changes in GPU/host residency can add transfers and affect both runtime and
sampled device memory. Per-session ratios, iteration counts and fresh checks
must be retained when interpreting component controls.

The executable definitions are in
[`benchmarks/reproduce`](../benchmarks/reproduce/README.md); production acceptance
and precision logic is in
[`framework.py`](../src/fft/linear/framework.py),
[`common.py`](../src/fft/linear/common.py),
[`coordinates.py`](../src/fft/linear/coordinates.py) and
[`hex8_fused.py`](../src/fft/linear/hex8_fused.py).
