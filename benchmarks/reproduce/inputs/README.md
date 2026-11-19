# Fixed benchmark inputs

These are inputs, not archived simulation results. Together with `../cases.py`
they regenerate the six comparison cells and the public image at any requested
cubic grid. No stored 448³/512³ geometry, solver checkpoint or external cache is
required. The input files total less than 0.3 MB before the checksum report.

| Case | Preserved definition | Sampling and material |
| --- | --- | --- |
| `particles` | 24 fixed centres and radii; continuous volume fraction 0.20 | Periodic spheres, strict interior; epoxy/glass |
| `polycrystal` | 40 fixed Voronoi seeds and orientation quaternions | Periodic nearest seed; rotated cubic copper |
| `gyroid` | One period, zero threshold | Positive gyroid level set; epoxy/glass |
| `random_short_fibres` | 230 fixed centres/directions, fraction 0.10, aspect ratio 15 | Periodic flat-ended cylinders; epoxy/glass |
| `laminate` | Integer normal (1,2,3), offset 0.137, fraction 0.40 | Periodic inclined layers; epoxy/glass |
| `hashin` | Outer radius 0.40, core/coating-volume ratio 0.14 | Matrix/coating/core, bulk contrast 10, Poisson ratio 0.25 |
| `public` | Original binary 192³ phase image, losslessly compressed | Original material (E,ν)=(75,0.30)/(400,0.20) |

`periodic_families.json` retains the original 20261016 generator seed and the
actual random draws; rerunning a random-number generator is unnecessary.
`random_short_fibres.json` preserves the actual cylinder placement, its original
2026110601 seed and placement audit.

`materials.json` preserves the nominal FP64 packed stiffness entries exactly,
including the 40 anisotropic copper tensors. Mandel order is
(11,22,33,23,13,12); packing follows the upper triangle row by row. A requested
FP32 table is a cast of this same FP64 table. The reference E/ν recorded there
is historical auxiliary information: the benchmark driver states its actual
preconditioner construction explicitly, and a tight tensor-error reference is
a separate numerical solve.

The six continuous cells use voxel-centre sampling in a periodic unit cube.
The public source image has the original dataset axis order; `cases.py`
transposes it to XYZ and samples source index `floor((i+0.5)*192/n)` on each
axis. Refinement of this image resamples existing information. It does not
create finer measured geometry.

## Public image source and attribution

`public_phase192.npz` contains the unchanged `test/dset_1/image` array (the
second test entry) from data file 478873 of Julius Herb and Felix Fritzen,
*Supplemental data for “Accelerating Conjugate Gradient Solvers for
Homogenization Problems with Unitary Neural Operators”*, DaRUS,
[doi:10.18419/DARUS-5686](https://doi.org/10.18419/DARUS-5686).
The dataset metadata identifies its license as
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); this external image
is covered by that license, rather than the repository's code MIT license.
The upstream dataset credits the 3-D microstructures to Prifling, Röding,
Townsend, Neumann and Schmidt (2020),
[doi:10.5281/zenodo.4047774](https://doi.org/10.5281/zenodo.4047774).
Compression is lossless; no phase values were changed. The XYZ transposition
and resampling occur only when preparing a case.

`public_source.json` retains the original extraction metadata and raw-array
SHA-256. `public_license.json` records the upstream dataset API license.
`geometry_checks.json` records regenerated input-array checksums at 21³,
189³ and 192³, with the original generation functions or recorded input hashes
used for comparison. The public 189³ index hash is
`88b774b6081cf67bdd9529b875b72b964ae63c82898a97bf28c9e46cfccd31d0`.
