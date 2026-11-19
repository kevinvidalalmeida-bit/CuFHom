# Reproducible homogenization benchmarks

This directory contains executable protocols and small fixed inputs, not archived
performance claims. It regenerates seven periodic microstructures at the requested
grid size and solves all six unit Mandel loads. No manuscript, stored solution
fields, model weights or previous campaign is required.

## Start with a CPU check

From the repository root at the chosen revision (the initial public version is
`v0.1.0`), install the library and run:

```bash
python -m pip install -e '.[test]'
python benchmarks/reproduce/run.py --dry-run --small
python benchmarks/reproduce/run.py --backend numpy --small --output results/smoke
python benchmarks/reproduce/summarize.py results/smoke
```

`--small` uses FP64 and 8³/9³ grids. It checks the installation and output contract;
it is not a performance benchmark. Use a new output directory for every run.
Failures are saved rather than discarded, and existing results are never replaced.

## Inputs and protocols

The [input definitions](inputs/README.md) include periodic spheres, a 40-grain
copper polycrystal, a gyroid, short fibers, inclined layers, a coated sphere and a
public two-phase image. Centers, orientations, nominal constitutive matrices and
sampling rules are fixed; no large voxel fields need to be committed.

| Protocol | Conditions |
| --- | --- |
| `public.json` | Public image, HEX8 192³/GaNi 189³, FP64/mixed, three sessions. |
| `six_cells.json`, `six_cells_fp32.json` | Six synthetic cells at three grids. |
| `fp32.json` | Explicitly rounded material with a nominal FP64 tensor reference. |
| `tolerance.json` | Tolerance attempts, including rejected targets. |
| `assembly.json` | Direct and corrected tensors from the same solutions. |
| `contrast.json` | Phase contrast from 10⁻³ to 10³, including the homogeneous case. |
| `reference.json` | Reference Poisson ratio crossed with material contrast. |
| `scaling.json` | 64³–512³, with actual storage residency recorded. |

Protocols are editable JSON in [protocols/](protocols/). The original public image
is attributed and licensed separately under CC BY 4.0 in
[inputs/README.md](inputs/README.md); CuFHom's MIT license does not replace it.

## GPU runs

Install CuPy for your CUDA environment separately. Start with the public protocol:

```bash
CUFHOM_FUSED_COMPILE=sync python benchmarks/reproduce/run.py \
  --protocol benchmarks/reproduce/protocols/public.json --output results/public
python benchmarks/reproduce/summarize.py results/public
```

Each condition/session runs in a separate process, sequentially. The protocol
specifies the grid, material, arithmetic, tolerance, reference and warm-call count.
The output records CPU/GPU identification, versions, source and input hashes,
affinity, cache settings, six update counts, fresh residuals, tensors, timings and
failure reasons. Warm calls solve all six loads again from zero. The runner does
not guarantee exclusive GPU access; reserve the device for timing comparisons.

The clock includes material-descriptor construction, route preparation, six solves,
native fresh acceptance, corrected assembly and GPU synchronization. It excludes
input sampling, imports, context initialization, reference selection and file
writing. Memory is sampled relative to the initialized device baseline, not a
certified process peak. Historical runs can have other timing boundaries.

`tables/points.csv` retains failed points. Reference-sensitivity runs also produce
absolute/normalized curves, sampled minima, the penalty at selected reference
values, and matched precision/reciprocal-contrast comparisons. Zero-work curves,
including homogeneous contrast, are excluded from normalization by their minimum.

## Component controls

```bash
python benchmarks/reproduce/ablations.py --pair hex_stencil --dry-run
CUFHOM_FUSED_COMPILE=sync python benchmarks/reproduce/ablations.py \
  --pair hex_stencil --precision fp64 --output results/stencil
```

Controls cover specialized versus general nodal actions, generated versus library
FFTs, Krylov/energy fusion, reference storage and slab scheduling. The runner
balances session ordering, warms each arm and records all per-load checks and
counters showing that the selected variant ran. Its timer prepares material and
reference outside the clock. Session ranges are descriptive, not confidence
intervals; variants that change several operations measure a combined effect.

The following controls complete the paired implementation study:

| Pair | Control / variant | Precisions |
| --- | --- | --- |
| `gani_loads` | Reused phase-indicator spectra / direct six-component load transforms | FP64, mixed, FP32 |
| `hex_evaluation` | df64 / IEEE FP64 fresh residual evaluation | Mixed |
| `gani_evaluation` | df64 / IEEE FP64 fresh residual evaluation | Mixed |

```bash
CUFHOM_FUSED_COMPILE=sync python benchmarks/reproduce/ablations.py \
  --pair gani_loads --precision mixed --n 21 --sessions 1 \
  --output results/pilot_loads
CUFHOM_FUSED_COMPILE=sync python benchmarks/reproduce/ablations.py \
  --pair hex_evaluation --precision mixed --n 24 --sessions 1 \
  --output results/pilot_evaluation
```

These small grids exercise the controls; they do not reproduce historical timings.
Omit `--n` to use the default 189³ GaNi or 192³ HEX8 grid for new measurements.
Evaluation pairs retain FP32 iterations and FP64 accumulation in both arms and
reject other precisions. Per-load metadata must confirm the selected arithmetic.
The load pair checks that all six production loads reuse phase spectra and all
six variant loads use direct transforms; it fails when the requested path is
not active. Summaries check matched input hashes and references, finite tensors,
and fresh physical acceptance even if an accepted flag was set incorrectly.
They retain all six update counts and residuals for every warm call, with paired
tensor differences. Count changes remain visible rather than being forced equal.

Host memory records now include RAM/swap capacity and availability, current RSS,
locked pages and the process-lifetime RSS high-water mark. Call records include
before/after snapshots. The high-water mark includes preparation and preceding
calls; it is not an isolated per-call peak or total host allocation. Missing Linux
procfs measurements are marked unavailable. Device-memory sampling is unchanged.

## Independent comparisons

FANS here means the PyTorch implementation in
[UNOCG](https://github.com/DataAnalyticsEngineering/UNOCG), not the standalone C++
[FANS](https://github.com/DataAnalyticsEngineering/FANS) executable. External
solvers are optional and are never replaced by the incorporated CuFHom core.
Pinned source revisions and requirements are in
[external_sources.json](protocols/external_sources.json).

```bash
python -m benchmarks.reproduce.external ffthompy --case public --grid 189 \
  --source external/FFTHomPy --precision fp64 --eta 1e-6 \
  --output results/ffthompy_session1.json
python -m benchmarks.reproduce.external fans --case public --grid 192 \
  --source external/UNOCG --eta 1e-6 --output results/fans_session1.json
```

Acquire the pinned source checkouts without downloading model weights:

```bash
git clone --filter=blob:none --sparse https://github.com/DataAnalyticsEngineering/UNOCG.git external/UNOCG
git -C external/UNOCG checkout 461f74f7a40f1c6c6d8ac053f6130428650a0133
git -C external/UNOCG sparse-checkout set unocg

git clone --filter=blob:none --sparse https://github.com/vondrejc/FFTHomPy.git external/FFTHomPy
git -C external/FFTHomPy checkout 2c23c80dd2cab46f1090103e613b4f886b3daac7
git -C external/FFTHomPy sparse-checkout set ffthompy
```

Install upstream dependencies in your benchmark environment. FANS requires CUDA-enabled PyTorch; no UNO weights are
needed. FFTHomPy comparisons require odd grids. The new adapters include material
preparation, solves, assembly and fresh physical auditing in their call clocks;
CPU/GPU hardware and native stopping norms still differ. Repeat commands in new
processes with new output names to measure independent sessions.

Read [numerical conventions](../../docs/NUMERICS.md) before interpreting tensor
errors, precision changes or accepted tolerances. Regenerating a benchmark is a new
measurement; the original wall times cannot be guaranteed on another system.
