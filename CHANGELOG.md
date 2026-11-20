# Changelog

## Unreleased

- `benchmarks/reproduce/section41.py`: resumable driver for the six-cell comparison
  of Section 4.1 (CuFHom, FANS, FFTHomPy) with environment check, pinned source
  fetch, kernel-cache priming and a Table 1-style summary.
- Installation notes for the paper's CuPy/PyTorch pins on Pascal-to-Blackwell
  GPUs and for small-memory cards; quick start now points to the `v0.1.1` tag.

## 0.1.0 — 2026-11-17

Initial public research-software release.

- Common indexed-material API for GaNi and HEX8, with specialized CUDA execution,
  mixed precision, fresh acceptance checks and corrected stiffness assembly.
- Fixed definitions for six synthetic cells and a licensed public image, with
  executable numerical protocols and optional pinned external comparators.
- CPU examples, numerical conventions, PEP 257 docstrings and CPU/GPU test policy.
- Installable wheel and source distribution; CI tests Python 3.11–3.13 on CPU
  and checks the installed package outside its source tree.

GPU validation requires a compatible CUDA environment and explicit opt-in.
Benchmark timings depend on the recorded hardware, arithmetic and clock boundary.
