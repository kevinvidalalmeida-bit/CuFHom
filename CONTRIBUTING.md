# Contributing to CuFHom

Describe the problem and expected behavior before proposing large changes to
numerical formulations or input/output formats. Use GitHub issues for bug
reports and pull requests for focused changes.

## Environment and checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ruff check src core tests examples benchmarks
python -m ruff format --check src tests examples benchmarks
python -m pytest -m 'not gpu and not slow'
python -m build
```

Own code follows PEP 8 naming/formatting and PEP 257 public docstrings. Document
input shapes, units, tensor conventions and failure modes where needed. Ruff
checks the configured rules; it cannot verify the mathematical content of a
docstring. The incorporated core retains upstream formatting and notices.

Numerical changes should include behavior checks against analytic solutions or
independent discrete references when appropriate. Use small generated inputs;
tests must not depend on private campaigns. The [testing guide](docs/TESTING.md)
explains CPU selection and explicit serial CUDA validation.

## Reporting a numerical or performance issue

Include a minimal input or fixed public protocol, the exact command, software
revision, dependency versions and operating system. For CUDA, also include the
GPU model, driver and CuPy/CUDA versions. Report the route, grid, material and
working precision, tolerance, native residual metric, all six fresh checks and
any exception. Do not report only an averaged iteration count.

For timing comparisons, state what the clock includes, whether the process and
kernels were initialized, the first/warm call state, storage residency and other
device activity. Keep accepted and failed points. Generated benchmark records
capture this information where available; attach a small relevant record rather
than a large cache or solution dump.

## Attribution

Own code uses the [MIT license](LICENSE). Preserve third-party licenses and
notices, and document the source, license and any transformations of new inputs.
The public-image dataset has its own [attribution](benchmarks/reproduce/inputs/README.md).
Manuscripts and editorial assets are maintained outside this software repository.
