# Testing

Install the package and development tools from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

For a quick installation check before running the suite:

```bash
python examples/linear_elasticity.py
```

This example solves a small heterogeneous cell with both linear routes and
checks six fresh acceptance decisions per route. `examples/smoke_cpu.py`
additionally checks homogeneous analytic linear and nonlinear elastic limits.

The default run executes CPU tests and skips tests marked `gpu`, even when a
CUDA device is present. It includes numerical reference checks, invalid-input
handling, transactional output, geometry generation, material laws, and solver
acceptance and restart contracts. Tests generate temporary files under pytest's
`tmp_path`; archived paper campaigns and large datasets are not required.

To select the CPU suite explicitly, as in CI:

```bash
python -m pytest -m 'not gpu and not slow'
```

The suite contains both pytest functions and `unittest.TestCase` classes. Pytest
collects both. Tests marked `integration` exercise subprocess imports or CLI
entry points; they need the package installed in the Python interpreter used to
run pytest. Operator diagnostics used by the tests reside in `tests/fft/linear/`; their
host-only contracts are exercised in the CPU suite.

## CUDA validation

Install the CuPy distribution matching the local CUDA runtime separately, then
run GPU tests **serially** on an available device:

```bash
python -m pytest --gpu -m gpu
```

`--gpu` enables execution; the `gpu` marker selects the cases. Missing CuPy,
missing CUDA drivers, and zero available devices produce explicit skips. CPU
parameter variants remain in the CPU suite; GPU variants have their own marker.
Tests compare GPU operations and freshly evaluated residuals against CPU,
analytic, or independently assembled discrete references where appropriate.

The operator probe also requires its existing explicit resource opt-in:

```bash
CUFHOM_P2_GPU_TESTS=1 python -m pytest --gpu -m gpu \
    tests/fft/linear/test_fft_operator_probe.py
```

This is a validation suite, not a performance campaign. GPU timings from tests
must not be interpreted as reproducible paper benchmarks. Large-grid benchmarks are not run in routine CI.

## Style and packaging

```bash
python -m ruff check src core tests examples benchmarks
python -m ruff format --check src tests examples benchmarks
python -m build
```

CI additionally installs the built wheel in a separate environment and checks
imports, command-line help and both CPU examples outside the package source
tree. It runs the CPU suite against that installation using tests extracted
from the source distribution, and checks the small benchmark protocol plan.
Docstring checks apply to
library code; test names and assertions document test behavior without requiring
redundant docstrings for every test function.
