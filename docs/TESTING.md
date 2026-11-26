# Testing

Install the package and development tools from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The default run executes CPU tests and skips tests marked `gpu`, even when a
CUDA device is present. It includes numerical reference checks, invalid-input
handling, transactional output, geometry generation, elastic materials, and solver
acceptance and restart contracts. Tests generate temporary files under pytest's
`tmp_path`; archived paper campaigns and large datasets are not required.

The public precision tests compare `precision="fp32"`, `"fp64"` and `"mixed"`
against the established storage/Krylov configurations, check conflicting options,
and cover automatic CPU/CUDA selection. The geometry command tests preserve
advanced options while checking the shorter `--help` and complete `--help-all`.

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

With an NVIDIA device, a compatible driver and CUDA Toolkit 12.x installed,
install the tested CuPy series through the optional GPU extra:

```bash
python -m pip install -e '.[gpu]'
```

Other CUDA installations need the corresponding CuPy distribution; see the
[CuPy installation guide](https://docs.cupy.dev/en/v13.6.0/install.html).
For NVIDIA libraries installed as Python wheels, activate the environment and
run `source tools/env_gpu.sh` if those libraries need to be added to the loader
path. Run GPU tests **serially** on an available device:

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
must not be interpreted as reproducible paper benchmarks. Large-grid campaigns
are kept under `benchmarks/paper/` and are not run in routine CI.

## Style and packaging

```bash
python -m ruff check src core tests examples benchmarks tools
python -m ruff format --check src tests examples benchmarks tools
python -m build
```

CI additionally installs the built wheel in a separate environment and checks
imports and command-line help outside the checkout. Docstring checks apply to
library code; test names and assertions document test behavior without requiring
redundant docstrings for every test function.
