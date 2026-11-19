"""Shared test policy: CPU by default, explicit serial CUDA validation."""

import pytest


def pytest_addoption(parser):
    """Register explicit opt-in for tests that allocate CUDA resources."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable GPU tests; run them serially on an available CUDA device.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """Skip CUDA tests before their fixtures unless GPU execution was requested."""
    if item.get_closest_marker("gpu") is None:
        return
    if not item.config.getoption("--gpu"):
        pytest.skip("GPU tests require explicit --gpu opt-in")
    cupy = pytest.importorskip("cupy", reason="GPU tests require CuPy")
    try:
        devices = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    if devices == 0:
        pytest.skip("GPU tests require an available CUDA device")
