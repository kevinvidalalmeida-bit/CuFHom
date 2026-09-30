from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path


cp = pytest.importorskip("cupy")
CORE = Path(__file__).resolve().parents[1] / "FFT" / "ffthompy_core" / "ffthompy"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from ffthompy.tensors.fft import (
    irfftn_cached,
    irfftn_unscaled_cached,
    set_cupy_plan_mode,
    set_fft_backend,
)


def test_raw_cufft_c2r_matches_normalized_irfftn() -> None:
    """The raw C2R path must differ only by the known FFT normalization."""
    set_fft_backend("cupy")
    set_cupy_plan_mode("manual")
    shape = (3, 12, 10, 8)
    axes = (-3, -2, -1)
    field = cp.random.random(shape, dtype=cp.float32)
    spectrum = cp.fft.rfftn(field, axes=axes)
    normalized = irfftn_cached(spectrum, shape[-3:], axes)
    raw = irfftn_unscaled_cached(spectrum, shape[-3:], axes)
    relative = cp.linalg.norm(raw / np.prod(shape[-3:]) - normalized) / cp.linalg.norm(normalized)
    assert float(relative) < 2.0e-6
