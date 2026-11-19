"""Stacked right-hand sides (sensitivity batches) must match one-by-one solves."""

import contextlib
import io

import numpy as np
import pytest
from cufhom.fft.linear.solver import cp, solve_homogenization


@pytest.mark.parametrize(
    "backend,dtype,tol",
    [
        ("scipy", "float64", 1e-12),
        pytest.param("cupy", "float64", 1e-12, marks=pytest.mark.gpu),
        pytest.param("cupy", "float32", 2e-5, marks=pytest.mark.gpu),
    ],
)
def test_batched_sensitivities_match_single_solves(backend, dtype, tol):
    if backend == "cupy" and cp is None:
        pytest.skip("CuPy unavailable")
    rng = np.random.default_rng(3)
    phase = (rng.random((8, 10, 12)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,))
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
        input_dir=".",
        phase_array=phase,
        ori_array=ori,
        Em=3.0,
        nu_m=0.3,
        Ef_L=90.0,
        Ef_T=15.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
        solver_rtol=1e-6,
        fft_backend=backend,
        solver_real_dtype=dtype,
        solution_sensitivity_return_in_memory=True,
    )
    fields = []
    for batch in (1, 2):
        p = dict(base, solution_sensitivity_batch_size=batch)
        with contextlib.redirect_stdout(io.StringIO()):
            solve_homogenization(p)
        fields.append(np.asarray(p["_solution_sensitivities_result"], dtype=np.float64))
    assert fields[0].shape == (7, 6, 6, 8, 10, 12)
    assert np.linalg.norm(fields[1] - fields[0]) <= tol * np.linalg.norm(fields[0])
