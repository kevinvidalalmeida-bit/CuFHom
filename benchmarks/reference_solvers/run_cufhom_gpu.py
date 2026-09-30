#!/usr/bin/env python3
"""Paired GPU timing of the shared SFRP case: legacy and optimized CuFHom cores."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import time

import numpy as np

from cufhom.pipeline.fft_solver import _CUPY_AVAILABLE, compile_cfield_geometry, cp, solve_homogenization


HERE = Path(__file__).resolve().parent
CASE = HERE / "sfrp_vf15_ar10_random_g48"
OUT = CASE / "results" / "cufhom_gpu"


def sync() -> None:
    cp.cuda.Stream.null.synchronize()


def solve_quiet(parameters: dict) -> np.ndarray:
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
        return np.asarray(solve_homogenization(parameters))


def main() -> None:
    if not _CUPY_AVAILABLE:
        raise RuntimeError("CuPy/GPU is required for this benchmark")
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CASE / "geometry.npz")
    phase, ori = data["phase"], data["ori"]
    phase.setflags(write=False)
    ori.setflags(write=False)
    compiled = compile_cfield_geometry(phase, ori)
    base = {
        "input_dir": str(CASE),
        "phase_array": phase, "ori_array": ori,
        "Em": 3.0, "nu_m": 0.35,
        "Ef_L": 70.0, "Ef_T": 12.0, "nu_LT": 0.22, "nu_TT": 0.25, "G_LT": 8.0,
        "fft_backend": "cupy", "solver_real_dtype": "float32",
        "solver_rtol": 1.0e-5, "solver_atol": 0.0, "solver_maxiter": 1000,
        "require_convergence": True, "solver_verbose": False,
        "active_load_ids": [0], "partial_load_output": True,
        "load_batch_size": 1, "postprocess_batch_size": 1,
        "force_disk_cfield": False, "gpu_timing_sync": True,
    }
    variants = {
        "legacy_dense_complex": {
            "solver_fft_form": "c", "cfield_storage": "full", "cfield_indexed": False,
            "projection_storage": "full", "projection_backend": "numpy",
            "cupy_plan_mode": "none", "cupy_preserve_plans": False,
            "cupy_fused_matvec": False, "cupy_unscaled_fft_pair": False,
            "cupy_lazy_scalars": False, "cupy_fused_cg_updates": False,
            "cupy_fused_xr_rr": False, "cupy_fused_dot": False,
            "keep_solution_on_device": False,
        },
        "optimized_indexed_rfft": {
            "solver_fft_form": "r", "cfield_storage": "sym21", "cfield_indexed": True,
            "compiled_cfield_geometry": compiled, "projection_storage": "direct", "projection_backend": "numpy",
            "cupy_plan_mode": "manual", "cupy_preserve_plans": True,
            "cupy_fused_matvec": True, "cupy_unscaled_fft_pair": True,
            "cupy_lazy_scalars": True, "cupy_fused_cg_updates": True,
            "cupy_fused_xr_rr": True, "cupy_fused_dot": True,
            "keep_solution_on_device": True,
        },
    }
    results = {"gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "grid": list(phase.shape),
               "vf": float(phase.mean()), "load": "epsilon_xx = 0.001", "variants": {}}
    solutions = {}
    for name, settings in variants.items():
        config = {**base, **settings}
        solve_quiet(config)  # excludes one-time module compilation and FFT plan creation
        sync()
        timing = OUT / f"{name}_timing.json"
        samples = []
        solution = None
        for _ in range(3):
            started = time.perf_counter()
            solution = solve_quiet({**config, "solver_timing_path": str(timing)})
            sync()
            samples.append(time.perf_counter() - started)
        assert solution is not None
        wall = float(np.median(samples))
        detail = json.loads(timing.read_text(encoding="utf-8"))
        detail["outer_wall_s"] = wall
        detail["wall_s_samples"] = samples
        detail["wall_s_median"] = wall
        detail["C_xxxx_GPa"] = float(solution[0, 0])
        (OUT / f"{name}.json").write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        results["variants"][name] = {"wall_s_median": wall, "wall_s_samples": samples,
                                     "C_xxxx_GPa": detail["C_xxxx_GPa"], "CG_s": detail["problem_calculate_s"]}
        solutions[name] = solution
        print(f"{name}: {wall:.3f} s")
    legacy = results["variants"]["legacy_dense_complex"]["wall_s_median"]
    optimized = results["variants"]["optimized_indexed_rfft"]["wall_s_median"]
    results["optimized_speedup"] = legacy / optimized
    results["relative_C_difference"] = float(np.linalg.norm(solutions["legacy_dense_complex"] - solutions["optimized_indexed_rfft"])
                                             / np.linalg.norm(solutions["legacy_dense_complex"]))
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
