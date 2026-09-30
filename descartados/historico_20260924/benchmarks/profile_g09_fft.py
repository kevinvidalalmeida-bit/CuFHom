#!/usr/bin/env python3
"""Profile the optimized periodic FFT solver on the existing G09 N240 RVE.

The script is read-only with respect to G09: it loads ``phase.npy``/``ori.npy``
and the existing material JSON, then writes its timing report only below
``FFT/benchmarks/results``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
FFT_ROOT = HERE.parents[1]
FFTHOMPY_ROOT = FFT_ROOT / "ffthompy_core" / "ffthompy"
for path in (FFT_ROOT, FFTHOMPY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline.fft_solver import (  # noqa: E402
    _CUPY_AVAILABLE,
    compile_cfield_geometry,
    cp,
    solve_homogenization,
)


G09_N240 = ROOT / "results/cmame_method/interpretable_vf05_25_ar5_20/trend_studies_20260914/rasterized_g09/N240"
MATERIAL = ROOT / "results/cmame_method/interpretable_vf05_25_ar5_20/runs/adaptive_pair_20260914_200529_direct_g09_rep3/monitor_truth/material_0000/material.json"


@contextmanager
def quiet_solver() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def sync() -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


def optimized_parameters(
    phase: np.ndarray,
    orientation: np.ndarray,
    material: dict[str, Any],
    compiled_geometry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input_dir": str(G09_N240),
        "phase_array": phase,
        "ori_array": orientation,
        "Em": float(material["Em"]),
        "nu_m": float(material["nu_m"]),
        "Ef_L": float(material["Ef_L"]),
        "Ef_T": float(material["Ef_T"]),
        "nu_LT": float(material["nu_LT"]),
        "nu_TT": float(material["nu_TT"]),
        "G_LT": float(material["G_LT"]),
        "fft_backend": "cupy",
        "solver_real_dtype": "float32",
        "solver_rtol": 1.0e-5,
        "solver_atol": 0.0,
        "solver_maxiter": 1000,
        "require_convergence": True,
        "solver_verbose": False,
        "solver_fft_form": "r",
        "cfield_origin": "zero",
        "cfield_storage": "sym21",
        "cfield_indexed": True,
        "compiled_cfield_geometry": compiled_geometry,
        "projection_storage": "direct",
        "projection_backend": "numpy",
        "cupy_plan_mode": "manual",
        "cupy_preserve_plans": True,
        "cupy_fused_matvec": True,
        "cupy_unscaled_fft_pair": True,
        "cupy_lazy_scalars": True,
        "cupy_fused_cg_updates": True,
        "cupy_fused_xr_rr": True,
        "cupy_fused_dot": True,
        "cupy_reuse_inverse_input": True,
        "keep_solution_on_device": True,
        "internal_load_workers": 1,
        "load_batch_size": 1,
        "postprocess_batch_size": 1,
        "postprocess_assembly": "scalar",
        "force_disk_cfield": False,
    }


def run_variant(name: str, parameters: dict[str, Any], repeats: int) -> tuple[np.ndarray, dict[str, Any]]:
    config = {**parameters, "cupy_matrix_reference": name == "reference_metric"}
    with quiet_solver():
        solve_homogenization(config)  # warm-up allocator, plans and kernels
    sync()
    times: list[float] = []
    solution: np.ndarray | None = None
    for _ in range(repeats):
        sync()
        started = time.perf_counter()
        with quiet_solver():
            solution = solve_homogenization(config)
        sync()
        times.append(time.perf_counter() - started)
    assert solution is not None
    return solution, {
        "wall_s": [float(value) for value in times],
        "wall_s_median": float(np.median(times)),
        "wall_s_min": float(np.min(times)),
        "cupy_matrix_reference": bool(config["cupy_matrix_reference"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-metric", action="store_true")
    parser.add_argument("--formulation", choices=["strain", "displacement"], default="displacement",
                        help="default: displacement (integrated matrix-preconditioned PCG)")
    parser.add_argument("--output", type=Path, default=FFT_ROOT / "benchmarks/results/g09_n240_optimized_profile.json")
    args = parser.parse_args()
    if args.formulation == 'displacement' and args.reference_metric:
        parser.error('Use --formulation strain with --reference-metric, or omit --reference-metric')
    if not _CUPY_AVAILABLE:
        raise RuntimeError("G09 benchmark requires CuPy and an NVIDIA GPU.")
    if args.repeats < 1:
        raise ValueError("repeats must be positive.")
    phase = np.load(G09_N240 / "phase.npy")
    orientation = np.load(G09_N240 / "ori.npy")
    phase.setflags(write=False)
    orientation.setflags(write=False)
    material = json.loads(MATERIAL.read_text(encoding="utf-8"))
    compiled_geometry = compile_cfield_geometry(phase, orientation)
    parameters = optimized_parameters(phase, orientation, material, compiled_geometry)
    parameters['linear_formulation'] = args.formulation
    names = ["optimized"] + (["reference_metric"] if args.reference_metric else [])
    solutions: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {
        "geometry": str(G09_N240),
        "grid_shape": list(phase.shape),
        "fibre_volume_fraction": float(phase.mean()),
        "material": {key: material[key] for key in ("Em", "nu_m", "Ef_L", "Ef_T", "G_LT", "nu_LT", "nu_TT")},
        "repeats": int(args.repeats),
        "linear_formulation": args.formulation,
        "variants": {},
    }
    for name in names:
        solution, summary = run_variant(name, parameters, args.repeats)
        solutions[name] = solution
        results["variants"][name] = summary
    reference = solutions["optimized"]
    for name, solution in solutions.items():
        results["variants"][name]["relative_Ceff_difference_to_optimized"] = float(
            np.linalg.norm(solution - reference, ord="fro") / np.linalg.norm(reference, ord="fro")
        )
    if "reference_metric" in solutions:
        results["reference_metric_speedup"] = float(
            results["variants"]["optimized"]["wall_s_median"]
            / results["variants"]["reference_metric"]["wall_s_median"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
