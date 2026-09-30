#!/usr/bin/env python3
"""Reproducible kernel-level benchmark for the CuFHom/CuPy FFT core.

It uses a synthetic periodic cylindrical inclusion, not a campaign geometry.
The benchmark reports both end-to-end timings and synchronized stage timings;
the latter are diagnostic only because synchronization intentionally adds
profiling overhead.
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
FFT_ROOT = HERE.parents[1]

from cufhom.pipeline.fft_solver import _CUPY_AVAILABLE, cp, solve_homogenization  # noqa: E402


def periodic_cylinder(grid: int, radius_fraction: float = 0.22) -> tuple[np.ndarray, np.ndarray]:
    """Return a periodic x-aligned fibre and its unit orientation field."""
    coordinate = np.arange(grid, dtype=np.float64) - 0.5 * grid
    _, yy, zz = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    radius = float(grid) * radius_fraction
    phase = (yy * yy + zz * zz <= radius * radius).astype(np.uint8)
    orientation = np.zeros((*phase.shape, 3), dtype=np.float32)
    orientation[..., 0] = 1.0
    return phase, orientation


def base_parameters(phase: np.ndarray, orientation: np.ndarray) -> dict[str, Any]:
    return {
        "input_dir": str(FFT_ROOT),
        "seed": 0,
        "phase_array": phase,
        "ori_array": orientation,
        "Em": 3.0,
        "nu_m": 0.30,
        "Ef_L": 150.0,
        "Ef_T": 12.0,
        "nu_LT": 0.25,
        "nu_TT": 0.30,
        "G_LT": 7.0,
        "fft_backend": "cupy",
        "solver_real_dtype": "float32",
        "solver_rtol": 1.0e-5,
        "solver_atol": 0.0,
        "solver_maxiter": 1000,
        "require_convergence": True,
        "solver_verbose": False,
        "load_batch_size": 1,
        "force_disk_cfield": False,
        "gpu_timing_sync": False,
        "profile_timing": False,
        "profile_cg_timing": False,
    }


VARIANTS = {
    "legacy": {
        "solver_fft_form": "c",
        "cfield_storage": "full",
        "cfield_indexed": False,
        "projection_storage": "full",
        "projection_backend": "numpy",
        "cupy_plan_mode": "none",
        "cupy_preserve_plans": False,
        "cupy_fused_matvec": False,
        "cupy_unscaled_fft_pair": False,
        "cupy_lazy_scalars": False,
        "cupy_fused_cg_updates": False,
        "cupy_fused_xr_rr": False,
        "cupy_fused_dot": False,
    },
    "optimized": {
        "solver_fft_form": "r",
        "cfield_storage": "sym21",
        "cfield_indexed": True,
        "projection_storage": "direct",
        "projection_backend": "numpy",
        "cupy_plan_mode": "manual",
        "cupy_preserve_plans": True,
        "cupy_fused_matvec": True,
        "cupy_unscaled_fft_pair": True,
        "cupy_lazy_scalars": True,
        "cupy_fused_cg_updates": True,
        "cupy_fused_xr_rr": False,
        "cupy_fused_dot": True,
    },
    "optimized_batch2": {"load_batch_size": 2},
    "optimized_batch3": {"load_batch_size": 3},
    "optimized_batch6": {"load_batch_size": 6},
    "optimized_no_fused_matvec": {"cupy_fused_matvec": False},
    "optimized_no_unscaled_fft": {"cupy_unscaled_fft_pair": False},
    "optimized_no_fused_updates": {
        "cupy_fused_cg_updates": False,
        "cupy_fused_dot": False,
    },
    # The current indexed material kernel intentionally accepts real CuPy
    # fields only; complex FFT form therefore uses the dense sym21 path.
    "optimized_complex_fft": {
        "solver_fft_form": "c",
        "cfield_indexed": False,
    },
    "optimized_full_cfield": {
        "cfield_storage": "full",
        "cfield_indexed": False,
    },
    "optimized_full_projection": {"projection_storage": "full"},
    "optimized_no_plan_cache": {
        "cupy_plan_mode": "none",
        "cupy_preserve_plans": False,
    },
}


def synchronize() -> None:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()


@contextmanager
def quiet_solver() -> Any:
    """Keep benchmark output machine-readable without changing solver timing."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


def run_variant(
    name: str,
    parameters: dict[str, Any],
    repeats: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    variant = VARIANTS[name]
    # Ablations inherit the production optimized core settings.
    config = {**parameters, **(VARIANTS["optimized"] if name.startswith("optimized_") else {}), **variant}
    # Warm kernels, FFT plans and the allocator. This solve is not measured.
    with quiet_solver():
        solve_homogenization(config)
    synchronize()
    wall_times: list[float] = []
    solution = None
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        with quiet_solver():
            solution = solve_homogenization(config)
        synchronize()
        wall_times.append(time.perf_counter() - started)

    timing: dict[str, Any] | None = None
    profile_path: Path | None = None
    if int(config["load_batch_size"]) == 1:
        profile_path = output_dir / f"{name}_stage_timing.json"
        profiled_config = {
            **config,
            "profile_timing": True,
            "profile_cg_timing": True,
            "solver_timing_path": str(profile_path),
        }
        with quiet_solver():
            solve_homogenization(profiled_config)
        synchronize()
        timing = json.loads(profile_path.read_text(encoding="utf-8"))
    return np.asarray(solution), {
        "config": {key: config[key] for key in sorted(variant)},
        "wall_s": [float(value) for value in wall_times],
        "wall_s_min": float(np.min(wall_times)),
        "wall_s_median": float(np.median(wall_times)),
        "timing_file": str(profile_path) if profile_path is not None else None,
        "stage_timing": timing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", default="legacy,optimized")
    parser.add_argument("--output-dir", type=Path, default=FFT_ROOT / "benchmarks" / "results")
    args = parser.parse_args()
    if not _CUPY_AVAILABLE:
        raise RuntimeError("This benchmark requires CuPy and an NVIDIA GPU.")
    if args.grid < 8 or args.repeats < 1:
        raise ValueError("grid must be >= 8 and repeats must be >= 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase, orientation = periodic_cylinder(args.grid)
    parameters = base_parameters(phase, orientation)
    results: dict[str, Any] = {
        "grid": int(args.grid),
        "voxel_count": int(phase.size),
        "fibre_volume_fraction": float(phase.mean()),
        "repeats": int(args.repeats),
        "variants": {},
    }
    solutions: dict[str, np.ndarray] = {}
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    if not variant_names or any(name not in VARIANTS for name in variant_names):
        raise ValueError(f"variants must be selected from: {', '.join(VARIANTS)}")
    for name in variant_names:
        solution, summary = run_variant(name, parameters, args.repeats, args.output_dir)
        solutions[name] = solution
        results["variants"][name] = summary
    reference_name = variant_names[0]
    reference_solution = solutions[reference_name]
    for name, solution in solutions.items():
        results["variants"][name]["relative_solution_difference_to_" + reference_name] = float(
            np.linalg.norm(solution - reference_solution, ord="fro")
            / np.linalg.norm(reference_solution, ord="fro")
        )
    if "legacy" in solutions and "optimized" in solutions:
        results["relative_solution_difference"] = float(
            np.linalg.norm(solutions["optimized"] - solutions["legacy"], ord="fro")
            / np.linalg.norm(solutions["legacy"], ord="fro")
        )
        results["speedup_median"] = float(
            results["variants"]["legacy"]["wall_s_median"]
            / results["variants"]["optimized"]["wall_s_median"]
        )
    label = "_".join(variant_names)
    output_path = args.output_dir / f"fft_core_profile_{label}.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    compact = {
        "grid": results["grid"],
        "repeats": results["repeats"],
        "wall_s_median": {
            name: values["wall_s_median"]
            for name, values in results["variants"].items()
        },
        "speedup_median": results.get("speedup_median"),
        "relative_solution_difference": results.get("relative_solution_difference"),
    }
    print(json.dumps(compact, indent=2))
    print(f"SAVED {output_path}")


if __name__ == "__main__":
    main()
