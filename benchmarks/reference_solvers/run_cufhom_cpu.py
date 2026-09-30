#!/usr/bin/env python3
"""Time the CPU CuFHom FFT solver on the shared SFRP voxel geometry."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from cufhom.pipeline.fft_solver import solve_homogenization


HERE = Path(__file__).resolve().parent
CASE = HERE / "sfrp_vf15_ar10_random_g48"
OUT = CASE / "results" / "cufhom_cpu"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(CASE / "geometry.npz")
    base = {
        "phase_array": data["phase"], "ori_array": data["ori"],
        "Em": 3.0e9, "nu_m": 0.35,
        "Ef_L": 70.0e9, "Ef_T": 12.0e9, "nu_LT": 0.22,
        "nu_TT": 0.25, "G_LT": 8.0e9,
        "fft_backend": "scipy", "solver_profile": "timing",
        "solver_maxiter": 1000, "require_convergence": True,
        "solver_verbose": False, "active_load_ids": [0],
        "partial_load_output": True, "load_batch_size": 1,
        "fft_workers": 1,
    }
    results = {"case": str(CASE), "variants": {}}
    for name, storage in (("dense_full", "full"), ("packed_sym21", "sym21")):
        timing = OUT / f"{name}_timing.json"
        started = time.perf_counter()
        ceff = solve_homogenization({**base, "cfield_storage": storage, "solver_timing_path": str(timing)})
        elapsed = time.perf_counter() - started
        record = json.loads(timing.read_text(encoding="utf-8"))
        record["outer_wall_s"] = elapsed
        record["C_xxxx_Pa"] = float(np.asarray(ceff)[0, 0])
        (OUT / f"{name}_result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        results["variants"][name] = {"wall_s": elapsed, "C_xxxx_Pa": record["C_xxxx_Pa"]}
        print(f"SAM {name}: {elapsed:.3f} s")
    (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
