"""Measure paired implementation controls in isolated GPU processes.

The process-local patches use private interfaces and preserve fresh acceptance.
They are experiments, not configurable production solver implementations.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..run import ROOT, digest, environment, plain, strict_record

PAIRS = {
    "hex_stencil": ("hex8", "production", "gather"),
    "hex_krylov": ("hex8", "production", "hex_krylov"),
    "hex_parseval": ("hex8", "hex_krylov", "spatial_rho"),
    "hex_inverse": ("hex8", "production", "inverse_rebuild"),
    "hex_fft": ("hex8", "production", "library"),
    "gani_fft": ("gani", "production", "library"),
    "gani_krylov": ("gani", "production", "vectors"),
    "gani_krylov_energy": ("gani", "production", "energy"),
    "gani_energy": ("gani", "vectors", "energy"),
    "gani_slabs": ("gani", "production", "slabs"),
    "gani_reference": ("gani", "production", "identity_reference"),
    "gani_loads": ("gani", "production", "direct_loads"),
    "hex_evaluation": ("hex8", "production", "ieee_evaluation"),
    "gani_evaluation": ("gani", "production", "ieee_evaluation"),
}


def patch(arm, counts):
    """Install only the requested experimental operation inside this worker."""
    from cufhom.fft.linear import coordinates, hex8_fused

    from . import controls as control

    if arm in ("gather", "hex_krylov", "spatial_rho", "inverse_rebuild"):
        control.patch_hex8("krylov_separate" if arm == "hex_krylov" else arm, counts)
    elif arm == "vectors":
        control.patch_gani_vectors(counts)
    elif arm == "energy":
        control.patch_gani_energy(counts)
    elif arm == "slabs":
        control.patch_gani_slabs(counts)
    elif arm == "direct_loads":
        control.patch_gani_direct_loads(counts)
    elif arm == "library":
        coordinates.fused_supported = lambda *args: False
        hex8_fused.fused_pcg = lambda op: None


def compare_records(control, variant):
    """Check matched inputs and physical acceptance before reporting a pair.

    Iteration counts may differ between variants and are reported explicitly.
    Tensor differences are diagnostic, not a proof of equal local fields.
    """
    for key in ("pair", "precision", "n", "input_hashes"):
        if control[key] != variant[key]:
            raise ValueError(f"Paired records disagree on {key}")
    reference_changed = not np.array_equal(control["reference"], variant["reference"])
    if reference_changed and control["pair"] != "gani_reference":
        raise ValueError("Paired records disagree on reference")
    details = {"reference_changed": reference_changed}
    tensors = []
    for name, record in (("control", control), ("variant", variant)):
        if record["status"] != "accepted":
            raise ValueError("Paired arm was not accepted")
        warm = [call for call in record["calls"] if call["state"] == "warm"]
        if len(warm) != 5:
            raise ValueError("Expected five warm calls per arm")
        for call in record["calls"]:
            if len(call["audits"]) != 6 or len(call["iterations"]) != 6:
                raise ValueError("Expected six audited loads per call")
            for audit in call["audits"]:
                if (
                    not audit["accepted"]
                    or not np.isfinite(audit["residual"])
                    or not np.isfinite(audit["threshold"])
                    or audit["residual"] > audit["threshold"]
                ):
                    raise ValueError("Fresh physical acceptance failed")
        matrices = [np.asarray(call["C"], dtype=float) for call in warm]
        if any(c.shape != (6, 6) or not np.isfinite(c).all() for c in matrices):
            raise ValueError("Invalid effective tensor")
        times = [call["time_s"] for call in warm]
        if not all(np.isfinite(t) and t > 0 for t in times):
            raise ValueError("Invalid warm-call time")
        tensors.append(matrices)
        details[name] = {
            "median_s": statistics.median(times),
            "iterations_per_call": [call["iterations"] for call in warm],
            "fresh_relative_per_call": [
                [audit["relative"] for audit in call["audits"]] for call in warm
            ],
            "evaluation_per_call": [
                [audit["evaluation"] for audit in call["audits"]] for call in warm
            ],
        }
    details["max_relative_tensor_difference"] = max(
        float(np.linalg.norm(v - c) / np.linalg.norm(c))
        for c in tensors[0]
        for v in tensors[1]
    )
    return details


def worker(args):
    """Warm one arm once, then record five fresh synchronized six-load calls."""
    import cupy as cp
    from cufhom.fft import IndexedMaterial, homogenize
    from cufhom.fft.linear.common import isotropic_reference

    from ..cases import prepare_case

    target = args.output
    if target.exists():
        raise FileExistsError(target)
    record = {
        "status": "running",
        "pair": args.pair,
        "arm": args.arm,
        "precision": args.precision,
        "environment": environment(),
    }
    try:
        kind, _, _ = PAIRS[args.pair]
        n = args.n or (192 if kind == "hex8" else 189)
        index, table, metadata = prepare_case(args.case, n)
        dtype = "float32" if args.precision == "fp32" else "float64"
        material = IndexedMaterial(index, table, dtype=dtype)
        reference, _, shear = isotropic_reference(material.stiffness)
        if args.arm == "identity_reference":
            reference = 2 * shear * np.eye(6)
        counts = defaultdict(int)
        patch(args.arm, counts)
        evaluation = "float64" if args.arm == "ieee_evaluation" else "df64"
        cp.zeros(1).sum()
        cp.cuda.Device().synchronize()
        record.update(
            n=n,
            inputs=metadata,
            input_hashes={
                "index": digest(index),
                "stiffness_solver": digest(material.stiffness),
            },
            reference=reference,
            evaluation_requested=evaluation,
            timing_boundary="material/reference built outside; synchronized six-load homogenize with fresh native audits and corrected assembly; one discarded warmup, five recorded warm calls",
            calls=[],
        )
        for repetition in range(6):
            cp.cuda.Device().synchronize()
            begin = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                result = homogenize(
                    material,
                    discretization=kind,
                    reference=reference,
                    working="float64" if args.precision == "fp64" else "float32",
                    rtol=1e-5 if dtype == "float32" else 1e-6,
                    evaluation=evaluation,
                )
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - begin
            if len(result["audits"]) != 6 or not all(
                item["accepted"] for item in result["audits"]
            ):
                raise RuntimeError("Fresh six-load audit failed")
            record["calls"].append(
                {
                    "state": "warmup" if repetition == 0 else "warm",
                    "time_s": elapsed,
                    "iterations": result["iterations"],
                    "audits": result["audits"],
                    "C": result["C"],
                    "execution": result["execution"],
                    "precision_contract": result["precision_contract"],
                }
            )
        exercised = {
            "gather": "gather_launches",
            "hex_krylov": "separate_iterations",
            "spatial_rho": "spatial_reductions",
            "inverse_rebuild": "tables_released",
            "vectors": "compiled_separate_fft_kernels",
            "energy": "compiled_separate_energy_kernels",
            "slabs": "staged_actions",
            "direct_loads": "direct_load_evaluations",
        }
        if args.arm in exercised and counts[exercised[args.arm]] == 0:
            raise RuntimeError("Requested experimental control was not exercised")
        if args.pair == "gani_loads":
            expected = args.arm != "direct_loads"
            if any(
                audit["native"]["phase_spectra_rhs"] != expected
                for call in record["calls"]
                for audit in call["audits"]
            ):
                raise RuntimeError("Requested GaNi load construction was not exercised")
        if args.pair in ("hex_evaluation", "gani_evaluation"):
            expected = "float64" if args.arm == "ieee_evaluation" else "df64"
            if any(
                not audit["evaluation"].startswith(expected)
                for call in record["calls"]
                for audit in call["audits"]
            ):
                raise RuntimeError(
                    "Requested fresh evaluation arithmetic was not exercised"
                )
        if args.arm == "library":
            if kind == "gani" and any(
                item["execution"] != "library_fft" for item in record["calls"]
            ):
                raise RuntimeError(
                    "GaNi library FFT control did not select library FFT"
                )
            if kind == "hex8" and any(
                audit["native"].get("preconditioner_fft") != "cufft"
                for item in record["calls"]
                for audit in item["audits"]
            ):
                raise RuntimeError("HEX8 library control did not select cuFFT")
        record.update(status="accepted", instrumentation=dict(counts))
    except Exception as error:
        record.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(
            strict_record(record), stream, default=plain, indent=2, allow_nan=False
        )
    return int(record["status"] != "accepted")


def main():
    """Alternate control/variant order across independent paired sessions."""
    os.environ.setdefault("CUFHOM_FUSED_COMPILE", "sync")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", choices=PAIRS, required=True)
    parser.add_argument(
        "--precision", choices=("fp64", "mixed", "fp32"), default="fp64"
    )
    parser.add_argument("--case", choices=("public", "gyroid"), default="public")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument(
        "--n", type=int, help="Explicit pilot grid; paper uses 192 HEX8, 189 GaNi"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory; default: results/ablations/<pair>_<precision>_<unique-id>",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output is None:
        if args.arm:
            parser.error("Internal workers require an explicit output file")
        args.output = (
            ROOT
            / "results"
            / "ablations"
            / f"{args.pair}_{args.precision}_{time.time_ns()}"
        )
    if args.sessions < 1:
        parser.error("sessions must be positive")
    if args.pair == "hex_inverse" and args.precision != "fp64":
        parser.error("The stored inverse control is FP64 only")
    if args.pair in ("hex_evaluation", "gani_evaluation") and args.precision != "mixed":
        parser.error("The fresh evaluation comparison is mixed precision only")
    if args.n is not None and args.n < 2:
        parser.error("n must be at least two")
    if args.arm and args.arm not in PAIRS[args.pair][1:]:
        parser.error("arm must belong to the selected pair")
    if args.arm:
        return worker(args)
    _, control, variant = PAIRS[args.pair]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "pair": args.pair,
                    "arms": [control, variant],
                    "sessions": args.sessions,
                    "calls_per_arm": "one warmup + five warm",
                    "pilot_n": args.n,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    if args.output.exists():
        raise FileExistsError(f"Use a new directory: {args.output}")
    args.output.mkdir(parents=True)
    ratios = []
    comparisons = []
    for session in range(1, args.sessions + 1):
        arms = (control, variant) if session % 2 else (variant, control)
        times = {}
        records = {}
        for arm in arms:
            target = args.output / f"session{session}_{arm}.json"
            command = [
                sys.executable,
                "-m",
                __package__,
                "--pair",
                args.pair,
                "--precision",
                args.precision,
                "--case",
                args.case,
                "--arm",
                arm,
                "--output",
                str(target),
            ]
            if args.n:
                command.extend(["--n", str(args.n)])
            subprocess.run(command, check=True, cwd=ROOT)
            record = json.loads(target.read_text())
            records[arm] = record
            times[arm] = statistics.median(
                item["time_s"] for item in record["calls"] if item["state"] == "warm"
            )
        comparisons.append(compare_records(records[control], records[variant]))
        ratios.append(times[variant] / times[control])
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "variant_over_control_per_session": ratios,
                "median": statistics.median(ratios),
                "range": [min(ratios), max(ratios)],
                "paired_diagnostics": comparisons,
                "interpretation": "descriptive paired ratios, not confidence intervals; changes in PCG counts and operation scope must be inspected",
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sessions": args.sessions,
                "median_variant_over_control": statistics.median(ratios),
            },
            indent=2,
        )
    )
    return 0
