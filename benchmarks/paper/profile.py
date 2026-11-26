"""Collect an optional warm CUDA activity trace without calling it a benchmark.

The Chrome trace retains individual GPU events. Summed event durations can
overlap; only the union of recorded intervals estimates observed busy time.
Profiler overhead makes this call unsuitable for comparison to unprofiled time.
"""

import argparse
import json
from pathlib import Path

from .cases import CASES, prepare_case
from .run import environment, plain, strict_record


def summarize_intervals(intervals):
    """Return the union length of half-open intervals in the input time unit."""
    total, end = 0, None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("An event cannot finish before it starts")
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def main(argv=None):
    """Warm the selected implementation, profile one call and save raw events."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("cufhom", "fans"), default="cufhom")
    parser.add_argument("--case", choices=CASES, default="public")
    parser.add_argument("--grid", type=int, default=192)
    parser.add_argument("--discretization", choices=("gani", "hex8"), default="hex8")
    parser.add_argument(
        "--precision", choices=("fp64", "mixed", "fp32"), default="fp64"
    )
    parser.add_argument("--source", type=Path, help="Optional UNOCG checkout for FANS")
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    trace_path = args.output.with_suffix(".trace.json")
    if args.output.exists() or trace_path.exists():
        parser.error("Refusing to overwrite an existing record or trace")
    if not 0 < args.eta < 1:
        parser.error("eta must be in (0,1)")
    if args.solver == "fans" and (
        args.discretization != "hex8" or args.precision != "fp64"
    ):
        parser.error("The independent FANS profile is HEX8 FP64")
    index, table, inputs = prepare_case(args.case, args.grid)
    if args.solver == "cufhom":
        import cupy as cp
        from cufhom.fft import IndexedMaterial, homogenize
        from cufhom.fft.linear.common import isotropic_reference

        cp.zeros(1).sum()
        reference, _, _ = isotropic_reference(IndexedMaterial(index, table).stiffness)
        dtype = "float32" if args.precision == "fp32" else "float64"
        working = "float64" if args.precision == "fp64" else "float32"

        def call():
            return homogenize(
                IndexedMaterial(index, table, dtype=dtype),
                backend="cupy",
                discretization=args.discretization,
                reference=reference,
                working=working,
                rtol=args.eta,
            )

        synchronize = cp.cuda.Device().synchronize
    else:
        from .external import _source_path, fans

        _source_path(args.source, "unocg")

        def call():
            return fans(index, table, args.eta, source=args.source)

    # CuPy initializes its CUDA libraries before PyTorch for the native route.
    import torch

    if args.solver == "fans":
        synchronize = torch.cuda.synchronize
    warm_result = call()
    synchronize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        profiled_result = call()
        synchronize()
    profiler.export_chrome_trace(str(trace_path))
    kernels, intervals = {}, []
    for event in profiler.profiler.kineto_results.events():
        if event.device_type() != torch.autograd.DeviceType.CUDA:
            continue
        name = event.name()
        row = kernels.setdefault(name, {"events": 0, "summed_duration_ns": 0})
        row["events"] += 1
        row["summed_duration_ns"] += int(event.duration_ns())
        start = int(event.start_ns())
        intervals.append((start, start + int(event.duration_ns())))
    record = {
        "environment": environment(),
        "torch": torch.__version__,
        "inputs": inputs,
        "configuration": vars(args),
        "warm_result": warm_result,
        "profiled_result": profiled_result,
        "kernels": kernels,
        "observed_gpu_interval_union_ns": summarize_intervals(intervals),
        "status": "recorded" if intervals else "no_cuda_events",
        "trace": str(trace_path),
        "scope": "one unprofiled warm call, then one profiled six-load call; fresh preparation on each",
        "limitation": "event sums can overlap; profiler durations are not unprofiled wall time",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(
            strict_record(record), stream, default=plain, indent=2, allow_nan=False
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
