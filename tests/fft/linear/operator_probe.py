"""Compare equivalent GaNi actions, not complete solves (no CUDA by default).

Example, after obtaining the serial GPU slot::

  python -B tests/fft/linear/operator_probe.py --gpu --working float32 \
    --evaluation df64 --shape 16 16 16 --shape 15 18 20 --shape 17 19 21 \
    --output /tmp/p2_operator.json

Use a new process and --cache-dir pointing to an empty directory for a cold
disk cache. First-call costs include planning/allocation; compile helper time
is recorded separately. CUDA event intervals are stream elapsed time, including
idle gaps caused by host dispatch, not a sum of kernel execution durations.
"""

import argparse
import ast
import gc
import hashlib
import json
import os
import statistics
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STAGES = ("k_ainv", "k_binv", "k_c", "k_bfwd", "k_afwd")


def source_hashes():
    files = [
        Path(__file__),
        ROOT / "src/fft/operators.py",
        ROOT / "core/tensors/fft.py",
    ]
    files += sorted((ROOT / "src/fft/linear").glob("*.py"))
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }


def host_report(shape, working, evaluation):
    """Static source inspection; deliberately imports neither CuPy nor CuFHom."""
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 2:
        raise ValueError("shape must contain three dimensions >= 2")
    if working not in ("float32", "float64") or evaluation not in ("df64", "float64"):
        raise ValueError(
            "working must be IEEE float32/float64; evaluation df64/float64"
        )
    tree = ast.parse((ROOT / "src/fft/linear/fused.py").read_text())
    definitions = {
        n.name: n.lineno
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    n = int(np.prod(shape))
    item = np.dtype(working).itemsize
    return dict(
        shape=shape,
        working=working,
        evaluation_requested=evaluation,
        definitions=definitions,
        generated_support="must query actual grid_supported on the authorized GPU",
        hybrid_spectrum_estimate_bytes=6
        * 2
        * item
        * shape[0]
        * shape[1]
        * (shape[2] // 2),
        previous_full_solver_estimate_126N_bytes=126 * n,
        scope="GaNi operator only; no PCG, six-load solutions or tensor verification",
        evaluation_scope="separate fresh total-strain residual; never a df64 PCG",
    )


def error_metrics(actual, reference):
    """FP64 host comparison of complex vectors; rejects nonfinite data."""
    a, b = np.asarray(actual, np.complex128), np.asarray(reference, np.complex128)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("equivalence requires matching finite vectors")
    difference = a - b
    norm = np.linalg.norm(b)
    return dict(
        relative_l2=float(np.linalg.norm(difference) / max(norm, np.finfo(float).tiny)),
        absolute_linf=float(np.max(np.abs(difference), initial=0)),
    )


def array_inventory(values):
    """Exact logical nbytes and unique allocator blocks, including aliased views."""
    rows, blocks = [], {}
    for name, value in values.items():
        if (
            value is None
            or isinstance(value, type)
            or not hasattr(value, "nbytes")
            or not hasattr(value, "data")
        ):
            continue
        memory = getattr(value.data, "mem", None)
        ptr = int(memory.ptr) if memory is not None else int(value.data.ptr)
        size = int(memory.size) if memory is not None else int(value.nbytes)
        rows.append(
            dict(
                name=name,
                shape=list(value.shape),
                dtype=str(value.dtype),
                nbytes=int(value.nbytes),
                allocation_id=ptr,
            )
        )
        blocks[ptr] = size
    return dict(
        arrays=rows,
        logical_bytes=sum(x["nbytes"] for x in rows),
        unique_allocation_bytes=sum(blocks.values()),
        unique_allocations=len(blocks),
        note="Unique blocks include allocator alignment; views are counted once. "
        "Inventory excludes unnamed library temporaries and driver allocations.",
    )


def memory_checkpoint(cp, fft):
    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return dict(
        active_pool_bytes=pool.used_bytes(),
        reserved_pool_bytes=pool.total_bytes(),
        device_free_bytes=free,
        device_total_bytes=total,
        manual_fft_plan_count=len(fft._CUPY_PLAN_CACHE),
    )


@contextmanager
def compile_watch(module):
    """Measure the existing compile helper, including cache loads, without editing it."""
    original, samples = module._compile_many, []

    def wrapped(specs):
        specs = list(specs)
        start = time.perf_counter()
        result = original(specs)
        samples.append(
            dict(
                wall_s=time.perf_counter() - start,
                kernels=[s[1] for s in specs],
                source_bytes=sum(len(s[0].encode()) for s in specs),
            )
        )
        return result

    module._compile_many = wrapped
    try:
        yield samples
    finally:
        module._compile_many = original


def measure(cp, fft, action, repeats, batch):
    """Keep only the latest output; measure allocations in a separate call."""
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    out = action()
    cp.cuda.Device().synchronize()
    cold = time.perf_counter() - start
    del out
    out = action()
    cp.cuda.Device().synchronize()
    del out
    events, walls = [], []
    for _ in range(repeats):
        cp.cuda.Device().synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        start = time.perf_counter()
        begin.record()
        for _ in range(batch):
            out = action()
            del out
        end.record()
        end.synchronize()
        walls.append((time.perf_counter() - start) / batch)
        events.append(cp.cuda.get_elapsed_time(begin, end) / batch)
    before = memory_checkpoint(cp, fft)
    peak = [before["active_pool_bytes"], before["reserved_pool_bytes"]]
    stop = threading.Event()

    def sample():
        pool = cp.get_default_memory_pool()
        while not stop.wait(0.001):
            peak[0] = max(peak[0], pool.used_bytes())
            peak[1] = max(peak[1], pool.total_bytes())

    monitor = threading.Thread(target=sample)
    monitor.start()
    try:
        out = action()
        cp.cuda.Device().synchronize()
        after = memory_checkpoint(cp, fft)
        peak[0] = max(peak[0], after["active_pool_bytes"])
        peak[1] = max(peak[1], after["reserved_pool_bytes"])
        host_output = cp.asnumpy(out)
    finally:
        stop.set()
        monitor.join()
    return dict(
        first_call_wall_s=cold,
        event_ms=events,
        wall_s=walls,
        median_event_ms=statistics.median(events),
        median_wall_s=statistics.median(walls),
        min_event_ms=min(events),
        max_event_ms=max(events),
        memory_before=before,
        memory_with_output=after,
        sampled_active_peak_bytes=peak[0],
        sampled_reserved_peak_bytes=peak[1],
        memory_sampling="1 ms plus endpoints, separate call; may miss short temporaries. "
        "Not an exact high-water mark or RSS; caches persist between routes.",
    ), host_output


def kernel_inventory(pipeline):
    rows = []
    for name in STAGES:
        kernel = getattr(pipeline, name)
        rows.append(
            dict(
                stage=name,
                attributes=dict(kernel.attributes),
                dynamic_shared_bytes=getattr(
                    pipeline,
                    "smem_"
                    + (
                        "a"
                        if name in ("k_ainv", "k_afwd")
                        else "c"
                        if name == "k_c"
                        else "b"
                    ),
                ),
            )
        )
    return rows


def gpu_case(shape, args):
    import cupy as cp
    from cufhom.fft.linear import evaluation, fused
    from cufhom.fft.linear.energy import material_energy
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom.fft.operators import MixedProjector
    from cufhom_core.tensors import Tensor, fft

    record = host_report(shape, args.working, args.evaluation)
    free, _ = cp.cuda.runtime.memGetInfo()
    estimate = 768 * int(
        np.prod(shape)
    )  # conservative probe preflight, not measured memory
    record["preflight"] = dict(
        estimated_bytes=estimate,
        free_bytes=free,
        max_free_fraction=args.max_memory_fraction,
    )
    if estimate > free * args.max_memory_fraction:
        return dict(record, status="skipped_memory_preflight")
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual", preserve_cache=True)
    Q = SymmetricCoordinates(
        MixedProjector(shape, args.lengths, np.ones(6, bool), cp), 5.0, 2.0
    )
    rng = np.random.default_rng(args.seed)
    table = rng.normal(size=(2, 6, 6))
    table = table.transpose(0, 2, 1) @ table + 2 * np.eye(6)
    table[1] *= 4
    ids = rng.integers(0, 2, size=shape, dtype=np.int32)
    field = rng.standard_normal((6,) + tuple(shape)).astype(args.working)

    def descriptor(dtype):
        A = Tensor(
            name="p2_indexed",
            val=cp.asarray(ids[None]),
            N=shape,
            order=1,
            Fourier=False,
            fft_form="r",
            multype="sym21_indexed",
        )
        A.material_table = cp.asarray(
            np.ascontiguousarray(table[:, *np.triu_indices(6)]), dtype=dtype
        )
        return A

    A = descriptor(args.working)
    z = Q.adjoint(cp.asarray(field))
    record["input_sha256"] = hashlib.sha256(
        ids.tobytes() + table.tobytes() + field.tobytes()
    ).hexdigest()
    record["initial_memory"] = memory_checkpoint(cp, fft)

    def library():
        strain = Q.strain(z)
        stress, _ = material_energy(A, strain)
        return Q.adjoint(stress)

    record["library"], reference = measure(cp, fft, library, args.repeats, args.batch)
    record["library"]["implementation"] = (
        "production coordinates, cuFFT, material_energy, adjoint; same H=RQ*FFT C IFFT QR"
    )
    supported = fused.fused_supported(Q, A)
    record["generated_supported"] = supported
    record["library_input_inventory"] = array_inventory(
        dict(ids=A.val, table=A.material_table, z=z)
    )
    if not supported:
        record["generated"] = dict(
            status="unavailable",
            reason="Actual fused_supported rejected grid/contract; library remains measured",
        )
        record["fresh_evaluation"] = dict(
            status="unavailable", reason="No generated tiled pipeline on this grid"
        )
        return dict(record, status="library_only")
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    with compile_watch(fused) as compile_samples:
        pipeline = fused.FusedPipeline(Q, A)
        # Force the host matvec variants even when constructor prepares device graphs.
        pipeline._ensure(STAGES)
    cp.cuda.Device().synchronize()
    record["generated_setup"] = dict(
        wall_s=time.perf_counter() - start,
        compile_helper_calls=compile_samples,
        compile_helper_wall_s=sum(s["wall_s"] for s in compile_samples),
        note="Compile helper wall time includes NVRTC or disk-cache load, not necessarily fresh compilation. "
        "Constructor wall also includes code generation, tables, buffers and compaction.",
    )
    tiled = pipeline.to_fused(z)
    record["generated"], actual = measure(
        cp, fft, lambda: pipeline.matvec(tiled), args.repeats, args.batch
    )
    actual = cp.asnumpy(pipeline.to_production(cp.asarray(actual)))
    record["operator_error"] = error_metrics(actual, reference)
    record["kernels"] = kernel_inventory(pipeline)
    record["generated_config"] = pipeline.config
    values = {
        name: value
        for name, value in vars(pipeline).items()
        if hasattr(value, "nbytes")
    }
    values.update(
        {f"twiddle_{i}": t for i, t in enumerate(pipeline._parts["twiddles"])}
    )
    values.update(
        z=z, tiled_input=tiled, original_ids=A.val, original_table=A.material_table
    )
    norm = pipeline._parts.get("frequency_norm")
    if norm is not None:
        values["cached_frequency_norm"] = norm
    record["generated_buffers"] = array_inventory(values)
    record["stages"] = [
        "KA_inv: compatible map and inverse axis 0 -> writes S",
        "KB_inv: inverse axis 1 -> S in place",
        "KC: inverse real axis 2, material and energy, forward axis 2 -> S in place",
        "KB_fwd: forward axis 1 -> S in place",
        "KA_fwd: forward axis 0, adjoint, residual update and norm -> output; matvec negates output",
    ]
    record["traffic_estimate"] = dict(
        hybrid_spectrum_bytes=pipeline.S.nbytes,
        hybrid_logical_bytes_per_action=8 * pipeline.S.nbytes,
        note="Estimated S-only reads/writes (1+2+2+2+1). Excludes vectors, ids, tables, "
        "twiddles and reductions. L2/slab reuse means this is NOT measured DRAM traffic.",
    )
    record["fresh_evaluation"] = fresh_evaluation(
        cp, fft, evaluation, Q, pipeline, descriptor("float64"), field, args
    )
    record["final_memory"] = memory_checkpoint(cp, fft)
    gc.collect()
    return dict(record, status="compared")


def fresh_evaluation(cp, fft, module, Q, pipeline, A, field, args):
    """Compare the configured total-strain residual with production cuFFT FP64."""
    from cufhom.fft.linear.energy import material_energy

    Qt = Q.tiled(pipeline.config["ta"])
    if not module.evaluation_supported(Qt, A):
        return dict(
            status="unavailable", reason="Actual evaluation_supported rejected grid"
        )
    z = Qt.adjoint(cp.asarray(field, dtype=cp.float64))
    macro = np.array([0.7, -0.2, 0.1, 0.03, -0.04, 0.05], dtype=np.float64)

    def library():
        strain = Qt.strain(z)
        strain += cp.asarray(macro).reshape((6, 1, 1, 1))
        stress, _ = material_energy(A, strain)
        return (-Qt.adjoint(stress)).astype(cp.complex64)

    lib, reference = measure(cp, fft, library, args.repeats, args.batch)
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    with compile_watch(module) as samples:
        evaluator = module.EvaluationPipeline(Qt, A, args.evaluation, ids=pipeline.ids)
    cp.cuda.Device().synchronize()
    setup = time.perf_counter() - start
    residual = cp.empty(z.shape, cp.complex64)

    def generated():
        evaluator.inverse(z)
        evaluator.forward(macro, residual)
        return residual

    gen, actual = measure(cp, fft, generated, args.repeats, args.batch)
    values = {k: v for k, v in vars(evaluator).items() if hasattr(v, "nbytes")}
    values.update({f"twiddle_{i}": t for i, t in enumerate(evaluator.twiddles)})
    values.update(z_fp64=z, residual_fp32=residual)
    return dict(
        status="compared",
        arithmetic=args.evaluation,
        stored_residual="complex64",
        library=lib,
        generated=gen,
        error=error_metrics(actual, reference),
        setup_wall_s=setup,
        compile_helper_calls=samples,
        buffers=array_inventory(values),
        note="Generated forward includes production scalar/stress host copies. "
        "Library timing returns only the residual: timings are not identical audit workloads.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs=3, action="append")
    parser.add_argument("--working", required=True, choices=("float32", "float64"))
    parser.add_argument("--evaluation", required=True, choices=("df64", "float64"))
    parser.add_argument("--lengths", type=float, nargs=3, default=(0.8, 1.3, 1.1))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-memory-fraction", type=float, default=0.5)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run CUDA only during the authorized serial slot",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    shapes = args.shape or [(16, 16, 16), (15, 18, 20), (17, 19, 21)]
    if args.repeats < 1 or args.batch < 1 or any(min(s) < 2 for s in shapes):
        parser.error("repeats/batch >= 1 and shape dimensions >= 2 required")
    if (
        not 0 < args.max_memory_fraction <= 1
        or not np.isfinite(args.lengths).all()
        or min(args.lengths) <= 0
    ):
        parser.error(
            "finite positive lengths and 0 < max-memory-fraction <= 1 required"
        )
    report = dict(
        mode="cuda" if args.gpu else "host_only",
        source_sha256=source_hashes(),
        numpy=np.__version__,
        seed=args.seed,
        cases=[],
        configuration=dict(
            working=args.working,
            evaluation=args.evaluation,
            lengths=args.lengths,
            shapes=shapes,
            repeats=args.repeats,
            batch=args.batch,
            max_memory_fraction=args.max_memory_fraction,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
        ),
        limitations=[
            "No complete solve or six-load performance claim.",
            "CUDA events measure stream elapsed time; host stalls can appear as GPU idle intervals.",
            "No measured DRAM traffic, achieved occupancy or launch counts; use Nsight for those.",
            "Pool active/reserved are distinct from driver free memory and exact peak memory.",
            "Cases share process caches; use one --shape per new process for isolated cold comparisons.",
        ],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.gpu:
            os.environ["CUFHOM_FUSED_COMPILE"] = "sync"
            if args.cache_dir:
                args.cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ["CUPY_CACHE_DIR"] = str(args.cache_dir / "cupy")
                os.environ["CUFHOM_KERNEL_CACHE"] = str(args.cache_dir / "kernels.json")
            import cupy as cp

            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = props["name"]
            name = name.decode() if isinstance(name, bytes) else name
            report["cuda"] = dict(
                cupy=cp.__version__,
                device=name,
                runtime=cp.cuda.runtime.runtimeGetVersion(),
                driver=cp.cuda.runtime.driverGetVersion(),
            )
        for shape in shapes:
            try:
                case = (
                    gpu_case(tuple(shape), args)
                    if args.gpu
                    else host_report(shape, args.working, args.evaluation)
                )
            except Exception as exc:
                case = dict(
                    shape=shape,
                    status="failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
            report["cases"].append(case)
            save()
    except Exception as exc:
        report["fatal_error"] = dict(
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        save()
    print(f"Saved {len(report['cases'])} {report['mode']} cases to {args.output}")
    if any(c.get("status") == "failed" for c in report["cases"]):
        raise RuntimeError(f"Operator probe failed; diagnostics saved in {args.output}")
    return report


if __name__ == "__main__":
    main()
