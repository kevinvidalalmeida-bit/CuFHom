"""Measure three isolated cold calls at every native Figure-11 grid.

Use private empty CuPy/CuFHom disk caches for each worker. The CUDA driver
cache is retained. Historical first/warm measurements are never rewritten.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .cases import prepare_case
from .run import MemorySampler, digest, environment, host_memory, plain, strict_record

ROOT = Path(__file__).resolve().parents[2]
GRIDS = (64, 128, 192, 256, 320, 384, 448, 512)
MODES = (("hex8", "fp64"), ("gani", "mixed"), ("hex8", "mixed"), ("gani", "fp64"))


def dump(path, value):
    """Write strict JSON atomically without encoding nonfinite numbers as values."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(strict_record(value), default=plain, indent=2, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def conditions(grids=GRIDS, passes=3):
    """Balance mode order between grids and independent passes."""
    points = []
    for session in range(1, passes + 1):
        for index, n in enumerate(grids):
            offset = (session + index - 1) % len(MODES)
            for disc, precision in MODES[offset:] + MODES[:offset]:
                points.append(
                    dict(n=n, discretization=disc, precision=precision, session=session)
                )
    return points


def name(point):
    """Return the stable identifier of an intended independent cold call."""
    return f"n{point['n']}_{point['discretization']}_{point['precision']}_pass{point['session']}"


def source_manifest():
    """Identify actual worktree source bytes rather than relying on a commit alone."""
    paths = list((ROOT / "src").rglob("*.py")) + list((ROOT / "core").rglob("*.py"))
    paths += list((ROOT / "benchmarks/paper").rglob("*.py"))
    paths += list((ROOT / "benchmarks/paper/inputs").glob("*"))
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if p.is_file()
    }


def worker(point, destination, cache):
    """Run exactly one synchronized six-load call after context-only initialization."""
    cache.mkdir(parents=True, exist_ok=False)
    (cache / "cupy").mkdir()
    os.environ.update(
        CUPY_CACHE_DIR=str(cache / "cupy"),
        CUFHOM_KERNEL_CACHE=str(cache / "dispatch.json"),
        CUFHOM_FUSED_COMPILE="sync",
    )
    runtime = Path(sysconfig.get_path("purelib")) / "nvidia/cuda_runtime"
    if (
        not os.environ.get("CUDA_PATH")
        and (runtime / "include/cuda_runtime.h").exists()
    ):
        os.environ["CUDA_PATH"] = str(runtime)
    record = dict(point=point, status="running", cache=str(cache), case="public")
    try:
        import cupy as cp
        from cufhom.fft import IndexedMaterial, homogenize
        from cufhom.fft.linear.common import isotropic_reference

        record["environment"] = environment()
        # No CuPy array/reduction warmup: it could populate the private disk cache.
        cp.cuda.runtime.free(0)
        cp.cuda.Device().synchronize()
        index, table, inputs = prepare_case("public", point["n"])
        descriptor = IndexedMaterial(index, table, dtype="float64")
        reference, bulk, shear = isotropic_reference(descriptor.stiffness)
        record.update(
            inputs=inputs,
            input_hashes=dict(
                index=digest(index), stiffness=digest(descriptor.stiffness)
            ),
            reference=dict(matrix=reference, bulk=bulk, shear=shear),
            host_memory_before=host_memory(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
            gpu_properties=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
                "name"
            ].decode(),
            timing_boundary="host index/table, material descriptor, isotropic reference, imports and CUDA context excluded; synchronized homogenize with in-call setup, synchronous kernel compilation, six zero-start solves, native fresh audits, corrected assembly and host tensor return included",
            cache_policy="private empty application caches per worker; CUDA driver cache retained; no solver warmup",
        )
        before = [str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file()]
        record["application_cache_files_before"] = before
        if before:
            raise RuntimeError("Application cache populated before the cold clock")
        cp.cuda.Device().synchronize()
        with MemorySampler(cp) as sampler:
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                result = homogenize(
                    descriptor,
                    discretization=point["discretization"],
                    backend="cupy",
                    working="float64" if point["precision"] == "fp64" else "float32",
                    reference=reference,
                    rtol=1e-6,
                    atol=0.0,
                    maxiter=3000,
                    evaluation="df64",
                )
            cp.cuda.Device().synchronize()
            elapsed = time.perf_counter() - started
        audits = result["audits"]
        tensor = np.asarray(result["C"])
        if len(audits) != 6 or not all(
            a["accepted"] and a["relative"] <= 1e-6 for a in audits
        ):
            raise RuntimeError("All six fresh residual checks must pass")
        if (
            tensor.shape != (6, 6)
            or not np.isfinite(tensor).all()
            or np.linalg.eigvalsh(tensor).min() <= 0
        ):
            raise RuntimeError("Invalid cold-call stiffness tensor")
        record.update(
            status="accepted",
            time_s=elapsed,
            eta=1e-6,
            audits=audits,
            iterations=result["iterations"],
            C_corrected=tensor,
            C_direct=result.get("C_stress"),
            precision_contract=result["precision_contract"],
            execution=result["execution"],
            residency=result.get("memory"),
            memory_measurement=sampler.report(),
            host_memory_after=host_memory(),
            solver_stdout=captured.getvalue(),
        )
    except Exception as error:
        record.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["application_cache_files_after"] = [
        str(p.relative_to(cache)) for p in cache.rglob("*") if p.is_file()
    ]
    dump(destination, record)
    return int(record["status"] != "accepted")


def main():
    """Execute all cold workers serially, preserving every attempt and its caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grids", type=int, nargs="+", default=list(GRIDS))
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    parser.add_argument("--cache", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_json:
        return worker(json.loads(args.worker_json), args.output, args.cache)
    if (
        not args.grids
        or any(n not in GRIDS for n in args.grids)
        or len(set(args.grids)) != len(args.grids)
        or args.passes < 1
    ):
        parser.error("Choose distinct Figure-11 grids and a positive pass count")
    points = conditions(tuple(args.grids), args.passes)
    if args.dry_run:
        print(json.dumps(dict(conditions=len(points), points=points), indent=2))
        return 0
    output = args.output.resolve()
    if output.exists() and not args.resume:
        parser.error("Output exists; use a new directory or --resume")
    if not output.exists():
        output.mkdir(parents=True)
        for folder in ("records", "logs", "caches", "source"):
            (output / folder).mkdir()
        manifest = source_manifest()
        dump(output / "source_sha256.json", manifest)
        for relative in manifest:
            copied = output / "source" / relative
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, copied)
        dump(
            output / "protocol.json",
            dict(
                case="public",
                points=points,
                passes=args.passes,
                eta=1e-6,
                source="Figure 11 native CuFHom grids and precisions; new cold calls only",
                warm_calls=0,
                timing="Figure-11 historical descriptor-excluded clock",
                driver_cache="retained",
                application_cache="empty and private per worker",
                analysis="no cold-time interpretation or cold-minus-first analysis",
            ),
        )
    else:
        if json.loads((output / "protocol.json").read_text())["points"] != points:
            parser.error("Resume protocol differs from preserved conditions")
        saved_sources = json.loads((output / "source_sha256.json").read_text())
        saved_sources.pop("nonfinite_native_metadata_encoded_as_null", None)
        if saved_sources != source_manifest():
            parser.error(
                "Worktree solver/runner/input sources changed since campaign start"
            )
    failed, accepted = [], []
    for index, point in enumerate(points, 1):
        stem = name(point)
        previous = sorted((output / "records").glob(stem + "_attempt*.json"))
        valid = [
            p for p in previous if json.loads(p.read_text())["status"] == "accepted"
        ]
        if valid:
            accepted.append(str(valid[0].relative_to(output)))
            continue
        attempt = len(previous) + 1
        identifier = f"{stem}_attempt{attempt:02d}"
        destination = output / "records" / (identifier + ".json")
        print(f"[{index}/{len(points)}] {identifier}", flush=True)
        command = [
            sys.executable,
            "-B",
            "-m",
            "benchmarks.paper.cold_scaling",
            "--worker-json",
            json.dumps(point),
            "--cache",
            str(output / "caches" / identifier),
            "--output",
            str(destination),
        ]
        with (output / "logs" / (identifier + ".log")).open("x") as log:
            run = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if (
            run.returncode == 0
            and destination.exists()
            and json.loads(destination.read_text())["status"] == "accepted"
        ):
            accepted.append(str(destination.relative_to(output)))
            print(
                f"  accepted {json.loads(destination.read_text())['time_s']:.6f} s",
                flush=True,
            )
        else:
            failed.append(identifier)
            print(
                f"  failed: see logs/{identifier}.log and retained record", flush=True
            )
        dump(
            output / "state.json",
            dict(
                status="running",
                completed=index,
                total=len(points),
                accepted_records=accepted,
                failed_attempts_this_invocation=failed,
                updated_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
    dump(
        output / "state.json",
        dict(
            status="complete" if len(accepted) == len(points) else "incomplete",
            completed=len(points),
            total=len(points),
            accepted_records=accepted,
            failed_attempts_this_invocation=failed,
            updated_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    print(
        f"Accepted {len(accepted)}/{len(points)} cold calls; data in {output}",
        flush=True,
    )
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
