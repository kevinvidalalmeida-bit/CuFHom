"""Run isolated, explicitly timed CuFHom paper campaigns from JSON protocols."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def plain(value):
    """Convert numerical metadata into strict JSON-compatible objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def strict_record(record):
    """Encode unavailable native-history values as null, recording their paths."""
    unavailable = []

    def convert(value, path):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            unavailable.append(path)
            return None
        if isinstance(value, dict):
            return {key: convert(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value

    result = convert(record, "record")
    result["nonfinite_native_metadata_encoded_as_null"] = unavailable
    return result


def digest(value):
    """Hash an array including its shape and dtype."""
    array = np.ascontiguousarray(value)
    header = json.dumps([array.shape, array.dtype.str]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def isotropic(young, poisson):
    """Build isotropic stiffness in unit-shear Mandel coordinates."""
    shear = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    matrix = 2 * shear * np.eye(6)
    matrix[:3, :3] += lame
    return matrix


def environment():
    """Capture the software revision, CPU, affinity and thread settings."""

    def command(args):
        try:
            run = subprocess.run(args, capture_output=True, text=True, timeout=15)
            return {"returncode": run.returncode, "stdout": run.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"unavailable": str(error)}

    versions = {}
    for name in ("cufhom", "numpy", "scipy", "numba", "cupy-cuda12x", "cupy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": command(["lscpu"]),
        "host_memory": host_memory(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "threads_and_caches": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUFHOM_FUSED_COMPILE",
                "CUPY_CACHE_DIR",
                "CUFHOM_KERNEL_CACHE",
            )
        },
        "revision": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "dirty": command(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]
        ),
        "versions": versions,
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv",
            ]
        ),
        "sources_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (ROOT / "src", ROOT / "core", HERE)
            for path in sorted(directory.rglob("*.py"))
        },
    }


def host_memory():
    """Read Linux host capacity and process high-water memory, in bytes.

    VmHWM is the process-lifetime resident-set high-water mark, not an isolated
    call peak. VmLck reports locked pages, not the size of every CUDA host buffer.
    Missing procfs measurements remain unavailable instead of being estimated.
    """
    measurements = {}
    for filename, names in (
        ("/proc/meminfo", ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")),
        ("/proc/self/status", ("VmRSS", "VmHWM", "VmLck")),
    ):
        try:
            for line in Path(filename).read_text().splitlines():
                name, _, value = line.partition(":")
                if name in names:
                    measurements[name + "_bytes"] = int(value.split()[0]) * 1024
        except (OSError, ValueError) as error:
            measurements[filename + "_unavailable"] = type(error).__name__
    measurements["scope"] = (
        "host capacity/availability and current-process RSS; VmHWM covers "
        "the process lifetime, including input preparation and prior calls; "
        "not an isolated per-call peak or total host allocation"
    )
    return measurements


class MemorySampler:
    """Sample global device-memory use; this is not a process peak measurement."""

    def __init__(self, cupy):
        self.cp = cupy
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        """Start periodic sampling when a CUDA backend is available."""
        if self.cp is not None:
            device = self.cp.cuda.Device().id
            self.baseline = self.cp.cuda.runtime.memGetInfo()[0]

            def sample():
                self.cp.cuda.Device(device).use()
                while not self.stop.is_set():
                    self.samples.append(self.cp.cuda.runtime.memGetInfo()[0])
                    self.stop.wait(0.02)

            self.thread = threading.Thread(target=sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *unused):
        """Stop the sampler and wait for its background thread."""
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def report(self):
        """Return observed increments and sampling limitations."""
        if self.cp is None:
            return None
        pool = self.cp.get_default_memory_pool()
        return {
            "sampled_device_increment_bytes": max(
                0, self.baseline - min(self.samples, default=self.baseline)
            ),
            "samples": len(self.samples),
            "interval_requested_s": 0.02,
            "pool_active_after_bytes": pool.used_bytes(),
            "pool_reserved_after_bytes": pool.total_bytes(),
            "scope": "global device use relative to pre-call free memory; sampled, not total process peak; sampling may perturb timing",
        }


def point_inputs(point):
    """Prepare deterministic geometry and optional public-image contrast."""
    if __package__:
        from .cases import prepare_case
    else:
        from cases import prepare_case

    index, table, metadata = prepare_case(point["case"], point["n"])
    contrast = point["contrast"]
    if contrast is not None:
        if point["case"] != "public":
            raise ValueError(
                "Contrast override is defined only for the public two-phase image"
            )
        table = np.stack([isotropic(1, 0.3), isotropic(contrast, 0.3)])
        metadata = dict(metadata, constituents={"E": [1, contrast], "nu": [0.3, 0.3]})
    return index, table, metadata


def worker(point, destination):
    """Resolve one independent process, preserving failures and all six audits."""
    from cufhom.fft import IndexedMaterial, homogenize
    from cufhom.fft.linear.common import isotropic_reference

    if destination.exists():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    record = {"point": point, "environment": environment(), "status": "running"}
    try:
        index, table, metadata = point_inputs(point)
        dtype = "float32" if point["precision"] == "fp32" else "float64"
        working = "float64" if point["precision"] == "fp64" else "float32"
        if point["backend"] == "numpy" and point["precision"] != "fp64":
            raise ValueError("CPU smoke uses FP64; mixed precision requires CUDA")
        cupy = None
        if point["backend"] == "cupy":
            import cupy

            cupy.zeros(1).sum()
            cupy.cuda.Device().synchronize()
        descriptor = IndexedMaterial(index, table, dtype=dtype)
        nominal_descriptor = IndexedMaterial(index, table, dtype="float64")
        reference, bulk, shear = isotropic_reference(nominal_descriptor.stiffness)
        if point["nu0"] is not None:
            nu0 = point["nu0"]
            young0 = 2 * shear * (1 + nu0)
            reference = isotropic(young0, nu0)
        record.update(
            inputs=metadata,
            input_hashes={
                "index": digest(index),
                "stiffness_nominal": digest(table),
                "stiffness_solver": digest(descriptor.stiffness),
            },
            reference={
                "matrix": reference,
                "mu0": shear,
                "initial_bulk0": bulk,
                "nu0_requested": point["nu0"],
                "policy": "isotropic arithmetic mean of nominal FP64 phase matrices; shear fixed within each contrast and across precisions",
            },
            timing_boundary="imports/context and input loading, reference selection excluded; fresh material descriptor construction, six solves from zero, native fresh audits and corrected assembly included; synchronized GPU; no disk writing",
            calls=[],
        )

        def call(label, eta):
            host_before = host_memory()
            if cupy is not None:
                cupy.cuda.Device().synchronize()
            with MemorySampler(cupy) as sampler:
                begin = time.perf_counter()
                material = IndexedMaterial(index, table, dtype=dtype)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = homogenize(
                        material,
                        discretization=point["discretization"],
                        backend=point["backend"],
                        working=working,
                        reference=reference,
                        rtol=eta,
                        evaluation="df64",
                        maxiter=3000,
                        atol=0.0,
                    )
                if cupy is not None:
                    cupy.cuda.Device().synchronize()
                elapsed = time.perf_counter() - begin
            if len(result["audits"]) != 6 or not all(
                a["accepted"] for a in result["audits"]
            ):
                raise RuntimeError("Six fresh physical acceptance audits required")
            return {
                "state": label,
                "eta": eta,
                "time_s": elapsed,
                "iterations": result["iterations"],
                "mean_updates": float(np.mean(result["iterations"])),
                "audits": result["audits"],
                "C_corrected": result["C"],
                "C_direct": result.get("C_stress"),
                "C_symmetrized": None
                if result.get("C_stress") is None
                else 0.5 * (result["C_stress"] + result["C_stress"].T),
                "execution": result["execution"],
                "residency": result.get("memory"),
                "precision_contract": result["precision_contract"],
                "memory_measurement": sampler.report(),
                "host_memory_before": host_before,
                "host_memory_after": host_memory(),
            }

        # First-call timing precedes any warm-up or tight reference solve.
        record["calls"].append(call("first", point["eta"]))
        for _ in range(point["warm_calls"]):
            record["calls"].append(call("warm", point["eta"]))
        if point["reference_eta"] is not None:
            # Reference always uses FP64 coefficients, Krylov arithmetic and evaluation.
            ref_material = IndexedMaterial(index, table, dtype="float64")
            with contextlib.redirect_stdout(io.StringIO()):
                tight = homogenize(
                    ref_material,
                    discretization=point["discretization"],
                    backend=point["backend"],
                    working="float64",
                    reference=reference,
                    rtol=point["reference_eta"],
                )
            record["tensor_reference"] = {
                "eta": point["reference_eta"],
                "C": tight["C"],
                "audits": tight["audits"],
                "scope": "same grid, nominal FP64 coefficients, same formulation; algebraic reference, not continuum error",
            }
            denominator = np.linalg.norm(tight["C"])
            for item in record["calls"]:
                item["error_corrected_fro"] = float(
                    np.linalg.norm(item["C_corrected"] - tight["C"]) / denominator
                )
                if item["C_direct"] is not None:
                    item["error_direct_fro"] = float(
                        np.linalg.norm(item["C_direct"] - tight["C"]) / denominator
                    )
                    item["error_symmetrized_fro"] = float(
                        np.linalg.norm(item["C_symmetrized"] - tight["C"]) / denominator
                    )
        record["status"] = "accepted"
    except Exception as error:
        record.update(
            status="failed", error=repr(error), traceback=traceback.format_exc()
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(
            strict_record(record), stream, default=plain, indent=2, allow_nan=False
        )
    return 0 if record["status"] == "accepted" else 1


def run_convergence(
    case,
    grids=(16, 24, 32, 48, 64, 96, 128, 192),
    *,
    grid_policy="auto",
    precision="mixed",
    backend="cupy",
    repeats=3,
    reference_bulk=None,
    output=None,
):
    """Return ``(study, finest_section, output)`` after fresh six-load solves.

    Requested sizes are lower bounds; select_grid applies the policy before
    geometry sampling. Selected sizes are printed and saved. Repeated selected
    sizes are evaluated once. Timings synchronize CUDA and include material
    preparation, audits and tensor assembly, excluding voxels/imports/context.
    First calls are separate from warm repetitions. Output directories must
    be new, and partial results survive a failed solve.
    """
    import csv
    import shutil
    import sysconfig
    import tempfile

    from cufhom.fft import select_grid
    from cufhom.runtime import save_json

    from benchmarks.paper.cases import CASES, prepare_case

    target = 0.001
    if case not in CASES:
        raise ValueError(f"Unknown case {case!r}; choose from {CASES}")
    if reference_bulk is not None and (
        not np.isfinite(reference_bulk) or reference_bulk <= 0
    ):
        raise ValueError("reference_bulk must be finite and positive")
    if backend not in ("numpy", "cupy") or precision not in ("mixed", "fp32", "fp64"):
        raise ValueError("Choose numpy/cupy and fp32/fp64/mixed")
    if backend == "numpy" and precision == "mixed":
        raise ValueError("Mixed precision requires CUDA; use --precision fp64 for CPU")
    if repeats < 1 or isinstance(repeats, bool) or not isinstance(repeats, int):
        raise ValueError("repeats must be a positive integer")
    grids = tuple(grids)
    if len(grids) < 2 or list(grids) != sorted(set(grids)):
        raise ValueError("Choose at least two strictly increasing grid sizes")
    selected = {
        method: list(
            dict.fromkeys(
                select_grid(n, grid_policy, discretization=method) for n in grids
            )
        )
        for method in ("gani", "hex8")
    }
    if any(len(sizes) < 2 for sizes in selected.values()):
        raise ValueError("Choose sizes giving at least two distinct selected grids")
    if output is not None and Path(output).exists():
        raise ValueError(
            "Choose a new output directory; previous results are preserved"
        )
    # NVIDIA wheel-only setups need a header root before compiling CUDA kernels.
    runtime = Path(sysconfig.get_path("purelib")) / "nvidia/cuda_runtime"
    if (
        backend == "cupy"
        and not os.environ.get("CUDA_PATH")
        and not shutil.which("nvcc")
        and not Path("/usr/local/cuda").exists()
        and (runtime / "include/cuda_runtime.h").is_file()
    ):
        os.environ["CUDA_PATH"] = str(runtime)
    os.environ.setdefault("CUFHOM_FUSED_COMPILE", "sync")
    from cufhom.fft import homogenize

    if backend == "cupy":
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        device = device.decode() if isinstance(device, bytes) else device
        synchronize = cp.cuda.Device().synchronize
    else:
        device = "CPU / NumPy"

        def synchronize():
            """CPU calls finish synchronously."""

    if output is None:
        (Path("results") / case).mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="run-", dir=Path("results") / case))
    else:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
    study = {
        "status": "running",
        "backend": backend,
        "precision": precision,
        "case": case,
        "device": device,
        "repeats": repeats,
        "K_reference": reference_bulk,
        "reference_scope": "caller-provided bulk modulus only"
        if reference_bulk is not None
        else None,
        "grid_policy": grid_policy,
        "requested_grids": grids,
        "selected_grids": selected,
        "target_relative_error_and_last_change": target,
        "timing_scope": "synchronized full six-load calls from zero; material preparation, audits and assembly included; voxels, imports and CUDA context excluded; first call separate; warm ranges are not confidence intervals",
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("cufhom", "numpy", "scipy")
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "points": [],
    }
    if backend == "cupy":
        study["versions"]["cupy"] = cp.__version__
    for method, sizes in selected.items():
        print(
            f"{method}: requested {list(grids)} -> selected {sizes} ({grid_policy})",
            flush=True,
        )
    section = None
    save_json(output / "results.json", strict_record(study))
    try:
        for level in range(max(map(len, selected.values()))):
            for method in ("gani", "hex8") if level % 2 == 0 else ("hex8", "gani"):
                if level >= len(selected[method]):
                    continue
                n = selected[method][level]
                labels, table, metadata = prepare_case(case, n)
                calls = []
                for repetition in range(repeats + 1):
                    synchronize()
                    start = time.perf_counter()
                    result = homogenize(
                        labels,
                        table,
                        discretization=method,
                        backend=backend,
                        precision=precision,
                    )
                    synchronize()
                    elapsed = time.perf_counter() - start
                    if len(result["audits"]) != 6 or not all(
                        a["accepted"] for a in result["audits"]
                    ):
                        raise RuntimeError(
                            f"{method}: all six residual checks must pass"
                        )
                    if not np.isfinite(result["C"]).all():
                        raise RuntimeError(f"{method}: nonfinite effective stiffness")
                    if calls and not np.allclose(
                        result["C"], calls[0]["C"], rtol=1e-7, atol=1e-10
                    ):
                        raise RuntimeError(
                            "Repeated calls produced inconsistent tensors"
                        )
                    calls.append(
                        {
                            "state": "first" if repetition == 0 else "warm",
                            "time_s": elapsed,
                            "C": result["C"],
                            "iterations": result["iterations"],
                            "audits": result["audits"],
                        }
                    )
                bulk = float(calls[0]["C"][:3, :3].sum() / 9)
                warm = [c["time_s"] for c in calls[1:]]
                previous = [p for p in study["points"] if p["method"] == method]
                point = {
                    "method": method,
                    "n": n,
                    "K": bulk,
                    "relative_error": abs(bulk / reference_bulk - 1)
                    if reference_bulk is not None
                    else None,
                    "relative_mesh_change": abs(bulk / previous[-1]["K"] - 1)
                    if previous
                    else None,
                    "first_s": calls[0]["time_s"],
                    "warm_median_s": float(np.median(warm)),
                    "warm_min_s": min(warm),
                    "warm_max_s": max(warm),
                    "execution": result["execution"],
                    "precision_contract": result["precision_contract"],
                    "phase_fractions": np.bincount(labels.ravel(), minlength=3)
                    / labels.size,
                    "inputs": metadata,
                    "calls": calls,
                }
                study["points"].append(point)
                if section is None or n > section.shape[0]:
                    section = labels[:, :, n // 2].copy()
                save_json(output / "results.json", strict_record(study))
                print(
                    f"{method:4s} {n:3d}³  K={bulk:.10f}  warm={point['warm_median_s']:.4f} s",
                    flush=True,
                )
        study["mesh_check"] = {}
        for method in selected:
            p = next(p for p in reversed(study["points"]) if p["method"] == method)
            study["mesh_check"][method] = {
                "finest_n": p["n"],
                "relative_error": p["relative_error"],
                "relative_last_change": p["relative_mesh_change"],
                "target_met": max(p["relative_error"] or 0, p["relative_mesh_change"])
                <= target,
                "scope": "observed finest-mesh check, not a proof for all finer meshes",
            }
        columns = (
            "method",
            "n",
            "K",
            "relative_error",
            "relative_mesh_change",
            "first_s",
            "warm_median_s",
            "warm_min_s",
            "warm_max_s",
        )
        with (output / "convergence.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows({k: p[k] for k in columns} for p in study["points"])
        study["status"] = "completed"
        save_json(output / "results.json", strict_record(study))
    except Exception as error:
        study.update(status="failed", error=f"{type(error).__name__}: {error}")
        save_json(output / "results.json", strict_record(study))
        raise
    return study, section, output


def plot_convergence(study, section, destination, *, title=None, phase_labels=None):
    """Save one scientific figure using the measured data without smoothing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    phase_count = len(phase_labels) if phase_labels else int(section.max()) + 1
    phases = (
        ("#E6E8EB", "#92C5C8", "#344966")
        if phase_count == 3
        else plt.get_cmap("tab20", phase_count).colors
    )
    phase_labels = phase_labels or [f"Phase {n}" for n in range(phase_count)]
    field = (
        "relative_error" if study["K_reference"] is not None else "relative_mesh_change"
    )
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7,
        }
    ):
        fig, (geometry, error, timing) = plt.subplots(
            1, 3, figsize=(7.5, 3.1), width_ratios=(1, 1.18, 1.18)
        )
        fig.subplots_adjust(left=0.06, right=0.985, bottom=0.28, top=0.77, wspace=0.52)
        n = section.shape[0]
        geometry.imshow(
            section.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            interpolation="nearest",
            cmap=ListedColormap(phases),
            vmin=0,
            vmax=max(1, phase_count - 1),
        )
        geometry.set(
            xlabel="$x/L$",
            ylabel="$y/L$",
            xticks=(0, 0.5, 1),
            yticks=(0, 0.5, 1),
            title=f"(a) Voxel section, $N={n}$",
        )
        geometry.text(
            0.5,
            0.03,
            f"$z/L={(n // 2 + 0.5) / n:.4f}$",
            ha="center",
            transform=geometry.transAxes,
            bbox={"facecolor": "white", "edgecolor": "none"},
        )
        if phase_count <= 6:
            geometry.legend(
                handles=[
                    Patch(facecolor=c, label=label)
                    for c, label in zip(phases, phase_labels, strict=True)
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=3,
                frameon=False,
                columnspacing=0.55,
                handlelength=0.8,
            )
        for method, color, marker, label in (
            ("gani", "#0072B2", "o", "GaNi"),
            ("hex8", "#D55E00", "s", "HEX8"),
        ):
            points = [p for p in study["points"] if p["method"] == method]
            sizes = [p["n"] for p in points]
            shown = [p for p in points if p[field] is not None]
            error.plot(
                [p["n"] for p in shown],
                [100 * p[field] for p in shown],
                color=color,
                marker=marker,
                mfc="white",
                ms=3.5,
                lw=1.1,
                label=label,
            )
            timing.loglog(
                sizes,
                [p["warm_median_s"] for p in points],
                color=color,
                marker=marker,
                mfc="white",
                ms=3.5,
                lw=1.1,
            )
            timing.fill_between(
                sizes,
                [p["warm_min_s"] for p in points],
                [p["warm_max_s"] for p in points],
                color=color,
                alpha=0.12,
                lw=0,
            )
        target = 100 * study["target_relative_error_and_last_change"]
        error.set_xscale("log")
        if all(p[field] > 0 for p in study["points"] if p[field] is not None):
            error.set_yscale("log")
        error.axhline(target, color=".45", ls="--", lw=0.7, label=f"{target:g}% target")
        error.set(
            title="(b) Mesh convergence",
            ylabel="Bulk error (%)"
            if study["K_reference"] is not None
            else "Bulk change from previous mesh (%)",
        )
        timing.set(title="(c) Execution time", ylabel="Warm six-load time (s)")
        error.legend(frameon=False, loc="upper right")
        for ax in (error, timing):
            ax.set_xlabel("Grid resolution $N$")
            ax.grid(axis="y", color=".9", lw=0.5)
            ax.tick_params(labelsize=7)
            ax.set_xticks([], minor=True)
            ax.xaxis.set_major_locator(
                FixedLocator(
                    np.geomspace(
                        min(p["n"] for p in study["points"]),
                        max(p["n"] for p in study["points"]),
                        4,
                    ).round()
                )
            )
            ax.xaxis.set_major_formatter(ScalarFormatter())
        fig.suptitle(
            title or f"{study['case']}: bulk-modulus mesh study", fontsize=10, y=0.99
        )
        fig.text(
            0.5,
            0.105,
            f"{study['precision']} · {study['device']}"
            + (
                f" · $K_{{ref}}={study['K_reference']:.10f}$"
                if study["K_reference"] is not None
                else ""
            ),
            ha="center",
            fontsize=7,
        )
        fig.text(
            0.5,
            0.045,
            f"Times: median and min–max of {study['repeats']} warm six-load calls from zero; voxels excluded.\n"
            "Different discrete grids and native residual norms. Voxelization may give nonmonotone errors.",
            ha="center",
            va="center",
            fontsize=6.2,
        )
        fig.savefig(destination, dpi=600)
        plt.close(fig)


def expand(protocol, backend, small):
    """Expand explicit protocol combinations without implicit material changes."""
    validate_protocol(protocol)
    points = []
    fields = ("cases", "discretizations", "precisions", "etas", "contrasts", "nu0")
    for case, disc, precision, eta, contrast, nu0 in itertools.product(
        *(protocol[key] for key in fields)
    ):
        for grid in protocol["grids"][disc]:
            for session in range(1, protocol["sessions"] + 1):
                points.append(
                    dict(
                        case=case,
                        discretization=disc,
                        precision=precision,
                        eta=eta,
                        contrast=contrast,
                        nu0=nu0,
                        n=grid,
                        session=session,
                        backend=backend,
                        warm_calls=protocol["warm_calls"],
                        reference_eta=protocol.get("reference_eta"),
                    )
                )
    if small:
        points = [
            dict(
                point,
                n=9 if point["discretization"] == "gani" else 8,
                precision="fp64",
                session=1,
                warm_calls=0,
                reference_eta=None,
            )
            for point in points
            if point["session"] == 1 and point["precision"] == protocol["precisions"][0]
        ]
        # A smoke run checks one representative point per case/formulation.
        points = list({(p["case"], p["discretization"]): p for p in points}.values())
    return points


def validate_protocol(protocol):
    """Reject invalid tolerances, grids and unsupported material combinations."""
    if __package__:
        from .cases import CASES
    else:
        from cases import CASES

    for key in ("cases", "discretizations", "precisions", "etas", "contrasts", "nu0"):
        if not isinstance(protocol.get(key), list) or not protocol[key]:
            raise ValueError(f"{key} must be a nonempty list")
    for key, allowed in (
        ("cases", set(CASES)),
        ("discretizations", {"hex8", "gani"}),
        ("precisions", {"fp64", "mixed", "fp32"}),
    ):
        if not set(protocol[key]) <= allowed:
            raise ValueError(f"Unsupported {key}: {protocol[key]}")
    for key in ("sessions", "warm_calls"):
        value = protocol.get(key)
        if type(value) is not int or value < (1 if key == "sessions" else 0):
            raise ValueError(f"Invalid {key}")
    for value in protocol["etas"] + (
        [protocol["reference_eta"]] if protocol.get("reference_eta") is not None else []
    ):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("Tolerances must be positive and finite")
    for disc in protocol["discretizations"]:
        sizes = protocol.get("grids", {}).get(disc)
        if (
            not isinstance(sizes, list)
            or not sizes
            or any(type(n) is not int or n < 2 for n in sizes)
        ):
            raise ValueError(f"Invalid grids for {disc}")
    for value in protocol["contrasts"]:
        if value is not None and (
            isinstance(value, bool) or not np.isfinite(value) or value <= 0
        ):
            raise ValueError("Contrast must be positive and finite or null")
    if any(value is not None for value in protocol["contrasts"]) and protocol[
        "cases"
    ] != ["public"]:
        raise ValueError("Contrast overrides require the public image only")
    for value in protocol["nu0"]:
        if value is not None and (
            isinstance(value, bool) or not np.isfinite(value) or not -1 < value < 0.5
        ):
            raise ValueError("Reference Poisson ratios must lie in (-1, 0.5)")


def main():
    """Launch one fresh worker per condition and session; never overwrite data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocols" / "public.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "paper")
    parser.add_argument("--backend", choices=["cupy", "numpy"], default="cupy")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print all conditions without solving"
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="FP64 8^3/9^3 smoke only; not a paper benchmark",
    )
    parser.add_argument("--worker-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_json is not None:
        return worker(json.loads(args.worker_json), args.output)
    protocol = json.loads(args.protocol.read_text())
    points = expand(protocol, args.backend, args.small)
    if args.backend == "numpy" and any(
        point["precision"] != "fp64" for point in points
    ):
        parser.error(
            "CPU backend supports FP64 here; use --small or an FP64-only protocol"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "protocol": str(args.protocol),
                    "conditions": len(points),
                    "smoke_only": args.small,
                    "points": points,
                },
                indent=2,
            )
        )
        return 0
    if args.output.exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    failed = 0
    for number, point in enumerate(points, 1):
        print(f"[{number}/{len(points)}] {point}", flush=True)
        destination = args.output / f"point_{number:05d}.json"
        run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-json",
                json.dumps(point),
                "--output",
                str(destination),
            ],
            check=False,
        )
        failed += run.returncode != 0
    print(f"Saved {len(points)} conditions; {failed} failures in {args.output}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
