"""Run the six-cell comparison of the paper's Section 4.1 end to end.

Subcommands, from the repository root:

    python benchmarks/reproduce/section41.py check     # environment and GPU
    python benchmarks/reproduce/section41.py fetch     # pinned FANS/FFTHomPy
    python benchmarks/reproduce/section41.py run --mode quick
    python benchmarks/reproduce/section41.py table

CuFHom HEX8 (64^3-192^3) and GaNi (63^3-189^3) run in FP64 and mixed
precision at eta=1e-6 and in FP32 at eta=1e-5, through run.py. FANS (UNOCG,
GPU, FP64) and FFTHomPy (CPU, FP64/FP32) run through external.py. Every
condition and session is a separate process. A run can be interrupted and
started again with the same command: finished conditions are kept and skipped,
incomplete ones are renamed, never overwritten.

Before each measured CuFHom condition an untabulated priming run (one call, no
reference, saved under ``prime/``) fills the CuPy and CuFHom kernel caches, so
first calls are timed as in repeated sessions; ``--no-prime`` disables it.

Modes: ``quick`` runs the largest grids once (one warm call); ``table1`` the
largest grids in three sessions, as Table 1; ``full`` adds 64^3/63^3 and
128^3/125^3, as the six-cell protocols. ``--grids 128 192`` overrides the grids
of a mode (useful on GPUs whose memory cannot hold FANS at 192^3). Regenerated times are new measurements
on the local hardware; they are not expected to equal the published ones.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CELLS = (
    "particles",
    "polycrystal",
    "gyroid",
    "random_short_fibres",
    "laminate",
    "hashin",
)
GRIDS = {"hex8": (64, 128, 192), "gani": (63, 125, 189)}
ETA = {"fp64": 1e-6, "mixed": 1e-6, "fp32": 1e-5}
REFERENCE_ETA = 1e-10
MODES = {
    # name: (grid indices, sessions, CuFHom warm calls)
    "quick": ((2,), 1, 1),
    "table1": ((2,), 3, 3),
    "full": ((0, 1, 2), 3, 3),
}
EXTERNAL = {
    "UNOCG": (
        "https://github.com/DataAnalyticsEngineering/UNOCG.git",
        "461f74f7a40f1c6c6d8ac053f6130428650a0133",
        "unocg",
    ),
    "FFTHomPy": (
        "https://github.com/vondrejc/FFTHomPy.git",
        "2c23c80dd2cab46f1090103e613b4f886b3daac7",
        "ffthompy",
    ),
}
PAPER_CUPY = "13.6.0"


def child_environment():
    """Environment for solver processes: synchronous compiles, pip CUDA libraries."""
    env = dict(os.environ, CUFHOM_FUSED_COMPILE="sync")
    if sys.platform.startswith("linux"):
        import site

        roots = site.getsitepackages() + [site.getusersitepackages()]
        libs = [str(p) for r in roots for p in sorted(Path(r).glob("nvidia/*/lib"))]
        if libs:
            old = env.get("LD_LIBRARY_PATH")
            env["LD_LIBRARY_PATH"] = ":".join(libs + ([old] if old else []))
    return env


def reexec_with_libraries():
    """Restart once so the dynamic loader sees pip-installed CUDA libraries."""
    if os.environ.get("_CUFHOM_SECTION41_ENV") == "1":
        return
    env = child_environment()
    env["_CUFHOM_SECTION41_ENV"] = "1"
    if env.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH"):
        sys.stdout.flush()
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


# ---------------------------------------------------------------- check


def _gib(value):
    return f"{value / 2**30:.2f} GiB"


def host_ram():
    """Total and available host memory in bytes (Linux), else None."""
    try:
        fields = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        return tuple(
            int(fields[k].split()[0]) * 1024 for k in ("MemTotal", "MemAvailable")
        )
    except (OSError, KeyError, ValueError):
        return None


def check(args):
    """Report every requirement; return nonzero when a selected solver cannot run."""
    problems, notes = [], []
    print(f"Python      {sys.version.split()[0]}  ({platform.platform()})")
    ram = host_ram()
    if ram:
        print(f"Host RAM    {_gib(ram[0])} total, {_gib(ram[1])} available")
        if ram[0] < 24 * 2**30:
            notes.append(
                "FFTHomPy runs on the CPU with its fields in host RAM: about 2.3 GiB "
                "peak at 95^3 FP64, i.e. roughly 5 GiB at 125^3 and 18 GiB at "
                "189^3. With less RAM the largest grid may swap or be killed; the "
                "failure is recorded and the campaign continues."
            )
    try:
        import cupy as cp

        print(f"CuPy        {cp.__version__}")
        wheels = {}
        for package in ("nvidia-cuda-nvrtc-cu12", "nvidia-cuda-runtime-cu12"):
            try:
                wheels[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        if wheels:
            print("CUDA wheels " + ", ".join(f"{k} {v}" for k, v in wheels.items()))
        if len({".".join(v.split(".")[:2]) for v in wheels.values()}) > 1:
            problems.append(
                "nvidia-cuda-nvrtc-cu12 and nvidia-cuda-runtime-cu12 differ in "
                "major.minor; CuPy then cannot find the CUDA headers. Install the "
                "same 12.x release of both."
            )
        if cp.__version__ != PAPER_CUPY:
            msg = f"CuPy {cp.__version__}; the paper used {PAPER_CUPY}"
            if int(cp.__version__.split(".")[0]) >= 14:
                problems.append(
                    msg + ". CuPy 14 changes NumPy-2 type promotion and fails part "
                    "of the GPU suite: pip install 'cupy-cuda12x==13.6.0'"
                )
            else:
                notes.append(msg)
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = (
            props["name"].decode()
            if isinstance(props["name"], bytes)
            else props["name"]
        )
        cc = (props["major"], props["minor"])
        free, total = cp.cuda.runtime.memGetInfo()
        print(f"GPU         {name}, compute capability {cc[0]}.{cc[1]}")
        print(f"GPU memory  {_gib(total)} total, {_gib(free)} free")
        print(
            f"CUDA        driver {cp.cuda.runtime.driverGetVersion()}, "
            f"runtime {cp.cuda.runtime.runtimeGetVersion()}, "
            f"NVRTC {'.'.join(map(str, cp.cuda.nvrtc.getVersion()))}"
        )
        print(
            f"Shared mem  {props['sharedMemPerBlockOptin'] // 1024} KiB/block (opt-in)"
        )
        nvrtc = cp.cuda.nvrtc.getVersion()
        if cc >= (10, 0) and nvrtc < (12, 8):
            problems.append(
                f"NVRTC {nvrtc[0]}.{nvrtc[1]} cannot compile for compute capability "
                f"{cc[0]}.{cc[1]}; install the CUDA 12.8 wheels (see README)"
            )
        if cc < (6, 0):
            problems.append("CuFHom needs compute capability >= 6.0 (FP64 atomics)")
        if cp.cuda.runtime.runtimeGetVersion() >= 13000 and cc < (7, 5):
            problems.append("CUDA 13 does not support this GPU; use cupy-cuda12x")
        if total < 6 * 2**30:
            notes.append(
                "Small GPU memory: CuFHom selects host residency automatically when "
                "fields do not fit. FANS keeps every field on the GPU; at 192^3 it "
                "can run out of memory (the polycrystal stores 36 coefficient "
                "fields). Failures are recorded and the campaign continues."
            )
        if "GeForce" in name:
            notes.append(
                "Consumer GPUs have low FP64 throughput: FP64 and the FP64 "
                "reference solves are much slower than mixed/FP32."
            )
        kernel = cp.RawKernel(
            r'extern "C" __global__ void k(double* x){atomicAdd(x, 1.0);}', "k"
        )
        x = cp.zeros(1)
        kernel((4,), (32,), (x,))
        assert float(x[0]) == 128.0
        print("NVRTC       kernel compiled and executed for this GPU")
        smoke_gpu()
    except Exception as error:  # noqa: BLE001
        problems.append(f"CuFHom GPU check failed: {error!r}")

    if "fans" in args.solvers:
        try:
            import torch

            ok = torch.cuda.is_available()
            print(
                f"PyTorch     {torch.__version__}, CUDA {torch.version.cuda}, available={ok}"
            )
            if ok:
                major, minor = torch.cuda.get_device_capability(0)
                arches = torch.cuda.get_arch_list()
                # A cubin for sm_XY runs on sm_XZ with Z >= Y (same major).
                built = [int(a[3:].rstrip("af")) for a in arches if a.startswith("sm_")]
                compatible = any(cc // 10 == major and cc % 10 <= minor for cc in built)
                print(f"            built for {' '.join(arches)}")
                if not compatible:
                    wheel = "cu126" if major < 7 else "cu128"
                    problems.append(
                        f"PyTorch has no kernels for sm_{major}{minor}. Install the "
                        f"{wheel} build: pip install torch==2.10.0 --index-url "
                        f"https://download.pytorch.org/whl/{wheel} (cu126 supports "
                        "Maxwell-Hopper, cu128 Volta-Blackwell)"
                    )
                else:
                    a = torch.ones(64, device="cuda", dtype=torch.float64)
                    assert float(torch.fft.fftn(a).real[0]) == 64.0
            else:
                problems.append("PyTorch cannot see a CUDA device (FANS needs it)")
            for module in ("h5py", "einops", "tqdm", "matplotlib"):
                __import__(module)
        except Exception as error:  # noqa: BLE001
            problems.append(f"FANS dependencies: {error!r}")
    for key, (_, commit, package) in EXTERNAL.items():
        solver = "fans" if key == "UNOCG" else "ffthompy"
        if solver not in args.solvers:
            continue
        path = args.external / key
        if not (path / package / "__init__.py").is_file():
            problems.append(f"{path} missing: run 'section41.py fetch'")
            continue
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        state = "pinned" if head == commit else f"NOT pinned ({head[:12]})"
        print(f"{key:<11} {path} {state}")
        if head != commit:
            problems.append(f"{key} is not at commit {commit}")
    for note in notes:
        print(f"note: {note}")
    for problem in problems:
        print(f"PROBLEM: {problem}")
    print("Environment ready." if not problems else f"{len(problems)} problem(s).")
    return int(bool(problems))


def smoke_gpu():
    """Solve a small six-load cell with both formulations in all three precisions."""
    sys.path.insert(0, str(HERE))
    from cases import prepare_case
    from cufhom.fft import IndexedMaterial, homogenize

    for disc, n in (("hex8", 16), ("gani", 15)):
        index, table, _ = prepare_case("polycrystal", n)
        for precision, working in (
            ("fp64", "float64"),
            ("mixed", "float32"),
            ("fp32", "float32"),
        ):
            material = IndexedMaterial(
                index, table, dtype="float32" if precision == "fp32" else "float64"
            )
            result = homogenize(
                material,
                discretization=disc,
                backend="cupy",
                working=working,
                rtol=ETA[precision],
                evaluation="df64",
                atol=0.0,
            )
            accepted = sum(a["accepted"] for a in result["audits"])
            if accepted != 6:
                raise RuntimeError(f"{disc} {precision}: {accepted}/6 loads accepted")
    print("CuFHom      HEX8 16^3 and GaNi 15^3 solved on the GPU (FP64, mixed, FP32)")


# ---------------------------------------------------------------- fetch


def fetch(args):
    """Clone the pinned upstream sources (no UNO model weights)."""
    args.external.mkdir(parents=True, exist_ok=True)
    for key, (url, commit, package) in EXTERNAL.items():
        path = args.external / key
        if not path.exists():
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--sparse", url, str(path)],
                check=True,
            )
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "origin", commit])
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--quiet", commit], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "sparse-checkout", "set", package], check=True
        )
        print(f"{key}: {path} at {commit}")
    return 0


# ---------------------------------------------------------------- run


def plan(args):
    """Ordered work units: CuFHom first (GPU), then FFTHomPy (CPU), then FANS."""
    indices, sessions, warm = MODES[args.mode]
    if args.grids:
        indices = tuple(GRIDS["hex8"].index(n) for n in sorted(set(args.grids)))
    sessions = args.sessions or sessions
    units = []
    for solver in ("cufhom", "ffthompy", "fans"):
        if solver not in args.solvers:
            continue
        for i in indices:
            for cell in args.cells:
                if solver == "cufhom":
                    for disc in ("hex8", "gani"):
                        for precision in ("fp64", "mixed", "fp32"):
                            unit = dict(
                                solver=solver,
                                cell=cell,
                                disc=disc,
                                precision=precision,
                                n=GRIDS[disc][i],
                                sessions=sessions,
                                warm=warm,
                            )
                            if args.prime:
                                # Fills the CuPy/CuFHom kernel caches so that the
                                # measured first calls do not include compilation.
                                units.append(
                                    dict(unit, solver="prime", sessions=1, warm=0)
                                )
                            units.append(unit)
                    continue
                disc = "hex8" if solver == "fans" else "gani"
                for precision in ("fp64",) if solver == "fans" else ("fp64", "fp32"):
                    for session in range(1, sessions + 1):
                        units.append(
                            dict(
                                solver=solver,
                                cell=cell,
                                disc=disc,
                                precision=precision,
                                n=GRIDS[disc][i],
                                session=session,
                            )
                        )
    return units


def unit_path(output, unit):
    """Output directory (CuFHom) or JSON file (external) of one work unit."""
    stem = f"{unit['cell']}_{unit['disc']}_{unit['precision']}_n{unit['n']}"
    if unit["solver"] in ("cufhom", "prime"):
        return output / unit["solver"] / stem
    return output / unit["solver"] / f"{stem}_s{unit['session']}.json"


def finished(path, unit):
    """Return whether the unit wrote its record (CuFHom: one per session)."""
    if unit["solver"] not in ("cufhom", "prime"):
        return path.exists()
    return len(list(path.glob("point_*.json"))) == unit["sessions"]


def command(args, unit, path):
    """Process command for one unit."""
    if unit["solver"] in ("cufhom", "prime"):
        protocol = dict(
            cases=[unit["cell"]],
            discretizations=[unit["disc"]],
            precisions=[unit["precision"]],
            etas=[ETA[unit["precision"]]],
            contrasts=[None],
            nu0=[None],
            grids={unit["disc"]: [unit["n"]]},
            sessions=unit["sessions"],
            warm_calls=unit["warm"],
            reference_eta=REFERENCE_ETA if unit["solver"] == "cufhom" else None,
        )
        protocol_file = path.parent / f".{path.name}.protocol.json"
        protocol_file.parent.mkdir(parents=True, exist_ok=True)
        protocol_file.write_text(json.dumps(protocol, indent=2))
        return [
            sys.executable,
            str(HERE / "run.py"),
            "--protocol",
            str(protocol_file),
            "--output",
            str(path),
        ]
    source = args.external / ("UNOCG" if unit["solver"] == "fans" else "FFTHomPy")
    return [
        sys.executable,
        "-m",
        "benchmarks.reproduce.external",
        unit["solver"],
        "--case",
        unit["cell"],
        "--grid",
        str(unit["n"]),
        "--precision",
        unit["precision"],
        "--eta",
        str(ETA[unit["precision"]]),
        "--source",
        str(source),
        "--warm",
        "0",
        "--output",
        str(path),
    ]


def run(args):
    """Execute pending units sequentially; keep failures, never overwrite."""
    units = plan(args)
    pending = [u for u in units if not finished(unit_path(args.output, u), u)]
    print(
        f"{len(units)} units in mode '{args.mode}', {len(pending)} pending -> {args.output}"
    )
    if args.dry_run:
        for unit in pending:
            print(json.dumps(unit))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "section41_log.jsonl"
    env = child_environment()
    for number, unit in enumerate(pending, 1):
        path = unit_path(args.output, unit)
        if path.exists():  # interrupted CuFHom unit: keep it aside
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path.rename(path.with_name(f"{path.name}.incomplete-{stamp}"))
        print(f"[{number}/{len(pending)}] {json.dumps(unit)}", flush=True)
        begin = time.time()
        code = subprocess.run(command(args, unit, path), cwd=ROOT, env=env).returncode
        with log.open("a") as stream:
            stream.write(
                json.dumps(dict(unit, returncode=code, wall_s=time.time() - begin))
                + "\n"
            )
    print("Done. Summarize with: python benchmarks/reproduce/section41.py table")
    return 0


# ---------------------------------------------------------------- table


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def collect(output):
    """First-call times, warm medians and errors of every accepted record."""
    own, external, failures = {}, {}, []
    for path in sorted((output / "cufhom").glob("*/point_*.json")):
        if ".incomplete-" in path.parent.name:
            continue
        record = _read(path)
        if record is None:
            continue
        p = record["point"]
        key = (p["case"], p["discretization"], p["precision"], p["n"])
        if record.get("status") != "accepted":
            failures.append(
                f"CuFHom {key}: {record.get('error', record.get('status'))}"
            )
            continue
        first = record["calls"][0]
        warm = [c["time_s"] for c in record["calls"][1:]]
        own.setdefault(key, []).append(
            dict(
                first=first["time_s"],
                warm=statistics.median(warm) if warm else None,
                error=first.get("error_corrected_fro"),
                error_direct=first.get("error_direct_fro"),
                updates=first.get("mean_updates"),
                gpu=record["environment"]["gpu"].get("stdout", "").splitlines()[-1:],
            )
        )
    for solver in ("fans", "ffthompy"):
        for path in sorted((output / solver).glob("*.json")):
            record = _read(path)
            if record is None:
                continue
            cell, rest = path.stem.split("_hex8_" if solver == "fans" else "_gani_")
            precision, n = rest.split("_")[0], int(rest.split("_")[1][1:])
            key = (solver, cell, precision, n)
            if record.get("status") != "accepted":
                failures.append(
                    f"{solver} {cell} {precision} n={n}: "
                    f"{record.get('error', record.get('status'))}"
                )
                continue
            external.setdefault(key, []).append(record["calls"][0]["audited_call_s"])
    return own, external, failures


def table(args):
    """Write section41_table.csv/.md with medians and speed-ups (as Table 1)."""
    own, external, failures = collect(args.output)
    if not own and not external:
        print(f"No records under {args.output}")
        return 1
    rows = []
    order = {"hex8": 0, "gani": 1, "fp64": 0, "mixed": 1, "fp32": 2}
    for (cell, disc, precision, n), items in sorted(
        own.items(),
        key=lambda kv: (
            CELLS.index(kv[0][0]),
            order[kv[0][1]],
            kv[0][3],
            order[kv[0][2]],
        ),
    ):
        solver = "fans" if disc == "hex8" else "ffthompy"
        base_precision = "fp32" if precision == "fp32" else "fp64"
        if solver == "fans" and precision == "fp32":
            base = []  # FANS FP32 is excluded, as in the paper
        else:
            base = external.get((solver, cell, base_precision, n), [])
        t_own = statistics.median(i["first"] for i in items)
        t_ext = statistics.median(base) if base else None
        warm = [i["warm"] for i in items if i["warm"] is not None]
        rows.append(
            dict(
                cell=cell,
                route=disc.upper() if disc == "hex8" else "GaNi",
                n=n,
                precision=precision,
                eta=ETA[precision],
                sessions=len(items),
                cufhom_first_s=round(t_own, 3),
                cufhom_warm_s=round(statistics.median(warm), 3) if warm else "",
                updates_mean=round(statistics.median(i["updates"] for i in items), 1),
                error_corrected=f"{statistics.median(i['error'] for i in items):.2e}"
                if all(i["error"] is not None for i in items)
                else "",
                error_direct=f"{statistics.median(i['error_direct'] for i in items):.2e}"
                if all(i["error_direct"] is not None for i in items)
                else "",
                comparator=f"{'FANS' if solver == 'fans' else 'FFTHomPy'} {base_precision}",
                comparator_s=round(t_ext, 2) if t_ext else "",
                comparator_sessions=len(base),
                speedup=round(t_ext / t_own, 1) if t_ext else "",
            )
        )
    csv_path = args.output / "section41_table.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gpus = sorted({g for items in own.values() for i in items for g in i["gpu"]})
    header = (
        "| Cell | Route | n | Mode | CuFHom first (s) | warm (s) | e_corr | e_direct "
        "| Comparator | Comparator (s) | Speed-up |\n|" + "---|" * 11
    )
    lines = [
        f"# Section 4.1 regenerated ({args.output})",
        "",
        f"GPU: {'; '.join(gpus) or 'unknown'}. Medians over accepted sessions; "
        "speed-up = comparator first call / CuFHom first call. Mixed rows use the "
        "FP64 comparator. The FFTHomPy clock here includes its fresh audit "
        "(the published one excluded it). New measurements, not the paper's.",
        "",
        header,
    ]
    for r in rows:
        lines.append(
            f"| {r['cell']} | {r['route']} | {r['n']} | {r['precision']} | "
            f"{r['cufhom_first_s']} | {r['cufhom_warm_s']} | {r['error_corrected']} | "
            f"{r['error_direct']} | {r['comparator']} | {r['comparator_s'] or '—'} | "
            f"{r['speedup'] or '—'} |"
        )
    if failures:
        lines += ["", "## Failed or rejected records", ""] + [
            f"- {f}" for f in failures
        ]
    md_path = args.output / "section41_table.md"
    md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {csv_path} and {md_path}")
    return 0


def main(argv=None):
    """Parse the subcommand and dispatch."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=("check", "fetch", "run", "table"))
    parser.add_argument("--mode", choices=tuple(MODES), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--external", type=Path, default=ROOT / "external")
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("cufhom", "fans", "ffthompy"),
        default=["cufhom", "fans", "ffthompy"],
    )
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--sessions", type=int, help="override the mode's sessions")
    parser.add_argument(
        "--grids",
        nargs="+",
        type=int,
        choices=GRIDS["hex8"],
        help="override the mode's grids: HEX8/FANS size; GaNi/FFTHomPy use "
        "the paired odd size (64->63, 128->125, 192->189)",
    )
    parser.add_argument("--dry-run", action="store_true", help="list pending units")
    parser.add_argument(
        "--no-prime",
        dest="prime",
        action="store_false",
        help="measure cold first calls (kernel compilation included); by default "
        "an untabulated CuFHom run first fills the kernel caches",
    )
    args = parser.parse_args(argv)
    args.external = args.external.resolve()
    if args.output is None:
        args.output = ROOT / "results" / f"section41_{args.mode}"
    args.output = args.output.resolve()
    if args.sessions is not None and args.sessions < 1:
        parser.error("--sessions must be positive")
    if args.action == "check":
        reexec_with_libraries()
    return {"check": check, "fetch": fetch, "run": run, "table": table}[args.action](
        args
    )


if __name__ == "__main__":
    raise SystemExit(main())
