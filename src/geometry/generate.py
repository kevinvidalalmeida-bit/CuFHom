"""Generate, certify and export periodic geometries through one SAM route."""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cufhom.runtime import progress, save_json


@dataclass
class GeometryRequest:
    """Configure periodic fiber generation, acceptance criteria and output.

    Lengths use one consistent physical unit; ``grid`` is the voxel count per
    axis. Generation and voxel acceptance tolerances are checked separately.
    ``provided`` identifies explicit overrides when resuming a checkpoint.
    """

    output: Path
    fiber_model: str = "straight"
    shape: str = "capsule"
    resume: Path | None = None
    wall_seconds: float | None = None
    vf: float = 0.15
    ar: float = 8.0
    box: float = 16.0
    diameter: float = 1.0
    grid: int = 96
    a2: tuple[float, float] | None = None
    orientation_json: Path | None = None
    seed: int = 2026092901
    threads: int = 2
    solver: str = "auto"
    target_count: int | None = None
    initial_count: int = 12
    bootstrap_max_count: int = 48
    batch_size: int | None = None
    adaptive_batches: bool | None = None
    candidates: int | None = None
    attempts: int = 3
    retry_shake: float | None = None
    max_stages: int | None = None
    relax_iterations: int | None = None
    max_relax_iterations: int | None = None
    neighbor_skin: float = 0.5
    orientation_weight: float | None = None
    min_gap: float = 0.0
    a4_tolerance: float = 1e-4
    a2_tolerance: float = 0.035
    overlap_tolerance: float | None = None
    vf_tolerance: float = 0.006
    voxel_a2_tolerance: float = 0.06
    voxel_a4_tolerance: float = 0.06
    geometry_only: bool = False
    segments: int = 4
    bending_weight: float = 0.01
    max_turn_degrees: float = 30.0
    verbose: bool = True
    provided: list[str] = field(default_factory=list, repr=False)


def atomic_npz(path, **arrays):
    """Write a compressed NumPy archive through a sibling temporary file."""
    import numpy as np

    pending = path.with_suffix(".pending.npz")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)


def generate_geometry(args: GeometryRequest):
    """Generate, audit and export a periodic fiber geometry.

    The request selects straight or flexible fibers and orientation targets.
    The output directory must not exist. Settings and accepted checkpoints are
    written there; the request is updated with resolved defaults.
    """
    import json

    import numba
    import numpy as np
    from cufhom.geometry.flexible import FlexibleSettings, decode, generate_flexible
    from cufhom.geometry.orientation import resolve_orientation, validate_a2
    from cufhom.geometry.straight import Settings, generate_staged
    from cufhom.geometry.voxel import (
        fiber_volume,
        rasterize_capsules,
        rasterize_cylinders,
        voxel_a2,
    )

    if (
        not np.isfinite(
            [
                args.vf,
                args.ar,
                args.box,
                args.diameter,
                args.vf_tolerance,
                args.voxel_a2_tolerance,
                args.voxel_a4_tolerance,
            ]
        ).all()
        or not 0 < args.vf < 1
        or min(
            args.ar,
            args.box,
            args.diameter,
            args.vf_tolerance,
            args.voxel_a2_tolerance,
            args.voxel_a4_tolerance,
        )
        <= 0
        or args.grid < 2
        or args.threads < 1
        or args.fiber_model not in ("straight", "flexible")
    ):
        raise ValueError("Invalid geometry settings")
    args.output = Path(args.output)
    emit = progress if args.verbose else lambda *a, **kw: None
    initial = resume_state = resumed_a2 = resumed_a4 = None
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as saved:
            initial = saved["state"].copy()
            previous = json.loads(str(saved["settings_json"]))
            inherited = {k: v for k, v in previous.items() if hasattr(args, k)}
            inherited.update(
                ar=previous["length"] / previous["diameter"],
                fiber_model=str(saved["fiber_model"]),
                shape=str(saved["shape"]) if "shape" in saved else "capsule",
            )
            if "vf_target" in saved:
                inherited["vf"] = float(saved["vf_target"])
            if "grid" in saved:
                inherited["grid"] = int(saved["grid"])
            inherited["solver"] = {"paper-a4": "a4", "lbfgs-a2": "a2"}.get(
                previous["solver"], previous["solver"]
            )
            for key, value in inherited.items():
                if key not in args.provided:
                    setattr(args, key, value)
            resumed_a2 = saved["A2"].copy()
            resumed_a4 = saved["A4"].copy() if saved["A4"].size else None
            resume_state = (
                json.loads(str(saved["resume_json"]))
                if "resume_json" in saved
                else None
            )
        if (
            resume_state is not None
            and "seed" in args.provided
            and args.seed != previous["seed"]
        ):
            raise ValueError(
                "Resume restores the saved random state; cannot change its seed"
            )
        if resume_state is not None and "batch_size" in args.provided:
            resume_state["batch_size"] = min(
                resume_state["batch_size"], args.batch_size
            )
        if (
            args.fiber_model != inherited["fiber_model"]
            or args.shape != inherited["shape"]
            or not np.allclose(
                [args.box, args.ar * args.diameter, args.diameter],
                [previous["box"], previous["length"], previous["diameter"]],
                rtol=0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                "Resume cannot change box, fiber dimensions, model or physical shape"
            )
        if args.fiber_model != "straight":
            raise ValueError("Resume currently supports straight fibers")
    if args.solver == "auto":
        args.solver = "a4" if args.fiber_model == "straight" else "a2"
    if args.orientation_weight is None:
        args.orientation_weight = 1.0 if args.solver == "a4" else 10.0
    if args.relax_iterations is None:
        args.relax_iterations = 3000 if args.solver == "a4" else 200
    if args.overlap_tolerance is None:
        args.overlap_tolerance = 1e-6 if args.solver == "a4" else 1 / 120
    if args.adaptive_batches is None:
        args.adaptive_batches = args.solver == "a4"
    if args.max_relax_iterations is None:
        args.max_relax_iterations = (
            4 * args.relax_iterations if args.solver == "a4" else 0
        )
    if args.retry_shake is None:
        args.retry_shake = 0.05 if args.solver == "a4" else 0.0
    if args.candidates is None:
        args.candidates = 128 if args.solver == "a4" else 32
    target = None if args.a2 is None else np.diag([*args.a2, 1 - sum(args.a2)])
    target_a4 = None
    if args.orientation_json:
        supplied = json.loads(args.orientation_json.read_text())
        if (
            not isinstance(supplied, dict)
            or not set(supplied).issubset({"A2", "A4"})
            or not supplied
        ):
            raise ValueError("Orientation JSON must contain A2 and/or A4 full tensors")
        if any(value is None for value in supplied.values()):
            raise ValueError("Orientation tensors cannot be null; omit a key instead")
        if "A2" in supplied:
            from_file = validate_a2(supplied["A2"])
            if target is not None and not np.allclose(
                target, from_file, rtol=0, atol=1e-10
            ):
                raise ValueError("--a2 conflicts with orientation JSON A2")
            target = from_file
        target_a4 = supplied.get("A4")
    if resumed_a2 is not None:
        if target is not None and not np.allclose(
            target, resumed_a2, rtol=0, atol=1e-10
        ):
            raise ValueError("Resume A2 differs from requested orientation")
        if target_a4 is not None and (
            resumed_a4 is None
            or not np.allclose(target_a4, resumed_a4, rtol=0, atol=1e-10)
        ):
            raise ValueError("Resume A4 differs from requested orientation")
        target, target_a4 = resumed_a2, resumed_a4
    if args.solver != "a4" and (target_a4 is not None or args.min_gap):
        raise ValueError("A4 and explicit minimum gap require straight a4")
    if args.fiber_model == "flexible" and args.solver != "a2":
        raise ValueError("Flexible fibers currently support a2 only")
    if args.fiber_model == "flexible" and (
        args.shape != "capsule" or args.wall_seconds is not None
    ):
        raise ValueError(
            "Cylinder shape and wall budget currently require straight fibers"
        )
    orientation = None
    if args.solver == "a4":
        target, target_a4, orientation = resolve_orientation(target, target_a4)
        emit("Orientation: " + orientation["source"] + "; checking BOTH A2 and A4")
    else:
        target = validate_a2(np.eye(3) / 3 if target is None else target)
        emit("Orientation: checking A2 only")
    numba.set_num_threads(args.threads)
    length = args.ar * args.diameter
    capsule_volume = fiber_volume(length, args.diameter, "capsule")
    physical_volume = fiber_volume(length, args.diameter, args.shape)
    # A new Vf target recalculates the count when resuming unless explicitly fixed.
    if args.resume and "vf" in args.provided and "target_count" not in args.provided:
        args.target_count = None
    count = args.target_count or max(
        1, int(np.floor(args.vf * args.box**3 / physical_volume + 0.5))
    )
    if args.batch_size is None:
        args.batch_size = (
            max(4, int(round(0.01 * args.box**3 / physical_volume)))
            if args.solver == "a4"
            else 4
        )
    if args.max_stages is None:
        args.max_stages = max(40, count)
    common = dict(
        box=args.box,
        length=length,
        diameter=args.diameter,
        target_count=count,
        initial_count=args.initial_count,
        batch_size=args.batch_size,
        candidates=args.candidates,
        attempts=args.attempts,
        max_stages=args.max_stages,
        relax_iterations=args.relax_iterations,
        orientation_weight=args.orientation_weight,
        overlap_tolerance=args.overlap_tolerance,
        a2_tolerance=args.a2_tolerance,
        seed=args.seed,
        solver=args.solver,
        a4_tolerance=args.a4_tolerance,
        min_gap=args.min_gap,
        bootstrap_max_count=args.bootstrap_max_count,
        neighbor_skin=args.neighbor_skin,
        adaptive_batches=args.adaptive_batches,
        max_relax_iterations=args.max_relax_iterations,
        retry_shake=args.retry_shake,
    )
    settings = (
        FlexibleSettings(
            **common,
            segments=args.segments,
            bending_weight=args.bending_weight,
            max_turn_degrees=args.max_turn_degrees,
        )
        if args.fiber_model == "flexible"
        else Settings(**common)
    )
    from cufhom.geometry.straight import validate_settings

    validate_settings(settings, target)
    if args.wall_seconds is not None and (
        not np.isfinite(args.wall_seconds) or args.wall_seconds <= 0
    ):
        raise ValueError("Positive finite wall budget required")
    emit(
        f"PRECHECK solver={args.solver} fibers={count} L/box={length / args.box:.4g} "
        f"voxels/diameter={args.grid * args.diameter / args.box:.3g}; "
        f"physical shape={args.shape}; batch={args.batch_size}"
    )
    initial_size = 0 if initial is None else len(initial)
    ceiling = (
        initial_size + args.max_stages * args.batch_size
        if initial_size
        else max(args.initial_count, args.bootstrap_max_count)
        + (args.max_stages - 1) * args.batch_size
    )
    if count > ceiling:
        emit(
            f"Budget permits at most {ceiling}/{count} fibers; this run will be partial"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(
        args.output / "request.json",
        dict(
            parameters=vars(args),
            settings=asdict(settings),
            orientation_target=orientation,
            A2=target,
            schema="sam_staged_v1",
            fiber_model=args.fiber_model,
            note="Flexible AR uses contour length/diameter, not end-to-end length. "
            "Count estimate is not a volume guarantee for bent/overlapping capsules. "
            if args.fiber_model == "flexible"
            else "AR uses axial length/diameter. ",
        ),
    )

    def report(event):
        with (args.output / "events.jsonl").open("a", encoding="utf8") as log:
            log.write(json.dumps(event) + "\n")
        if event["event"] == "relax_progress":
            overlap = event.get(
                "maximum_relative_overlap", event.get("overlap_relative")
            )
            a4 = f" A4={event['A4_error']:.5g}" if "A4_error" in event else ""
            label = "gap_violation/d" if args.min_gap else "overlap/d"
            emit(
                f"relax iter={event['iteration']} {label}={overlap:.5g} A2={event['A2_error']:.5g}{a4}"
            )
        else:
            emit(json.dumps(event))

    def checkpoint(table, event):
        # One atomic compact checkpoint, never overwrite historical geometries.
        atomic_npz(
            args.output / "last_geometrically_valid.npz",
            state=table,
            fiber_model=np.array(args.fiber_model),
            settings_json=np.array(json.dumps(asdict(settings))),
            A2=target,
            A4=np.asarray(target_a4) if target_a4 is not None else np.empty(0),
            shape=np.array(args.shape),
            vf_target=np.array(args.vf),
            grid=np.array(args.grid),
            **(
                dict(resume_json=np.array(json.dumps(event["resume_state"])))
                if "resume_state" in event
                else {}
            ),
        )
        save_json(args.output / "checkpoint.json", event)

    start = time.perf_counter()
    generator = generate_flexible if args.fiber_model == "flexible" else generate_staged
    extra = dict(target_a4=target_a4) if args.solver == "a4" else {}
    if args.fiber_model == "straight":
        extra.update(
            initial=initial, resume_state=resume_state, wall_seconds=args.wall_seconds
        )
    table, summary = generator(
        settings, target, progress=report, checkpoint=checkpoint, **extra
    )
    if len(table):
        checkpoint(
            table,
            dict(
                event="final_checkpoint",
                status=summary["status"],
                accepted_count=len(table),
                certificate=summary["geometric_certificate"],
                **(
                    dict(resume_state=summary["resume_state"])
                    if "resume_state" in summary
                    else {}
                ),
            ),
        )
    summary["orientation_target"] = orientation
    summary.update(
        fiber_model=args.fiber_model,
        accepted=False,
        wall_s=time.perf_counter() - start,
        grid=args.grid,
        vf_target=args.vf,
        shape=args.shape,
        contact_model="capsule",
        nominal_vf=len(table) * physical_volume / args.box**3,
        nominal_capsule_vf=len(table) * capsule_volume / args.box**3,
        nominal_cylinder_vf=len(table)
        * fiber_volume(length, args.diameter, "cylinder")
        / args.box**3,
        resumed=bool(args.resume),
        resume_rng_restored=resume_state is not None,
    )
    if len(table):
        if args.fiber_model == "flexible":
            vertices, c, u, _ = decode(table, length)
            centers, directions = c.reshape(-1, 3), u.reshape(-1, 3)
            segment_length = length / settings.segments
            atomic_npz(
                args.output / "continuous_chains.npz",
                vertices_um=vertices,
                diameter_um=np.array(args.diameter),
                fiber_model=np.array("flexible"),
                contour_length_um=np.array(length),
            )
        else:
            centers, directions, segment_length = table[:, :3], table[:, 3:6], length
            np.savetxt(
                args.output / "continuous_fibers_master.csv",
                table,
                delimiter=",",
                header="cx_um,cy_um,cz_um,ux,uy,uz,L_um,d_um",
                comments="",
            )
        if args.geometry_only:
            summary.update(
                voxel_checks_performed=False,
                vf_not_certified=True,
                nominal_capsule_vf=len(table) * capsule_volume / args.box**3,
                nominal_cylinder_vf=len(table)
                * np.pi
                * args.diameter**2
                * length
                / 4
                / args.box**3,
                total_wall_s=time.perf_counter() - start,
            )
            save_json(args.output / "summary.json", summary)
            emit(
                f"GEOMETRY ONLY status={summary['status']} fibers={len(table)}/{count}; "
                "Vf voxel not certified, accepted=False"
            )
            return summary
        emit("Rasterizing last feasible geometry, not failed trials")
        scale = args.grid / args.box
        if args.shape == "cylinder":
            phase, ori, voxel_count, _ = rasterize_cylinders(
                centers * scale,
                directions,
                segment_length * scale,
                0.5 * args.diameter * scale,
                args.grid,
            )
        else:
            phase, ori, voxel_count, _ = rasterize_capsules(
                centers * scale,
                directions,
                segment_length * scale,
                0.5 * args.diameter * scale,
                args.grid,
            )
        vf = float(np.count_nonzero(phase) / phase.size)
        a2_error = float(
            np.linalg.norm(voxel_a2(phase, ori) - target) / np.linalg.norm(target)
        )
        summary.update(
            vf=vf,
            vf_ok=abs(vf - args.vf) <= args.vf_tolerance,
            vf_not_certified=False,
            voxel_checks_performed=True,
            voxel_A2_error=a2_error,
            voxel_A2_ok=a2_error <= args.voxel_a2_tolerance,
            voxel_orientation_rule="last fiber/segment containing sample point",
            voxel_boundary_rule="strict interior"
            if args.shape == "cylinder"
            else "closed capsule",
            accepted=bool(
                summary["status"] == "target_count_reached"
                and summary["geometric_certificate"]["accepted"]
                and abs(vf - args.vf) <= args.vf_tolerance
                and a2_error <= args.voxel_a2_tolerance
            ),
        )
        if target_a4 is not None:
            from cufhom.geometry.voxel import voxel_a4

            actual_a4 = voxel_a4(phase, ori)
            error_a4 = float(
                np.linalg.norm(actual_a4 - target_a4) / np.linalg.norm(target_a4)
            )
            summary.update(
                voxel_A4_error=error_a4,
                voxel_A4_tolerance=args.voxel_a4_tolerance,
                voxel_A4_ok=error_a4 <= args.voxel_a4_tolerance,
            )
            summary["accepted"] = bool(summary["accepted"] and summary["voxel_A4_ok"])
        if summary["accepted"]:
            atomic_npz(
                args.output / "geometry.npz",
                phase=phase,
                ori=ori,
                fiber_model=np.array(args.fiber_model),
                schema=np.array("sam_staged_v1"),
                shape=np.array(args.shape),
                A2_target=target,
                A4_target=np.asarray(target_a4)
                if target_a4 is not None
                else np.empty(0),
            )
    summary["total_wall_s"] = time.perf_counter() - start
    save_json(args.output / "summary.json", summary)
    emit(
        f"DONE status={summary['status']} fibers={len(table)}/{count} accepted={summary['accepted']} "
        f"Vf={summary.get('vf')} elapsed={summary['total_wall_s']:.1f}s | {args.output}/summary.json"
    )
    # A feasible but underfilled checkpoint must not be mistaken for success.
    return summary
