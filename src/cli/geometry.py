"""Generate periodic straight or flexible fiber geometries, with A2 or A4 targets."""

import argparse
import sys
from pathlib import Path

from cufhom.geometry import GeometryRequest, generate_geometry


def main():
    """Parse command-line arguments and generate a certified periodic geometry."""
    p = argparse.ArgumentParser(
        prog="cufhom-generate",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--help-all", action="help", help="Show all numerical and acceptance controls"
    )
    p.add_argument(
        "--fiber-model",
        choices=["straight", "flexible"],
        default="straight",
        help="Fiber centreline model",
    )
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument(
        "--resume",
        type=Path,
        help="Certified checkpoint; inherits saved geometry and scheduler settings",
    )
    p.add_argument(
        "--shape",
        choices=["capsule", "cylinder"],
        default="capsule",
        help="Physical volume and raster; both use conservative capsule contacts",
    )
    p.add_argument(
        "--wall-seconds",
        type=float,
        help="Stop at next progress event, preserving last certified geometry",
    )
    p.add_argument(
        "--vf", type=float, default=0.15, help="Target fiber volume fraction"
    )
    p.add_argument("--ar", type=float, default=8.0, help="Fiber length/diameter ratio")
    p.add_argument("--box", type=float, default=16.0, help="Periodic cell edge length")
    p.add_argument(
        "--diameter", type=float, default=1.0, help="Physical fiber diameter"
    )
    p.add_argument("--grid", type=int, default=96, help="Voxel count per cell axis")
    p.add_argument(
        "--a2",
        type=float,
        nargs=2,
        help="Diagonal A2: a11 a22; a33=1-a11-a22. Use --solver a2 to enforce A2 alone; auto also enforces an A4 closure.",
    )
    p.add_argument(
        "--orientation-json",
        type=Path,
        help="JSON with full A2 (3x3) and/or A4 (3x3x3x3); checks consistency",
    )
    p.add_argument("--seed", type=int, default=2026092901, help="Random seed")
    p.add_argument("--threads", type=int, default=2, help="Numba CPU thread count")
    p.add_argument(
        "--target-count", type=int, help="Override count inferred from volume fraction"
    )
    p.add_argument(
        "--initial-count",
        type=int,
        default=12,
        help="Fibers in the first placement batch",
    )
    p.add_argument(
        "--bootstrap-max-count",
        type=int,
        default=48,
        help="A4 only: bounded growth of an orientation-limited initial batch",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        help="A4 default: approximately one volume percentage point",
    )
    p.add_argument(
        "--adaptive-batches",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Adapt placement batch sizes; selected automatically for A4",
    )
    p.add_argument(
        "--candidates",
        type=int,
        help="Placement proposals per fiber: A4 default 128, A2 default 32",
    )
    p.add_argument(
        "--attempts", type=int, default=3, help="Placement/relaxation retries per stage"
    )
    p.add_argument(
        "--retry-shake",
        type=float,
        help="Collective Gaussian retry displacement/d; A4 default .05, zero disables it",
    )
    p.add_argument(
        "--max-stages",
        type=int,
        help="Default max(40, target count); failed stages consume this budget",
    )
    p.add_argument(
        "--relax-iterations", type=int, help="Default 3000 for A4, 200 for A2"
    )
    p.add_argument(
        "--max-relax-iterations",
        type=int,
        help="Extend improving A4 trials up to this total; default 4x initial budget",
    )
    p.add_argument(
        "--neighbor-skin",
        type=float,
        default=0.5,
        help="Verlet skin in physical diameter units",
    )
    p.add_argument(
        "--orientation-weight", type=float, help="Default 1 for A4; 10 for A2"
    )
    p.add_argument(
        "--solver",
        choices=["auto", "a2", "a4"],
        default="auto",
        help="a2: A2 only; a4: A2 and A4. auto uses a4 for straight and a2 for flexible.",
    )
    p.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Minimum surface clearance in physical units; straight A4 only",
    )
    p.add_argument(
        "--a4-tolerance",
        type=float,
        default=1e-4,
        help="Continuous A4 acceptance tolerance",
    )
    p.add_argument(
        "--voxel-a4-tolerance",
        type=float,
        default=0.06,
        help="Relative full-tensor norm; raster check independent of continuous A4",
    )
    p.add_argument(
        "--a2-tolerance",
        type=float,
        default=0.035,
        help="Continuous A2 acceptance tolerance",
    )
    p.add_argument(
        "--overlap-tolerance",
        type=float,
        help="Default 1e-6 for A4, 1/120 for A2; relative to physical diameter",
    )
    p.add_argument(
        "--geometry-only",
        action="store_true",
        help="Skip potentially large raster; never exports geometry.npz or certifies voxel Vf",
    )
    p.add_argument(
        "--segments", type=int, default=4, help="Segments per flexible fiber"
    )
    p.add_argument(
        "--bending-weight",
        type=float,
        default=0.01,
        help="Flexible-fiber bending penalty",
    )
    p.add_argument(
        "--max-turn-degrees",
        type=float,
        default=30.0,
        help="Maximum turn between flexible segments, in degrees",
    )
    p.add_argument(
        "--vf-tolerance",
        type=float,
        default=0.006,
        help="Absolute voxel volume fraction tolerance",
    )
    p.add_argument(
        "--voxel-a2-tolerance",
        type=float,
        default=0.06,
        help="Relative voxel second moment tolerance",
    )
    p.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Disable console progress; verbose otherwise defaults to %(default)s",
    )
    if "--help-all" not in sys.argv[1:]:
        advanced = {
            "target_count",
            "initial_count",
            "bootstrap_max_count",
            "batch_size",
            "adaptive_batches",
            "candidates",
            "attempts",
            "retry_shake",
            "max_stages",
            "relax_iterations",
            "max_relax_iterations",
            "neighbor_skin",
            "orientation_weight",
            "a4_tolerance",
            "voxel_a4_tolerance",
            "a2_tolerance",
            "overlap_tolerance",
            "segments",
            "bending_weight",
            "max_turn_degrees",
            "vf_tolerance",
            "voxel_a2_tolerance",
        }
        for action in p._actions:
            if action.dest in advanced:
                action.help = argparse.SUPPRESS
    args = p.parse_args()
    flags = {value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")}
    args.provided = [
        action.dest
        for action in p._actions
        if flags.intersection(action.option_strings)
    ]
    if args.solver == "a4" and args.fiber_model != "straight":
        p.error("--solver a4 applies to straight fibers only")
    if (
        not 0 < args.vf < 1
        or args.ar <= 0
        or args.box <= 0
        or args.diameter <= 0
        or (args.a2 is not None and (min(args.a2) < 0 or sum(args.a2) > 1))
        or args.threads < 1
        or args.grid < 2
        or not 0 < args.voxel_a4_tolerance < float("inf")
        or (args.target_count is not None and args.target_count < 1)
    ):
        p.error("Invalid geometry settings")
    summary = generate_geometry(GeometryRequest(**vars(args)))
    raise SystemExit(
        0
        if summary["accepted"]
        or (args.geometry_only and summary["status"] == "target_count_reached")
        else 2
    )


if __name__ == "__main__":
    main()
