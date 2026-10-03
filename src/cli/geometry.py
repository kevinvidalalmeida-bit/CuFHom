"""Generate periodic straight or flexible fiber geometries, with A2 or A4 targets."""
import argparse
from pathlib import Path
import sys
from cufhom.geometry import GeometryRequest, generate_geometry


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fiber-model', choices=['straight', 'flexible'], default='straight')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path, help='Certified checkpoint; inherits saved geometry and scheduler settings')
    p.add_argument('--shape', choices=['capsule','cylinder'], default='capsule',
                   help='Physical volume and raster; both use conservative capsule contacts')
    p.add_argument('--wall-seconds', type=float, help='Stop at next progress event, preserving last certified geometry')
    p.add_argument('--vf', type=float, default=.15)
    p.add_argument('--ar', type=float, default=8.)
    p.add_argument('--box', type=float, default=16.)
    p.add_argument('--diameter', type=float, default=1.)
    p.add_argument('--grid', type=int, default=96)
    p.add_argument('--a2', type=float, nargs=2,
                   help='Diagonal A2: a11 a22; a33=1-a11-a22. Use --solver a2 to enforce A2 alone; auto also enforces an A4 closure.')
    p.add_argument('--orientation-json', type=Path,
                   help='JSON with full A2 (3x3) and/or A4 (3x3x3x3); checks consistency')
    p.add_argument('--seed', type=int, default=2026092901)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--target-count', type=int)
    p.add_argument('--initial-count', type=int, default=12)
    p.add_argument('--bootstrap-max-count', type=int, default=48,
                   help='A4 only: bounded growth of an orientation-limited initial batch')
    p.add_argument('--batch-size', type=int, help='A4 default: approximately one volume percentage point')
    p.add_argument('--adaptive-batches', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--candidates', type=int, help='Placement proposals per fiber: A4 default 128, A2 default 32')
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--retry-shake', type=float,
                   help='Collective Gaussian retry displacement/d; A4 default .05, zero disables it')
    p.add_argument('--max-stages', type=int, help='Default max(40, target count); failed stages consume this budget')
    p.add_argument('--relax-iterations', type=int, help='Default 3000 for A4, 200 for A2')
    p.add_argument('--max-relax-iterations', type=int, help='Extend improving A4 trials up to this total; default 4x initial budget')
    p.add_argument('--neighbor-skin', type=float, default=.5, help='Verlet skin in physical diameter units')
    p.add_argument('--orientation-weight', type=float,
                   help='Default 1 for A4; 10 for A2')
    p.add_argument('--solver', choices=['auto', 'a2', 'a4'], default='auto',
                   help='a2: A2 only; a4: A2 and A4. auto uses a4 for straight and a2 for flexible.')
    p.add_argument('--min-gap', type=float, default=0., help='Minimum surface clearance in physical units; straight A4 only')
    p.add_argument('--a4-tolerance', type=float, default=1e-4)
    p.add_argument('--voxel-a4-tolerance', type=float, default=.06,
                   help='Relative full-tensor norm; raster check independent of continuous A4')
    p.add_argument('--a2-tolerance', type=float, default=.035)
    p.add_argument('--overlap-tolerance', type=float,
                   help='Default 1e-6 for A4, 1/120 for A2; relative to physical diameter')
    p.add_argument('--geometry-only', action='store_true',
                   help='Skip potentially large raster; never exports geometry.npz or certifies voxel Vf')
    p.add_argument('--segments', type=int, default=4)
    p.add_argument('--bending-weight', type=float, default=.01)
    p.add_argument('--max-turn-degrees', type=float, default=30.)
    p.add_argument('--vf-tolerance', type=float, default=.006, help='Absolute voxel volume fraction tolerance')
    p.add_argument('--voxel-a2-tolerance', type=float, default=.06, help='Relative voxel second moment tolerance')
    p.add_argument('--quiet', dest='verbose', action='store_false', help='Write events without console progress')
    args = p.parse_args()
    flags = {value.split('=',1)[0] for value in sys.argv[1:] if value.startswith('--')}
    args.provided = [action.dest for action in p._actions if flags.intersection(action.option_strings)]
    if args.solver == 'a4' and args.fiber_model != 'straight':
        p.error('--solver a4 applies to straight fibers only')
    if (not 0 < args.vf < 1 or args.ar <= 0 or args.box <= 0 or args.diameter <= 0
            or (args.a2 is not None and (min(args.a2) < 0 or sum(args.a2) > 1))
            or args.threads < 1 or args.grid < 2 or not 0 < args.voxel_a4_tolerance < float('inf')
            or (args.target_count is not None and args.target_count < 1)):
        p.error('Invalid geometry settings')
    summary = generate_geometry(GeometryRequest(**vars(args)))
    raise SystemExit(0 if summary['accepted'] or (args.geometry_only and summary['status'] == 'target_count_reached') else 2)


if __name__ == '__main__':
    main()
