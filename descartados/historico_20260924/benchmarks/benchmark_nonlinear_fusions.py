"""Paired full-curve ablations on G09 240^3, without changing the increments."""
import argparse
import json
from pathlib import Path
import statistics
import time

from profile_nonlinear_stages import parameters, solver, sync
from compare_nonlinear_curves import compare_runs

VARIANTS = {
    'baseline': dict(coordinate_pcg='classic', reuse_trial_response=False, inner_forcing='ew'),
    'reuse': dict(coordinate_pcg='classic', reuse_trial_response=True, inner_forcing='ew'),
    'fused': dict(coordinate_pcg='fused', reuse_trial_response=False, inner_forcing='ew'),
    'combined': dict(coordinate_pcg='fused', reuse_trial_response=True, inner_forcing='ew'),
    'target_only': dict(coordinate_pcg='classic', reuse_trial_response=False, inner_forcing='ew_target'),
    'target_combined': dict(coordinate_pcg='fused', reuse_trial_response=True, inner_forcing='ew_target'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or 'baseline' not in args.variants:
        parser.error('Use a new output directory, positive repeats, and include baseline')
    args.output.mkdir(parents=True)
    p = parameters()
    # Isolate Newton/PCG ablations from the later component-FFT optimization.
    p['cupy_rfft_batch_mode'] = 'full'
    # Compile both kernel routes and warm FFT plans outside measurements.
    for name in ('baseline', 'combined'):
        solver.solve_nonlinear_homogenization(dict(p, path=[0., .00025], **VARIANTS[name]))
    report = dict(scope='Same G09 240^3, 40 uniform increments, 0 to 1% strain, '
        'same rtol=1e-5 and demonstrative J2 parameters. Warm GPU timings include '
        'geometry/state setup, solve, audits and commits; exclude field/checkpoint '
        'exports, metadata serialization and plotting. Rotated order; no concurrent GPU work.',
        runs=[], summary={})
    reference = None
    for repeat in range(args.repeats):
        offset = repeat % len(args.variants)
        order = args.variants[offset:]+args.variants[:offset]
        for name in order:
            def progress(row):
                step = int(round(row['target_strain']/.00025))
                if step % 10 == 0:
                    print(repeat, name, step, round(row['seconds'], 3), flush=True)
            sync(); start = time.perf_counter()
            result = solver.solve_nonlinear_homogenization(dict(p, on_step=progress, **VARIANTS[name]))
            sync(); external = time.perf_counter()-start
            if reference is None:
                reference = result
            comparison = compare_runs(reference, result)
            if not comparison['passed'] or result['rejected_steps'] or len(result['history']) != 41:
                raise RuntimeError('Same-curve audit failed: '+name)
            out = args.output/f'{name}_{repeat}'; out.mkdir()
            (out/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            report['runs'].append(dict(variant=name, repeat=repeat, external_wall_s=external,
                solver_wall_s=result['wall_seconds'], comparison=comparison))
            for variant in args.variants:
                rows = [r for r in report['runs'] if r['variant'] == variant]
                if rows:
                    report['summary'][variant] = dict(
                        median_seconds=statistics.median(r['solver_wall_s'] for r in rows),
                        seconds=[r['solver_wall_s'] for r in rows],
                        cg=[r['comparison']['runs']['candidate']['cg_iterations'] for r in rows],
                        max_curve_error_over_scale=max(r['comparison']['max_stress_error_over_scale'] for r in rows))
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('DONE', name, result['wall_seconds'], comparison['runs']['candidate']['cg_iterations'], flush=True)


if __name__ == '__main__':
    main()
