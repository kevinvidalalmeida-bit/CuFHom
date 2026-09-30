"""Paired exact-operation candidates on the unchanged 40-increment G09 curve."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from profile_nonlinear_stages import parameters, solver, sync
from compare_nonlinear_curves import compare_runs
from hotspot_candidates import installed

VARIANTS = dict(baseline={}, split_fft=dict(split_fft=True),
                registers=dict(tangent=True, coordinates=True),
                combined=dict(split_fft=True, tangent=True, coordinates=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=['baseline', 'split_fft'])
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or 'baseline' not in args.variants:
        parser.error('Use a new directory, positive repeats and include baseline')
    args.output.mkdir(parents=True)
    p = parameters()
    p['cupy_rfft_batch_mode'] = 'full'  # baseline before component-policy integration
    for name in args.variants:
        with installed(**VARIANTS[name]):
            solver.solve_nonlinear_homogenization(dict(p, path=[0., .00025]))
    report = dict(scope='Warm G09 240^3 GPU, 40 uniform increments, same physics and tolerances. '
        'Solver wall includes setup/solve/audit/commit; excludes exports and plotting. '
        'One GPU job; rotated variant order. Baseline is current fused PCG/ew_target/reuse.',
        candidate_sha256=hashlib.sha256(Path(__file__).with_name('hotspot_candidates.py').read_bytes()).hexdigest(),
        runs=[], summary={})
    reference = None
    for repeat in range(args.repeats):
        offset = repeat % len(args.variants)
        for name in args.variants[offset:]+args.variants[:offset]:
            def progress(row):
                if round(row['target_strain']/.00025) % 10 == 0:
                    print(repeat, name, row['target_strain'], row['seconds'], flush=True)
            with installed(**VARIANTS[name]):
                sync(); start = time.perf_counter()
                result = solver.solve_nonlinear_homogenization(dict(p, on_step=progress))
                sync(); elapsed = time.perf_counter()-start
            if reference is None:
                reference = result
            comparison = compare_runs(reference, result)
            out = args.output/f'{name}_{repeat}'; out.mkdir()
            (out/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            if not comparison['passed'] or result['rejected_steps'] or len(result['history']) != 41:
                raise RuntimeError('Curve audit failed: '+name)
            report['runs'].append(dict(variant=name, repeat=repeat, solver_wall_s=result['wall_seconds'],
                external_wall_s=elapsed, comparison=comparison))
            for variant in args.variants:
                rows = [r for r in report['runs'] if r['variant'] == variant]
                if rows:
                    report['summary'][variant] = dict(
                        median_seconds=statistics.median(r['solver_wall_s'] for r in rows),
                        seconds=[r['solver_wall_s'] for r in rows],
                        cg=[r['comparison']['runs']['candidate']['cg_iterations'] for r in rows],
                        max_curve_error_over_scale=max(r['comparison']['max_stress_error_over_scale'] for r in rows))
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('DONE', name, result['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
