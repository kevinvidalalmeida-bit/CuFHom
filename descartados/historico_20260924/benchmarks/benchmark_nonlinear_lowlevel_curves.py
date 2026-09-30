"""Full-curve audit of isolated low-level and tangent-reference experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from contextlib import nullcontext
from profile_nonlinear_stages import parameters, solver, sync
from compare_nonlinear_curves import compare_runs
from nonlinear_lowlevel_candidates import installed

VARIANTS = dict(baseline=None, fused_dot=dict(fused_dot=True),
    matrix_voigt=dict(reference='matrix_voigt'), matrix_reuss=dict(reference='matrix_reuss'),
    full_voigt=dict(reference='full_voigt'),
    fused_matrix_voigt=dict(fused_dot=True, reference='matrix_voigt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS,
                        default=['baseline','fused_dot','matrix_voigt','matrix_reuss','full_voigt'])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-curve-seconds', type=float, default=180.)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.variants[0] != 'baseline':
        parser.error('Use a new output directory, positive repeats, baseline first')
    args.output.mkdir(parents=True)
    p = parameters()
    p.update(cupy_rfft_batch_mode='auto', coordinate_pcg='fused',
             reuse_trial_response=True, inner_forcing='ew_target')
    for name in args.variants:
        with installed(**VARIANTS[name]) if VARIANTS[name] else nullcontext():
            solver.solve_nonlinear_homogenization(dict(p, path=[0.,.00025]))
    paths = [Path(__file__), Path(__file__).with_name('nonlinear_lowlevel_candidates.py'),
             Path(__file__).with_name('benchmark_fused_adjoint_dot.py')]
    report = dict(scope='Same G09 240^3, 40 uniform increments, current production baseline, '
        'one GPU job. Warm plans/kernels. Setup/solve/audits/commits included; fields/serialization '
        'excluded. No production changes. Rotated order.',
        candidate_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        runs=[], summary={})
    reference = None
    for rep in range(args.repeats):
        names = args.variants[rep % len(args.variants):]+args.variants[:rep % len(args.variants)]
        for name in names:
            started = time.perf_counter()
            def progress(row):
                if round(row['target_strain']/.00025) % 10 == 0:
                    print(rep, name, row['target_strain'], row['seconds'], flush=True)
                if time.perf_counter()-started > args.max_curve_seconds:
                    raise TimeoutError('Curve budget exceeded; candidate not accepted')
            out = args.output/f'{name}_{rep}'; out.mkdir()
            stats, result = None, None
            try:
                with installed(**VARIANTS[name]) if VARIANTS[name] else nullcontext() as context:
                    stats = context
                    sync(); started = time.perf_counter()
                    result = solver.solve_nonlinear_homogenization(dict(p, on_step=progress))
                    sync(); wall = time.perf_counter()-started
                if reference is None:
                    reference = result
                comparison = compare_runs(reference, result)
                (out/'metadata.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                run = dict(variant=name, repeat=rep, solver_wall_s=result['wall_seconds'],
                    external_wall_s=wall, comparison=comparison,
                    accepted=bool(comparison['passed'] and not result['rejected_steps']
                                  and len(result['history'])==41), experiment=stats)
            except (TimeoutError, solver.NonlinearConvergenceError) as exc:
                run = dict(variant=name, repeat=rep, accepted=False, error=str(exc),
                           external_wall_s=time.perf_counter()-started, experiment=stats)
                if name == 'baseline':
                    raise
            report['runs'].append(run)
            for variant in args.variants:
                rows = [r for r in report['runs'] if r['variant']==variant and r.get('accepted')]
                if rows:
                    report['summary'][variant] = dict(
                        median_seconds=statistics.median(r['solver_wall_s'] for r in rows),
                        seconds=[r['solver_wall_s'] for r in rows],
                        cg=[r['comparison']['runs']['candidate']['cg_iterations'] for r in rows],
                        max_curve_error_over_scale=max(r['comparison']['max_stress_error_over_scale'] for r in rows))
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print('DONE', name, run.get('solver_wall_s'), run['accepted'], flush=True)


if __name__ == '__main__':
    main()
