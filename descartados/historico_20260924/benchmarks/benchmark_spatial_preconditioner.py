"""Frozen real Newton systems on an unchanged G09 240^3, 40-step trajectory.

Candidates never advance physical state. After each screening the original
PCG solves the same system and drives the reference trajectory. No timings
from the instrumented trajectory are advertised as production curve times.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch
import numpy as np
from profile_nonlinear_stages import parameters, solver, sync
from nonlinear_lowlevel_candidates import action_context
from spatial_preconditioner_candidate import BlockCompliance, SpatialPCG, cp, device_dot
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from compare_nonlinear_curves import compare_runs


def configurations():
    choices = {}
    for edge in (4, 8, 16):
        for averaging in ('voigt', 'reuss'):
            for weight in (.1, 1.):
                name = f'matrix_{averaging}_b{edge}_w{weight:g}'
                choices[name] = dict(edge=edge, support='matrix', averaging=averaging, weight=weight)
    for support in ('all', 'plastic'):
        for averaging in ('voigt', 'reuss'):
            choices[f'{support}_{averaging}_b8_w0.1'] = dict(edge=8,
                support=support, averaging=averaging, weight=.1)
    return choices


def timed(fn):
    sync(); start = time.perf_counter()
    result = fn()
    sync()
    return result, time.perf_counter()-start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants', nargs='+', choices=configurations())
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--max-solve-seconds', type=float, default=15.)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reduction', choices=['warp', 'generic'], default='warp')
    parser.add_argument('--late-newton', action='store_true',
        help='Capture Newton 1/3/2 at steps 1/12/36, instead of the first at all steps')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a fresh output directory and positive repetitions')
    args.output.mkdir(parents=True)
    variants = configurations()
    if args.variants:
        variants = {name: variants[name] for name in args.variants}
    sources = [Path(__file__), Path(__file__).with_name('spatial_preconditioner_candidate.py'),
        Path(solver.__file__), Path(solver.__file__).with_name('nonlinear_krylov.py')]
    report = dict(scope='Real Newton linearizations at steps 1,12,36 of unchanged '
        'G09 240^3 40 uniform increments. Same RHS, zero initial correction and captured '
        'inner tolerance per state. Warm GPU. Build plus solve plus production true '
        'residual audit timed; extra independent audits excluded. One GPU compute job. '
        'Block field is preconditioning only, never changes physical discretization.',
        hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        variants=variants, reduction=args.reduction, systems=[], completed=False)
    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    p = parameters()
    p.update(coordinate_pcg='fused', reuse_trial_response=True,
        inner_forcing='ew_target', cupy_rfft_batch_mode='auto')
    solver.solve_nonlinear_homogenization(dict(p, path=[0., .00025]))
    original = FusedCoordinatePCG.solve
    position = dict(accepted_step=0, seen=set(), calls={})
    capture = {1: 1, 12: 3 if args.late_newton else 1, 36: 2 if args.late_newton else 1}
    report['captured_newton_by_step'] = capture

    def progress(row):
        position['accepted_step'] = round(row['target_strain']/.00025)
        if position['accepted_step'] % 10 == 0:
            print('TRAJECTORY', position['accepted_step'], row['seconds'], flush=True)

    def screen(engine, action, rhs, dot, tolerance, maxiter, reference):
        step = position['accepted_step']+1
        position['calls'][step] = position['calls'].get(step, 0)+1
        if step not in capture or position['calls'][step] != capture[step] or step in position['seen']:
            return original(engine, action, rhs, dot, tolerance, maxiter, reference)
        position['seen'].add(step)
        fields, gpu = action_context(action)
        response = fields['response']
        rhs_saved = rhs.copy()
        arrays = [response.stress, response.direction, response.a, response.b, response.dp]
        saved = [a.copy() for a in arrays]
        row = dict(step=step, newton=position['calls'][step], target_strain=step*.00025, inner_tolerance=tolerance,
            shift=fields['shift'], plastic_fraction_RVE=float(cp.mean(response.dp>0)),
            runs=[], summary={})
        report['systems'].append(row); save()
        print('SYSTEM', step, 'eta', tolerance, 'plastic', row['plastic_fraction_RVE'], flush=True)
        # Compile every candidate kernel and warm all relevant FFT plans once.
        warm = BlockCompliance(engine.Q, gpu, response, **next(iter(variants.values())), shift=fields['shift'])
        warm.correction(rhs)
        device_dot(rhs, rhs)
        del warm
        sync()
        reference_solution = original(engine, action, rhs, dot, tolerance, maxiter, reference)[0]
        reference_energy = dot(reference_solution, action(reference_solution))
        rhs_norm = np.sqrt(dot(rhs, rhs))
        for rep in range(args.repeats):
            names = list(variants)
            cut = rep % len(names)
            names = ['baseline']+names[cut:]+names[:cut]+['baseline_after']
            for name in names:
                base = name.startswith('baseline')
                local = None
                build = 0.
                if not base:
                    local, build = timed(lambda: BlockCompliance(engine.Q, gpu, response,
                        **variants[name], shift=fields['shift']))
                pcg = engine
                if not base:
                    pcg, engine_setup = timed(lambda: SpatialPCG(engine.Q, local, reduction=args.reduction))
                    build += engine_setup
                counted = dict(operator_calls=0)
                started = time.perf_counter()
                def guarded(z):
                    counted['operator_calls'] += 1
                    if time.perf_counter()-started > args.max_solve_seconds:
                        raise TimeoutError('Frozen solve budget exceeded')
                    return action(z)
                context = nullcontext() if base else pcg.installed()
                run = dict(variant=name, repeat=rep, setup_seconds=build)
                try:
                    with context:
                        (x, iterations), seconds = timed(lambda: original(pcg, guarded, rhs,
                            dot, tolerance, maxiter, reference))
                    true = rhs-action(x)
                    residual = np.sqrt(dot(true, true))/rhs_norm
                    difference = x-reference_solution
                    energy_error = np.sqrt(max(0., dot(difference, action(difference)))/reference_energy)
                    admissible = engine.Q.adjoint(engine.Q.strain(x))-x
                    compatibility = np.sqrt(dot(admissible, admissible)/dot(x, x))
                    immutable = bool(cp.array_equal(rhs, rhs_saved)) and all(
                        bool(cp.array_equal(a, b)) for a, b in zip(arrays, saved))
                    run.update(solve_seconds=seconds, total_seconds=build+seconds,
                        cg=iterations, true_relative_residual=residual,
                        relative_energy_difference=energy_error,
                        compatibility_error=compatibility, response_unchanged=immutable,
                        accepted=bool(residual <= tolerance and compatibility < 2e-6 and immutable),
                        operator_calls=counted['operator_calls'],
                        gpu_pool_reserved_bytes=cp.get_default_memory_pool().total_bytes())
                    del x, true, difference, admissible
                except (TimeoutError, solver.NonlinearConvergenceError) as exc:
                    run.update(accepted=False, error=str(exc),
                        solve_seconds=time.perf_counter()-started,
                        operator_calls=counted['operator_calls'])
                row['runs'].append(run)
                for candidate in ['baseline', 'baseline_after']+list(variants):
                    selected = [r for r in row['runs'] if r['variant']==candidate and r['accepted']]
                    if selected:
                        row['summary'][candidate] = dict(
                            total_seconds_median=statistics.median(r['total_seconds'] for r in selected),
                            solve_seconds_median=statistics.median(r['solve_seconds'] for r in selected),
                            setup_seconds_median=statistics.median(r['setup_seconds'] for r in selected),
                            cg=[r['cg'] for r in selected])
                save()
                print('RESULT', step, rep, name, run.get('cg'),
                    round(run.get('total_seconds', run['solve_seconds']), 5), run['accepted'], flush=True)
                if base and not run['accepted']:
                    raise RuntimeError('Baseline failed audit')
                if not base:
                    del pcg, local
        # The candidate correction is not used to advance the loading path.
        return original(engine, action, rhs, dot, tolerance, maxiter, reference)

    with patch.object(FusedCoordinatePCG, 'solve', screen):
        result = solver.solve_nonlinear_homogenization(dict(p, on_step=progress))
    canonical = json.loads(json.dumps(result, allow_nan=False))
    (args.output/'reference_trajectory.json').write_text(json.dumps(canonical, indent=2)+'\n')
    if args.reference:
        report['trajectory_audit'] = compare_runs(json.loads(args.reference.read_text()), canonical)
        if not report['trajectory_audit']['passed']:
            raise RuntimeError('Screening modified the reference trajectory')
    report['completed'] = (position['seen']=={1, 12, 36} and len(result['history'])==41
        and not result['rejected_steps'])
    report['note'] = 'Instrumented trajectory wall time includes all screening solves; '
    report['note'] += 'not a new production-curve benchmark. Pool reserve is not peak live memory.'
    save()
    if not report['completed']:
        raise RuntimeError('Incomplete screening trajectory')
    print('COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
