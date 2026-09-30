"""Paired six-load G09 tests of isolated/combined candidates, including setup."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
    quiet_solver, sync, cp, solve_homogenization, compile_cfield_geometry)
from linear_strategy_candidates import candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reference-ratios', nargs='*', type=float, default=[],
                        help='Extra combined variants: longitudinal/transverse inverse eigenvalue ratio')
    parser.add_argument('--modes', nargs='+',
                        choices=['baseline', 'production', 'symmetric', 'fused_updates', 'combined', 'anisotropic',
                                 'polarization', 'polarization_macro', 'polarization_local'],
                        default=['baseline', 'symmetric', 'fused_updates', 'anisotropic'])
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('New output directory and at least two repeats required')
    if 'baseline' not in args.modes or len(set(args.modes)) != len(args.modes):
        parser.error('Include baseline exactly once; duplicate modes are not allowed')
    if (len(set(args.reference_ratios)) != len(args.reference_ratios)
            or any(not np.isfinite(r) or not 0 < r < 1.5 for r in args.reference_ratios)):
        parser.error('Reference ratios must be distinct and in (0, 1.5)')
    args.output.mkdir(parents=True)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(MATERIAL.read_text())
    config = optimized_parameters(phase, ori, material, compile_cfield_geometry(phase, ori))
    ratios = {f'combined_ratio_{r:g}':r for r in args.reference_ratios}
    modes = args.modes+list(ratios)
    def experiment(mode):
        return candidate('combined', reference_ratio=ratios[mode]) if mode in ratios else candidate(mode)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    paths = ['FFT/pipeline/fft_solver.py', 'FFT/pipeline/linear_coordinates.py', 'FFT/pipeline/linear_krylov.py',
             'FFT/pipeline/nonlinear_coordinates.py', 'FFT/pipeline/nonlinear_solver.py',
             'FFT/benchmarks/linear_strategy_candidates.py', 'FFT/benchmarks/benchmark_linear_strategies.py',
             'FFT/benchmarks/linear_polarization_candidate.py']
    report = dict(shape=list(phase.shape), material=material, rtol=config['solver_rtol'],
        device=device.decode() if isinstance(device, bytes) else device,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        scope='Warm plans/kernels; same geometry/materials; six loads; anisotropic reference '
              'mean and blocks rebuilt within each solve, reused only across its six loads. '
              'Compiled geometry preloaded, no field export; metadata IO included.',
        variants={m: [] for m in modes}, warmup_errors={})
    reference = None
    for mode in modes:
        try:
            with experiment(mode), quiet_solver():
                ceff = solve_homogenization(config)
            if mode == 'baseline':
                reference = ceff
            print('warmup', mode, flush=True)
        except Exception as error:
            report['warmup_errors'][mode] = str(error)
            print('warmup failed', mode, str(error), flush=True)
    if reference is None:
        raise RuntimeError('Baseline failed')
    for repeat in range(args.repeats):
        order = modes[repeat % len(modes):]+modes[:repeat % len(modes)]
        for mode in order:
            timing = args.output/f'{repeat}_{mode}.json'
            sync(); started = time.perf_counter()
            entry = dict(repeat=repeat)
            try:
                with experiment(mode) as (infos, cache), quiet_solver():
                    ceff = solve_homogenization(dict(config, solver_timing_path=str(timing)))
                sync(); entry['wall_s'] = time.perf_counter()-started
                error = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
                entry.update(Ceff=ceff.tolist(), Ceff_relative_difference=error,
                    Ceff_comparison_passed=bool(np.isfinite(error) and error <= 1e-6),
                    loads=infos, cg=sum(i['coordinate_iterations']+i['correction_iterations'] for i in infos),
                    equilibrium_passed=len(infos)==6 and all(i['true_residual_converged']
                        and i['true_norm_res_rel'] <= config['solver_rtol'] for i in infos),
                    reference_setup={k:v for k,v in cache.items() if k not in ('inverse', 'C', 'engine')})
            except Exception as error:
                sync(); entry.update(wall_s=time.perf_counter()-started, error=str(error), equilibrium_passed=False)
            report['variants'][mode].append(entry)
            print(repeat, mode, entry['wall_s'], entry.get('cg'), entry.get('Ceff_relative_difference'),
                  entry.get('error', ''), flush=True)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    report['summary'] = {}
    for mode, runs in report['variants'].items():
        report['summary'][mode] = dict(median_s=float(np.median([r['wall_s'] for r in runs])),
            equilibrium_passed=all(r['equilibrium_passed'] for r in runs),
            Ceff_comparison_passed=all(r.get('Ceff_comparison_passed', False) for r in runs))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
