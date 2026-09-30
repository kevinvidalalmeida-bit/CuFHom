"""Validate the integrated linear displacement route on G09, against strain CG."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
    quiet_solver, solve_homogenization, sync, compile_cfield_geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('Use a new output directory and at least two repeats')
    args.output.mkdir(parents=True)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(MATERIAL.read_text())
    config = optimized_parameters(phase, ori, material, compile_cfield_geometry(phase, ori))
    source_paths = ['FFT/pipeline/fft_solver.py', 'FFT/pipeline/linear_coordinates.py',
                    'FFT/pipeline/nonlinear_coordinates.py', 'FFT/pipeline/nonlinear_solver.py',
                    'FFT/ffthompy_core/ffthompy/ffthompy/applications.py']
    report = dict(shape=list(phase.shape), material=material, rtol=1e-5,
        source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        variants={}, note='Same-grid six-load solves; preloaded compiled geometry, warm FFT plans. '
        'Baseline true-residual audit explicitly enabled; candidate always audits. Metadata IO included.')
    modes = ['strain', 'displacement']
    for mode in modes:
        with quiet_solver():
            solve_homogenization(dict(config, linear_formulation=mode))
        report['variants'][mode] = dict(wall_s=[], cg=[], max_residual=[], Ceff=[])
    for repeat in range(args.repeats):
        for mode in (modes if repeat % 2 == 0 else modes[::-1]):
            timing = args.output/f'{mode}_{repeat}.json'
            sync(); started = time.perf_counter()
            with quiet_solver():
                ceff = solve_homogenization(dict(config, linear_formulation=mode,
                    check_true_residual=True, solver_timing_path=str(timing)))
            sync(); elapsed = time.perf_counter()-started
            metadata = json.loads(timing.read_text())
            infos = metadata['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
            assert len(infos) == 6 and all(i['true_residual_converged'] for i in infos)
            item = report['variants'][mode]
            item['wall_s'].append(elapsed); item['Ceff'].append(ceff.tolist())
            item['cg'].append(sum(i['cg_iterations'] for i in infos))
            item['max_residual'].append(max(i['true_norm_res_rel'] for i in infos))
            print(mode, elapsed, item['cg'][-1], item['max_residual'][-1], flush=True)
    reference = np.asarray(report['variants']['strain']['Ceff'][0])
    for mode, item in report['variants'].items():
        item['median_s'] = float(np.median(item['wall_s']))
        item['max_relative_Ceff_difference'] = max(float(
            np.linalg.norm(np.asarray(c)-reference)/np.linalg.norm(reference)) for c in item['Ceff'])
    report['equilibrium_passed'] = all(max(v['max_residual']) <= report['rtol']
                                     for v in report['variants'].values())
    report['Ceff_comparison_threshold'] = 1e-6
    report['strict_Ceff_comparison_passed'] = (
        report['variants']['displacement']['max_relative_Ceff_difference'] <= 1e-6)
    report['strict_reference_note'] = ('Separate tighter baseline runs were inconclusive: '
        'at rtol=2e-6 its true residual exceeded the target; at rtol=1e-6 '
        'the original float32 strain route stagnated and returned nonfinite Ceff. '
        'This report certifies equilibrium at 1e-5, not 1e-6 Ceff equivalence.')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('variants','strict','source_sha256','material')}, indent=2))
    if not report['equilibrium_passed']:
        raise RuntimeError('Physical equilibrium audit failed')


if __name__ == '__main__':
    main()
