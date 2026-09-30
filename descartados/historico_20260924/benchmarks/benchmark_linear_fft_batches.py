"""Paired six-load G09 audit of real-FFT execution policies."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from profile_g09_fft import (G09_N240, MATERIAL, optimized_parameters, quiet_solver,
    solve_homogenization, sync, compile_cfield_geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('New output directory and at least two repeats required')
    args.output.mkdir(parents=True)
    phase, ori = np.load(G09_N240/'phase.npy'), np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    p = optimized_parameters(phase, ori, json.loads(MATERIAL.read_text()),
                             compile_cfield_geometry(phase, ori))
    report = dict(scope='Six elastic loads, G09 240^3, warm plans/kernels, compiled geometry; '
        'metadata IO included, field export excluded. Rotated order; same rtol=1e-5.',
        runs=[], summary={})
    modes, reference = ['full', 'auto'], None
    for repeat in range(-1, args.repeats):
        for mode in modes[repeat%2:]+modes[:repeat%2]:
            output = args.output/f'{mode}_{repeat}.json'
            with quiet_solver():
                sync(); start = time.perf_counter()
                ceff = solve_homogenization(dict(p, cupy_rfft_batch_mode=mode,
                    solver_timing_path=str(output)))
                sync(); elapsed = time.perf_counter()-start
            info = json.loads(output.read_text())['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
            if reference is None:
                reference = ceff
            error = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
            residual = max(i['true_norm_res_rel'] for i in info)
            if not np.isfinite(ceff).all() or error > 1e-6 or residual > 1e-5:
                raise RuntimeError('Physical audit failed')
            if repeat < 0:
                continue
            report['runs'].append(dict(mode=mode, repeat=repeat, seconds=elapsed,
                Ceff=ceff.tolist(), relative_Ceff_error=error, max_residual=residual,
                cg=sum(i['cg_iterations'] for i in info)))
            for key in modes:
                rows = [r for r in report['runs'] if r['mode'] == key]
                if rows:
                    report['summary'][key] = dict(median_seconds=float(np.median([r['seconds'] for r in rows])),
                        cg=[r['cg'] for r in rows], max_Ceff_error=max(r['relative_Ceff_error'] for r in rows))
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(mode, elapsed, flush=True)


if __name__ == '__main__':
    main()
