"""Compare late residual smoothing on general periodic, synthetic geometries."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import time

import numpy as np
from cufhom.pipeline.fft_solver import compile_cfield_geometry, cp, solve_homogenization
from profile_g09_linear import sha256


def run(args):
    if args.output.exists() or args.repeats < 1 or args.size < 8:
        raise ValueError('Use a new output directory, size >= 8 and positive repetitions')
    args.output.mkdir(parents=True)
    rows = []
    for name, contrast, odd in [('laminate', 30., False), ('compact', 30., False),
                                ('regions', 100., True), ('low_contrast', 1.5, False)]:
        n = args.size + int(odd)
        grid = np.indices((n, n, n))/n
        phase = np.zeros((n, n, n), np.uint8)
        if name == 'laminate':
            phase[:n//3] = 1
        elif name == 'regions':
            phase[(grid[0]+.6*grid[1]+.3*grid[2]) % .5 < .2] = 1
        else:
            for center in ((.25, .25, .25), (.7, .6, .65)):
                phase[np.sum((grid-np.array(center)[:, None, None, None])**2, axis=0) < .17**2] = 1
        ori = np.empty(phase.shape+(3,), np.float32)
        ori[:] = np.array([1., 2., 3.])/np.sqrt(14)
        if name == 'regions':
            ori[n//2:] = np.array([-2., 1., 1.])/np.sqrt(6)
        phase.flags.writeable = ori.flags.writeable = False
        common = dict(input_dir='.', phase_array=phase, ori_array=ori,
            compiled_cfield_geometry=compile_cfield_geometry(phase, ori),
            Em=3., nu_m=.3, Ef_L=3*contrast, Ef_T=3*np.sqrt(contrast),
            nu_LT=.25, nu_TT=.3, G_LT=1.2*np.sqrt(contrast),
            fft_backend='cupy', solver_rtol=1e-5, solver_atol=0.,
            solver_maxiter=1000, require_convergence=True, solver_verbose=False)
        reference = None
        for rep in range(-1, args.repeats):
            order = [False, True] if rep % 2 == 0 else [True, False]
            for smooth in order:
                timing = args.output/f'{name}_{smooth}_{rep}.json'
                cp.cuda.get_current_stream().synchronize()
                start = time.perf_counter()
                with open(os.devnull, 'w') as sink, redirect_stdout(sink):
                    C = solve_homogenization(dict(common, linear_residual_smoothing=smooth,
                        solver_timing_path=str(timing)))
                cp.cuda.get_current_stream().synchronize()
                wall = time.perf_counter()-start
                if reference is None:
                    reference = C.copy()
                error = float(np.linalg.norm(C-reference)/np.linalg.norm(reference))
                infos = json.loads(timing.read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
                assert len(infos) == 6 and error < 1e-6
                assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)
                rows.append(dict(case=name, shape=list(phase.shape), contrast=contrast,
                    smoothing=smooth, repetition=rep, wall_s=wall,
                    cg=sum(i['cg_iterations'] for i in infos), Ceff_relative_difference=error,
                    max_physical_residual=max(i['true_norm_res_rel'] for i in infos)))
        summary = {str(s): float(np.median([r['wall_s'] for r in rows if
            r['case'] == name and r['smoothing'] == s and r['repetition'] >= 0])) for s in (False, True)}
        print(name, summary, flush=True)
    root = Path(__file__).resolve().parents[1]
    report = dict(scope='Six loads, compiled geometry, warm alternating runs; full physical audit. '
        'Synthetic cases are independent of G09 and do not model individual fibres.',
        rows=rows, source_sha256={str(p.relative_to(root)): sha256(p) for p in (root/'src').rglob('*.py')},
        benchmark_sha256=sha256(Path(__file__)), gpu=str(cp.cuda.runtime.getDeviceProperties(0)['name']))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=5)
    run(parser.parse_args())
