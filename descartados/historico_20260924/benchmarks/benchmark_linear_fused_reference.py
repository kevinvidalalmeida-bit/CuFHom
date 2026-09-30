"""Test transfer of nonlinear reference-PCG fusion; no default changes."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np

from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
    quiet_solver, solve_homogenization, sync, compile_cfield_geometry)
from pipeline import linear_coordinates as adapter
from pipeline.nonlinear_krylov import FusedCoordinatePCG


def fused_reference(action, rhs, dot, tolerance, maxiter, xp, project=None,
                    fused=False, precondition=None, initial=None):
    # Benchmark-only injection: reuse the adapter's exact current Q/reference.
    # Fail loudly if that closure's contract changes. Production is untouched.
    if project is not None or initial is not None or not fused or precondition is None:
        raise ValueError('Unexpected linear adapter contract')
    variables = dict(zip(precondition.__code__.co_freevars,
                         (c.cell_contents for c in precondition.__closure__)))
    Q, reference = variables['Q'], variables['reference']
    return FusedCoordinatePCG(Q).solve(action, rhs, dot, tolerance, maxiter, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('Use a new output directory and at least two repeats')
    args.output.mkdir(parents=True)
    phase, ori = np.load(G09_N240/'phase.npy'), np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(MATERIAL.read_text())
    p = optimized_parameters(phase, ori, material, compile_cfield_geometry(phase, ori))
    modes = ['symmetric_fused', 'matrix', 'matrix_fused_candidate']
    report = dict(scope='G09 240^3, same material and rtol=1e-5; six elastic loads. '
        'Warm plans, precompiled geometry, no field export; timing metadata IO included. '
        'No concurrent GPU work. Candidate does not change production defaults.', runs=[], summary={},
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(adapter.__file__),
                ROOT/'FFT/pipeline/nonlinear_krylov.py', ROOT/'FFT/pipeline/linear_krylov.py')})
    reference = None
    for repeat in range(-1, args.repeats):
        order = modes if repeat < 0 else modes[repeat%3:]+modes[:repeat%3]
        for name in order:
            output = args.output/f'{name}_{repeat}.json'
            injection = (patch.object(adapter, '_cg', fused_reference)
                         if name == 'matrix_fused_candidate' else nullcontext())
            with injection, quiet_solver():
                sync(); start = time.perf_counter()
                ceff = solve_homogenization(dict(p, linear_formulation='displacement',
                    linear_preconditioning='symmetric_fused' if name == 'symmetric_fused' else 'matrix',
                    solver_timing_path=str(output)))
                sync(); elapsed = time.perf_counter()-start
            if reference is None:
                reference = ceff
            info = json.loads(output.read_text())['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
            error = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
            residual = max(i['true_norm_res_rel'] for i in info)
            if not np.isfinite(ceff).all() or error > 1e-6 or residual > 1e-5:
                raise RuntimeError('Physical/constitutive audit failed: '+name)
            if repeat < 0:
                continue
            report['runs'].append(dict(mode=name, repeat=repeat, seconds=elapsed,
                Ceff=ceff.tolist(), Ceff_error=error, max_true_residual=residual,
                cg=sum(i['cg_iterations'] for i in info)))
            for mode in modes:
                rows = [r for r in report['runs'] if r['mode'] == mode]
                if rows:
                    report['summary'][mode] = dict(median_seconds=float(np.median([r['seconds'] for r in rows])),
                        cg=[r['cg'] for r in rows], max_Ceff_error=max(r['Ceff_error'] for r in rows))
            (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(name, elapsed, report['runs'][-1]['cg'], flush=True)


if __name__ == '__main__':
    main()
