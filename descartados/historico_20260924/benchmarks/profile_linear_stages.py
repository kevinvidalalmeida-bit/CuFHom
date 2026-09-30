"""Non-mutating solver instrumentation: nested CUDA-event and host timings.

Reports exclusive intervals, not kernel-active time: CUDA events also include
device idle gaps waiting for Python. No synchronization is added per stage.
An uninstrumented run brackets each profile to quantify instrumentation cost.
"""
import argparse
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
    quiet_solver, sync, cp, solve_homogenization, compile_cfield_geometry)
from pipeline import linear_coordinates as linear
from pipeline import linear_krylov
from pipeline.nonlinear_coordinates import DisplacementCoordinates as Coordinates
from ffthompy.tensors import fft


class EventProfile:
    def __init__(self):
        self.records = []
        self.stack = []

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            entry = dict(name=name, children=[], start=cp.cuda.Event(), end=cp.cuda.Event())
            if self.stack:
                self.stack[-1]['children'].append(entry)
            self.records.append(entry)
            self.stack.append(entry)
            entry['start'].record()
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                entry['host_s'] = time.perf_counter()-started
                entry['end'].record()
                self.stack.pop()
        return measured

    def summary(self):
        sync()
        result = {}
        for e in self.records:
            e['gpu_s'] = cp.cuda.get_elapsed_time(e['start'], e['end'])/1000
        for e in self.records:
            item = result.setdefault(e['name'], dict(calls=0, gpu_inclusive_s=0.,
                gpu_exclusive_s=0., host_inclusive_s=0., host_exclusive_s=0.))
            item['calls'] += 1
            for clock in ('gpu', 'host'):
                item[clock+'_inclusive_s'] += e[clock+'_s']
                item[clock+'_exclusive_s'] += e[clock+'_s']-sum(
                    child[clock+'_s'] for child in e['children'])
        return result

    def install(self, stack):
        for attr, label in [('strain', 'coordinates_strain'), ('adjoint', 'coordinates_adjoint'),
                            ('_map', 'coordinate_map'), ('precondition', 'preconditioner')]:
            stack.enter_context(patch.object(Coordinates, attr, self.wrap(label, getattr(Coordinates, attr))))
        for attr in ('rfftn_cached', 'irfftn_unscaled_cached'):
            stack.enter_context(patch.object(fft, attr, self.wrap(attr, getattr(fft, attr))))
        stack.enter_context(patch.object(linear_krylov.SymmetricCoordinates, '_map',
            self.wrap('coordinate_map', linear_krylov.SymmetricCoordinates._map)))
        stack.enter_context(patch.object(linear_krylov, 'fused_update',
            self.wrap('fused_update_and_norm', linear_krylov.fused_update)))
        def wrap_cg(original):
            def cg(action, rhs, dot, *args, **kwargs):
                return self.wrap('cg', original)(self.wrap('operator_action', action), rhs,
                    self.wrap('dot_and_host_sync', dot), *args, **kwargs)
            return cg
        for attr in ('_cg', 'fused_cg'):
            stack.enter_context(patch.object(linear, attr, wrap_cg(getattr(linear, attr))))
        factory = linear.make_linear_solver
        def make(**kwargs):
            kwargs['A'] = self.wrap('constitutive_action', kwargs['A'])
            solve = factory(**kwargs)
            def solver(**parameters):
                parameters['Afun'] = self.wrap('physical_audit_operator', parameters['Afun'])
                return self.wrap('load_solver', solve)(**parameters)
            return solver
        stack.enter_context(patch.object(linear, 'make_linear_solver', make))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output directory and a positive repeat count')
    args.output.mkdir(parents=True)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    config = optimized_parameters(phase, ori, json.loads(MATERIAL.read_text()),
                                  compile_cfield_geometry(phase, ori))
    # Intentionally omit linear_formulation: validate the new automatic default.
    with quiet_solver():
        solve_homogenization(config)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    paths = ['FFT/pipeline/fft_solver.py', 'FFT/pipeline/linear_coordinates.py', 'FFT/pipeline/linear_krylov.py',
             'FFT/pipeline/nonlinear_coordinates.py', 'FFT/pipeline/nonlinear_solver.py',
             'FFT/benchmarks/profile_linear_stages.py']
    report = dict(shape=list(phase.shape), rtol=config['solver_rtol'],
        device=device['name'].decode() if isinstance(device['name'], bytes) else device['name'],
        cupy=cp.__version__, source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        scope='Warm six-load solve; preloaded compiled geometry; no field export. '
              'CUDA-event intervals include idle/launch gaps, not just GPU kernel execution. '
              'Exclusive stage times subtract nested measured children; do not sum inclusive times.',
        runs=[])
    reference = None
    for index in range(args.repeats):
        for mode in ('plain_before', 'profiled', 'plain_after'):
            timing = args.output/f'{index}_{mode}.json'
            profiler = EventProfile()
            with ExitStack() as stack:
                if mode == 'profiled':
                    profiler.install(stack)
                solve = profiler.wrap('total', solve_homogenization) if mode == 'profiled' else solve_homogenization
                sync(); started = time.perf_counter()
                with quiet_solver():
                    ceff = solve(dict(config, solver_timing_path=str(timing)))
                sync(); elapsed = time.perf_counter()-started
            if reference is None:
                reference = ceff
            metadata = json.loads(timing.read_text())
            infos = metadata['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
            if metadata['linear_formulation'] != 'displacement' or len(infos) != 6 or not all(
                    i['true_residual_converged'] and i['true_norm_res_rel'] <= config['solver_rtol'] for i in infos):
                raise RuntimeError('Default selection or physical residual audit failed')
            error = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
            if not np.isfinite(error) or error > 1e-6:
                raise RuntimeError('Instrumentation changed the solution')
            run = dict(mode=mode, wall_s=elapsed, cg=sum(i['cg_iterations'] for i in infos),
                       max_residual=max(i['true_norm_res_rel'] for i in infos), Ceff=ceff.tolist(),
                       difference=error, stages=profiler.summary() if mode == 'profiled' else {})
            report['runs'].append(run)
            print(index, mode, elapsed, run['cg'], flush=True)
    report['plain_median_s'] = float(np.median([r['wall_s'] for r in report['runs'] if r['mode'] != 'profiled']))
    report['profiled_median_s'] = float(np.median([r['wall_s'] for r in report['runs'] if r['mode'] == 'profiled']))
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Saved', args.output/'report.json')


if __name__ == '__main__':
    main()
