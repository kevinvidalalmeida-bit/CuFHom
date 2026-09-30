"""Bracket CUDA-event profiling of the unchanged G09 40-increment curve."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from profile_g09_fft import G09_N240, MATERIAL, sync, cp
from profile_linear_stages import EventProfile
from pipeline import nonlinear_solver as solver
from pipeline.nonlinear_material import CuPyMaterial
from pipeline.nonlinear_coordinates import DisplacementCoordinates as Coordinates
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from ffthompy.tensors import fft
from compare_nonlinear_curves import compare_runs


def parameters():
    material = json.loads(MATERIAL.read_text())
    return dict(phase_array=np.load(G09_N240/'phase.npy', mmap_mode='r'),
        ori_array=np.load(G09_N240/'ori.npy', mmap_mode='r'), backend='cupy', dtype='float32',
        matrix=dict(E=material['Em'], nu=material['nu_m'], yield_stress=.015, hardening=.03),
        fiber=dict(E_L=material['Ef_L'], E_T=material['Ef_T'], G_LT=material['G_LT'],
                   nu_LT=material['nu_LT'], nu_TT=material['nu_TT']),
        path=[0., .01], initial_increment=.00025, rtol=1e-5,
        formulation='displacement', preconditioner='matrix')


def install(profiler, stack):
    for cls, attrs in [(Coordinates, ['strain', 'adjoint', '_map', 'precondition']),
                       (CuPyMaterial, ['update', 'tangent']),
                       (solver.MixedProjector, ['__call__'])]:
        for attr in attrs:
            stack.enter_context(patch.object(cls, attr,
                profiler.wrap(cls.__name__+'.'+attr, getattr(cls, attr))))
    for attr in ['rfftn_cached', 'irfftn_unscaled_cached', 'irfftn_cached']:
        stack.enter_context(patch.object(fft, attr, profiler.wrap(attr, getattr(fft, attr))))
    original = solver._cg
    def cg(action, rhs, dot, *args, **kwargs):
        return profiler.wrap('cg', original)(profiler.wrap('operator', action), rhs,
            profiler.wrap('dot_and_host_sync', dot), *args, **kwargs)
    stack.enter_context(patch.object(solver, '_cg', cg))
    original_fused = FusedCoordinatePCG.solve
    def fused(engine, action, rhs, dot, *args, **kwargs):
        return profiler.wrap('cg', original_fused)(engine, profiler.wrap('operator', action),
            rhs, profiler.wrap('dot_and_host_sync', dot), *args, **kwargs)
    stack.enter_context(patch.object(FusedCoordinatePCG, 'solve', fused))
    for attr in ('update', 'direction'):
        stack.enter_context(patch.object(FusedCoordinatePCG, attr,
            profiler.wrap('fused_pcg_'+attr, getattr(FusedCoordinatePCG, attr))))
    updates = solver._cg_updates()
    wrapped = tuple(profiler.wrap(name, fun) for name,fun in zip(('cg_update_xr','cg_update_p'),updates))
    stack.enter_context(patch.object(solver, '_cg_updates', lambda: wrapped))
    stack.enter_context(patch.object(solver, 'commit_state', profiler.wrap('commit', solver.commit_state)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--coordinate-pcg', choices=['classic', 'fused', 'auto'], default='classic')
    parser.add_argument('--reuse-trial-response', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--inner-forcing', default='ew')
    parser.add_argument('--cupy-rfft-batch-mode', choices=['auto', 'full', 'component'], default='auto')
    parser.add_argument('--modes', nargs='+', choices=['plain_before', 'profiled', 'plain_after'],
                        default=['plain_before', 'profiled', 'plain_after'])
    parser.add_argument('--reference', type=Path, help='Optional same-problem metadata for curve audit')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    p = parameters()
    p.update(coordinate_pcg=args.coordinate_pcg, reuse_trial_response=args.reuse_trial_response,
             inner_forcing=args.inner_forcing, cupy_rfft_batch_mode=args.cupy_rfft_batch_mode)
    solver.solve_nonlinear_homogenization(dict(p, path=[0., .00025]))
    report = dict(scope='40 uniform increments, original G09 240^3. Event intervals '
        'include launch/idle gaps, not pure GPU kernel time. Nested inclusive times '
        'must not be summed. Each profiled curve is bracketed by plain curves. '
        'No fields or checkpoints exported; metadata serialization excluded.', runs=[])
    reference = json.loads(args.reference.read_text()) if args.reference else None
    for mode in args.modes:
        profile = EventProfile()
        def progress(row):
            if int(round(row['target_strain']/.00025)) % 10 == 0:
                print(mode, round(row['target_strain'],5), row['seconds'], flush=True)
        with ExitStack() as stack:
            if mode == 'profiled':
                install(profile, stack)
            fn = profile.wrap('solve', solver.solve_nonlinear_homogenization) if mode=='profiled' else solver.solve_nonlinear_homogenization
            sync(); started = time.perf_counter()
            result = fn(dict(p, on_step=progress))
            sync(); wall = time.perf_counter()-started
        # JSON converts tuple lengths to lists. Compare canonical metadata
        # when the reference was loaded from disk, not Python container types.
        serializable = json.loads(json.dumps(result, allow_nan=False))
        directory = args.output/mode; directory.mkdir()
        (directory/'metadata.json').write_text(json.dumps(serializable, indent=2)+'\n')
        if reference is None:
            reference = serializable
        comparison = compare_runs(reference, serializable)
        if not comparison['passed']:
            raise RuntimeError('Instrumentation changed the solution or physical audit failed')
        run = dict(mode=mode, external_wall_s=wall, solver_wall_s=result['wall_seconds'],
            cg=sum(r['cg_iterations'] for r in result['history']),
            newton=sum(r['newton_iterations'] for r in result['history']),
            comparison=comparison, stages=profile.summary() if mode=='profiled' else {})
        report['runs'].append(run)
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(mode, run['solver_wall_s'], run['cg'], flush=True)


if __name__ == '__main__':
    main()
