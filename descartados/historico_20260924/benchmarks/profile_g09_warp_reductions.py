#!/usr/bin/env python3
"""Same-grid CG experiment: atomic, shared-tree and warp-tree reductions.

Chooses the warp block count by isolated GPU timing, then validates full G09
solves. Overrides only the optional reduction class inside this process.
"""
import argparse
import json
import time
import numpy as np
from cupyx.profiler import benchmark

from profile_g09_fft import (
    FFT_ROOT, G09_N240, MATERIAL, compile_cfield_geometry, cp,
    optimized_parameters, quiet_solver, solve_homogenization, sync,
)
from ffthompy.tensors import Tensor
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction
import ffthompy.general.cupy_cg_reduction as reductions
import ffthompy.applications as applications


def tune():
    tensors = [Tensor(val=cp.full((6, 240, 240, 240), .25, dtype=cp.float32),
                      order=1, fft_form='r') for _ in range(4)]
    x, p, r, ap = tensors
    alpha = cp.asarray(.01, dtype=cp.float32)
    samples = {}
    for blocks in (1024, 4096, 16384, 65536):
        reduction = CupyWarpCGReduction(x, max_blocks=blocks)
        dot = benchmark(reduction.dot, (p, ap), n_repeat=10, n_warmup=3)
        update = benchmark(reduction.update_xr_rr, (x, alpha, p, r, ap), n_repeat=10, n_warmup=3)
        samples[blocks] = {'dot_gpu_s': float(dot.gpu_times.mean()),
                           'update_gpu_s': float(update.gpu_times.mean())}
        print('micro', blocks, samples[blocks], flush=True)
    best = min(samples, key=lambda b: sum(samples[b].values()))
    return best, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('at least two repetitions required')
    blocks, micro = tune()
    print('SELECTED', blocks, flush=True)
    class TunedWarp(CupyWarpCGReduction):
        def __init__(self, tensor):
            super().__init__(tensor, max_blocks=blocks)
    original_reduction = reductions.CupyCGReduction
    original_solver = applications.linear_solver
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.setflags(write=False); ori.setflags(write=False)
    config = optimized_parameters(phase, ori, json.loads(MATERIAL.read_text()),
                                  compile_cfield_geometry(phase, ori))
    infos, reference_fields, field_errors = [], [], []
    audit = False; save = False
    def observe(*args, **kwargs):
        x, info = original_solver(*args, **kwargs)
        infos.append(info)
        if audit:
            if save:
                reference_fields.append(x.val.copy())
            else:
                ref = reference_fields[len(field_errors)]
                field_errors.append(float((cp.linalg.norm(x.val-ref)/cp.linalg.norm(ref)).item()))
        return x, info
    def solve(mode):
        reductions.CupyCGReduction = TunedWarp if mode == 'warp' else original_reduction
        infos.clear(); field_errors.clear()
        with quiet_solver():
            return solve_homogenization(dict(config, cupy_stable_reductions=mode != 'atomic',
                                             check_true_residual=audit))
    modes = ('atomic', 'tree', 'warp')
    report = {'grid': list(phase.shape), 'rtol': config['solver_rtol'],
              'warp_max_blocks': blocks, 'microbenchmarks': micro,
              'variants': {m: {'wall_s': [], 'iterations': [], 'Ceff': []} for m in modes}}
    applications.linear_solver = observe
    try:
        for mode in modes:
            solve(mode)
            print('warmup', mode, flush=True)
        for repeat in range(args.repeats):
            for mode in (modes if repeat % 2 == 0 else modes[::-1]):
                sync(); started = time.perf_counter()
                c = solve(mode)
                sync(); elapsed = time.perf_counter()-started
                item = report['variants'][mode]
                item['wall_s'].append(elapsed); item['Ceff'].append(c.tolist())
                item['iterations'].append(sum(i['kit'] for i in infos))
                print(mode, elapsed, item['iterations'][-1], flush=True)
        reference = np.asarray(report['variants']['atomic']['Ceff'][0])
        for v in report['variants'].values():
            v['median_s'] = float(np.median(v['wall_s']))
            v['max_relative_Ceff_error'] = max(float(np.linalg.norm(np.asarray(c)-reference)/np.linalg.norm(reference)) for c in v['Ceff'])
        audit = True; save = True
        solve('atomic')
        save = False
        for mode in modes:
            solve(mode)
            report['variants'][mode]['audit'] = {
                'max_true_residual': max(i['true_norm_res_rel'] for i in infos),
                'all_true_converged': all(i['true_residual_converged'] for i in infos),
                'field_relative_differences': list(field_errors),
            }
            print('audit', mode, report['variants'][mode]['audit'], flush=True)
        output = FFT_ROOT/'benchmarks/results/g09_warp_reductions.json'
        output.write_text(json.dumps(report, indent=2)+'\n')
        noise = max(report['variants']['atomic']['audit']['field_relative_differences'])
        for v in report['variants'].values():
            assert v['audit']['all_true_converged']
            assert v['max_relative_Ceff_error'] < 1e-6
            assert max(v['audit']['field_relative_differences']) < max(2*noise, 1e-6)
        print('SAVED', output, flush=True)
    finally:
        reductions.CupyCGReduction = original_reduction
        applications.linear_solver = original_solver


if __name__ == '__main__':
    main()
