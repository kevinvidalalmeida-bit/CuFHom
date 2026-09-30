"""Separate allocation-hook profile; never use these timings as speed results."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, cp, sync, quiet_solver,
    optimized_parameters, compile_cfield_geometry, solve_homogenization)
from linear_strategy_candidates import candidate


class PoolPeak(cp.cuda.MemoryHook):
    name = 'fft_linear_pool_peak'
    def __init__(self):
        self.start = cp.get_default_memory_pool().used_bytes()
        self.peak = self.start
    def malloc_postprocess(self, **kwargs):
        self.peak = max(self.peak, cp.get_default_memory_pool().used_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['combined', 'polarization_macro', 'polarization_local'])
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    config = optimized_parameters(phase, ori, json.loads(MATERIAL.read_text()),
                                  compile_cfield_geometry(phase, ori))
    paths = ['FFT/benchmarks/profile_linear_memory.py', 'FFT/benchmarks/linear_strategy_candidates.py',
             'FFT/benchmarks/linear_polarization_candidate.py', 'FFT/pipeline/linear_coordinates.py',
             'FFT/pipeline/linear_krylov.py']
    report = dict(scope='Peak live bytes in default CuPy memory pool, observed at every allocation; '
        'includes its FFT buffers. Excludes CUDA context, non-pool library allocations, '
        'and unused reserved pool chunks. Instrumented run, not a timing benchmark.',
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}, runs={})
    for mode in args.modes:
        with candidate(mode), quiet_solver():
            solve_homogenization(config)
        gc.collect(); sync()
        hook = PoolPeak()
        with hook, candidate(mode) as (infos, cache), quiet_solver():
            ceff = solve_homogenization(config)
            sync()
        report['runs'][mode] = dict(start_pool_bytes=hook.start, peak_live_pool_bytes=hook.peak,
            cg=sum(i['coordinate_iterations']+i['correction_iterations'] for i in infos),
            max_physical_residual=max(i['true_norm_res_rel'] for i in infos), Ceff=ceff.tolist())
        print(mode, report['runs'][mode]['peak_live_pool_bytes'], flush=True)
        del cache, infos
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
