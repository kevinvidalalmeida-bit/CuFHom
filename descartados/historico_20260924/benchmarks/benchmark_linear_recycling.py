"""Full six-load G09 recycling trials, setup included and defaults untouched."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, cp, compile_cfield_geometry,
    optimized_parameters, quiet_solver, solve_homogenization, sync)
from linear_recycling_candidate import candidate


def parse_variant(name):
    if name == 'baseline':
        return dict(mode='baseline')
    match = re.fullmatch(r'(spectral|balanced|seed)_w(\d+)_r(\d+)(?:_(low|high))?(_rolling)?', name)
    if match is None:
        raise ValueError('Use e.g. spectral_w12_r4 or balanced_w8_r2_low')
    mode, window, rank, which, rolling = match.groups()
    if not 1 <= int(rank) <= int(window) <= 20:
        raise ValueError('Require 1 <= rank <= window <= 20')
    return dict(mode=mode, window=int(window), rank=int(rank), which=which or 'high', rolling=bool(rolling))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--variants', nargs='+', default=['baseline', 'spectral_w8_r2',
        'spectral_w12_r4', 'balanced_w12_r4', 'seed_w12_r4'])
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output directory and positive repeat count')
    if args.variants[0] != 'baseline' or len(set(args.variants)) != len(args.variants):
        parser.error('Put baseline first; no duplicate variants')
    specs = {name: parse_variant(name) for name in args.variants}
    args.output.mkdir(parents=True)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(MATERIAL.read_text())
    config = optimized_parameters(phase, ori, material, compile_cfield_geometry(phase, ori))
    config.update(linear_formulation='displacement', linear_preconditioning='symmetric_fused')
    paths = ['FFT/pipeline/linear_coordinates.py', 'FFT/pipeline/linear_krylov.py',
        'FFT/pipeline/nonlinear_coordinates.py', 'FFT/pipeline/nonlinear_solver.py',
        'FFT/pipeline/fft_solver.py', 'FFT/ffthompy_core/ffthompy/ffthompy/applications.py',
        'FFT/ffthompy_core/ffthompy/ffthompy/tensors/fft.py',
        'FFT/ffthompy_core/ffthompy/ffthompy/postprocess.py',
        'FFT/benchmarks/linear_recycling_candidate.py',
        'FFT/benchmarks/benchmark_linear_recycling.py']
    hashes = {p:sha(ROOT/p) for p in paths}
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    report = dict(scope='Warm sequential full G09 240^3 six-load solves. Basis built anew '
        'from first load; construction, AU, audits, corrections and timing-metadata IO INCLUDED. '
        'Geometry preloaded/precompiled, no field export. No nonlinear or production changes. '
        'One repeat is a pilot, not a speed claim; basis timings are host intervals including waits.',
        device=device.decode() if isinstance(device, bytes) else device, cupy=cp.__version__,
        shape=list(phase.shape), rtol=config['solver_rtol'], material=material,
        source_sha256_before=hashes, input_sha256={str(p):sha(p) for p in
            [G09_N240/'phase.npy', G09_N240/'ori.npy', MATERIAL]},
        variants=specs, warmups=[], runs=[], summary={})
    reference = None
    for repeat in range(-1, args.repeats):
        order = args.variants if repeat < 0 else args.variants[repeat%len(specs):]+args.variants[:repeat%len(specs)]
        for name in order:
            gc.collect()
            timing = args.output/f'{name}_{repeat}.json'
            entry = dict(variant=name, repeat=repeat)
            sync(); started = time.perf_counter()
            try:
                with candidate(**specs[name]) as (infos, state), quiet_solver():
                    ceff = solve_homogenization(dict(config, solver_timing_path=str(timing)))
                    sync()
                    entry['wall_s'] = time.perf_counter()-started
                    if reference is None:
                        reference = ceff.copy()
                    error = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
                    residual = max(i['true_norm_res_rel'] for i in infos)
                    entry.update(Ceff=ceff.tolist(), relative_Ceff_difference=error,
                        max_physical_residual=residual, loads=infos, recycling=state,
                        cg=sum(i['coordinate_iterations']+i['correction_iterations'] for i in infos),
                        physical_passed=len(infos)==6 and all(i['true_residual_converged']
                            and i['true_norm_res_rel'] <= config['solver_rtol'] for i in infos),
                        Ceff_passed=bool(np.isfinite(error) and error <= 1e-6),
                        pool_reserved_after_solve_bytes=cp.get_default_memory_pool().total_bytes())
            except Exception as error:
                sync(); entry.update(wall_s=time.perf_counter()-started, error=repr(error),
                                     physical_passed=False, Ceff_passed=False)
            report['warmups' if repeat < 0 else 'runs'].append(entry)
            print(repeat, name, entry['wall_s'], entry.get('cg'), entry.get('relative_Ceff_difference'),
                  entry.get('error', ''), flush=True)
            if name == 'baseline' and (not entry['physical_passed'] or not entry['Ceff_passed']):
                (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                raise RuntimeError('Baseline failed; cannot compare candidates')
            for mode in specs:
                runs = [r for r in report['runs'] if r['variant'] == mode]
                if runs:
                    report['summary'][mode] = dict(median_s=float(np.median([r['wall_s'] for r in runs])),
                        cg=[r.get('cg') for r in runs],
                        passed=all(r['physical_passed'] and r['Ceff_passed'] for r in runs))
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            # Free only unused experimental arrays, not FFT plans; outside timing.
            gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    report['source_sha256_after'] = {p:sha(ROOT/p) for p in paths}
    report['sources_unchanged'] = report['source_sha256_after'] == hashes
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2))
    if not report['sources_unchanged']:
        raise RuntimeError('Sources changed during measurements')


if __name__ == '__main__':
    main()
