#!/usr/bin/env python3
"""Separate startup, input preparation, load solves and output assembly on GPU.

Accepts existing phase/orientation arrays and a material JSON through paths.
No geometry-specific kernel, mesh size, orientation or constitutive constants.
The first call uses existing OS/JIT caches; it is not a cold-storage test.
"""
import time
SCRIPT_START = time.perf_counter()

import argparse
from contextlib import redirect_stdout
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
from unittest.mock import patch

import numpy as np
from cufhom.pipeline.fft_solver import compile_cfield_geometry, cp, solve_homogenization
from cufhom.pipeline import linear_coordinates

IMPORT_SECONDS = time.perf_counter() - SCRIPT_START


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--profile-host', action='store_true')
    parser.add_argument('--time-cg', action='store_true',
        help='Time the six inner CG calls with synchronized boundaries')
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error('Use a new output directory and positive repeats')
    args.output.mkdir(parents=True)
    files = dict(phase=args.geometry/'phase.npy', orientation=args.geometry/'ori.npy',
                 material=args.material)
    hash_start = time.perf_counter()
    hashes = {k: sha256(path) for k, path in files.items()}
    hash_seconds = time.perf_counter() - hash_start
    started = time.perf_counter()
    material = json.loads(args.material.read_text())
    material_read_seconds = time.perf_counter() - started
    common = dict(input_dir=str(args.geometry), fft_backend='cupy', solver_rtol=1e-5,
        solver_atol=0., solver_maxiter=1000, require_convergence=True,
        solver_verbose=False, gpu_timing_sync=True,
        **{k: material[k] for k in ('Em', 'nu_m', 'Ef_L', 'Ef_T', 'nu_LT', 'nu_TT', 'G_LT')})
    started = time.perf_counter()
    cp.cuda.get_current_stream().synchronize()
    gpu_init_seconds = time.perf_counter() - started
    reference = None

    def execute(name, repetition, settings):
        nonlocal reference
        path = args.output/f'{name}_{repetition}.json'
        cg_seconds = []
        original_cg = linear_coordinates.fused_cg
        def timed_cg(*a, **kw):
            cp.cuda.get_current_stream().synchronize()
            started_cg = time.perf_counter()
            result = original_cg(*a, **kw)
            cp.cuda.get_current_stream().synchronize()
            cg_seconds.append(time.perf_counter()-started_cg)
            return result
        cp.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        with patch.object(linear_coordinates, 'fused_cg', timed_cg if args.time_cg else original_cg), \
                open(os.devnull, 'w') as sink, redirect_stdout(sink):
            result = solve_homogenization(dict(common, **settings, solver_timing_path=str(path)))
        cp.cuda.get_current_stream().synchronize()
        wall = time.perf_counter()-started
        detail = json.loads(path.read_text())
        app = detail['cufhom_core_application_timing']
        primal = app['primaldual']['primal']
        infos = primal['load_solver_infos']
        if len(infos) != 6 or not all(i['true_residual_converged'] and
                i['true_norm_res_rel'] <= common['solver_rtol'] for i in infos):
            raise RuntimeError('Physical residual audit failed')
        if reference is None:
            reference = result.copy()
        error = float(np.linalg.norm(result-reference)/np.linalg.norm(reference))
        if not np.isfinite(error) or error > 1e-6:
            raise RuntimeError(f'Ceff changed: {error}')
        stages = dict(input_loading=detail['input_loading_s'],
            material_build_and_upload=detail['cfield_build_s']+detail['cfield_pack_s'],
            projection=app['projection_s'], material_operator=primal['material_operator_s'],
            six_load_solves=primal['load_solve_s'],
            effective_tensor_assembly=primal['postprocess_s'],
            output_postprocessing=detail['problem_postprocessing_s'])
        stages['other_including_timing_json'] = wall - sum(stages.values())
        if args.time_cg:
            if len(cg_seconds) != 6:
                raise RuntimeError('Expected six symmetric CG calls')
            loads = stages.pop('six_load_solves')
            stages['inner_cg'] = sum(cg_seconds)
            stages['rhs_coordinates_audits_macro'] = loads-sum(cg_seconds)
        row = dict(mode=name, repetition=repetition, wall_s=wall, stages_s=stages,
            cg=sum(i['cg_iterations'] for i in infos),
            Ceff_relative_difference=error,
            max_physical_residual=max(i['true_norm_res_rel'] for i in infos))
        print(f'{name} {repetition}: {wall:.6f}s, {row["cg"]} CG', flush=True)
        return row

    first_call = execute('disk', 'first_call', {})
    started = time.perf_counter()
    phase, ori = np.load(files['phase']), np.load(files['orientation'])
    phase.flags.writeable = ori.flags.writeable = False
    preload_seconds = time.perf_counter()-started
    started = time.perf_counter()
    compiled = compile_cfield_geometry(phase, ori)
    compile_seconds = time.perf_counter()-started
    variants = dict(disk={}, preloaded=dict(phase_array=phase, ori_array=ori),
        compiled=dict(phase_array=phase, ori_array=ori, compiled_cfield_geometry=compiled))
    warmups = [execute(name, 'warmup', settings) for name, settings in variants.items()]
    runs = []
    names = list(variants)
    for repetition in range(args.repeats):
        for name in names[repetition % len(names):]+names[:repetition % len(names)]:
            runs.append(execute(name, repetition, variants[name]))
    profiles = {}
    if args.profile_host:
        for name in ('disk', 'compiled'):
            profile = cProfile.Profile()
            profile.enable()
            execute(name, 'host_profile', variants[name])
            profile.disable()
            # Host cumulative times include waits for GPU; never sum parents and children.
            with (args.output/f'{name}_host_profile.txt').open('w') as stream:
                stats = pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative')
                stats.print_stats(40)
                stats.sort_stats('tottime').print_stats(30)
            profiles[name] = f'{name}_host_profile.txt'
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    repo = Path(__file__).resolve().parents[1]
    report = dict(
        scope='Six loads, full Ceff. GPU stage boundaries synchronized. No field export. '
              'Disk mode loads existing arrays and rebuilds geometry groups on every call. '
              'Compiled mode reuses explicit immutable geometry preparation. '
              'First call in this process retains existing filesystem and JIT caches. '
              'Load solves include RHS, coordinate setup, CG, physical audits and macro addition.',
        import_s=IMPORT_SECONDS, gpu_context_init_s=gpu_init_seconds,
        inner_cg_instrumented=args.time_cg,
        material_json_read_s=material_read_seconds, input_hashing_s=hash_seconds,
        geometry_preload_s=preload_seconds, geometry_compile_s=compile_seconds,
        first_call=first_call, warmups=warmups, runs=runs, host_profiles=profiles,
        median_s={name: float(np.median([r['wall_s'] for r in runs if r['mode']==name]))
                  for name in names},
        median_stages_s={name: {key: float(np.median([r['stages_s'][key]
            for r in runs if r['mode']==name])) for key in runs[0]['stages_s']} for name in names},
        shape=list(phase.shape), gpu=device.decode() if isinstance(device, bytes) else device,
        cupy=cp.__version__, input_sha256=hashes,
        inputs_unchanged=hashes == {k: sha256(path) for k, path in files.items()},
        source_sha256={str(p.relative_to(repo)): sha256(p) for p in (repo/'src').rglob('*.py')},
        benchmark_sha256=sha256(Path(__file__)))
    if not report['inputs_unchanged']:
        raise RuntimeError('Inputs changed')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['median_s']), flush=True)


if __name__ == '__main__':
    main()
