#!/usr/bin/env python3
"""Six-load linear GPU benchmark; explicit, read-only geometry/material inputs."""
import argparse
from contextlib import nullcontext, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from cufhom.pipeline.fft_solver import compile_cfield_geometry, cp, solve_homogenization


VERIFIED_SETTINGS = dict(
    solver_real_dtype='float32', solver_fft_form='r', cfield_origin='zero',
    cfield_storage='sym21', cfield_indexed=True, projection_storage='direct',
    projection_backend='numpy', cupy_plan_mode='manual', cupy_preserve_plans=True,
    cupy_fused_matvec=True, cupy_unscaled_fft_pair=True, cupy_lazy_scalars=True,
    cupy_fused_cg_updates=True, cupy_fused_xr_rr=True, cupy_fused_dot=True,
    cupy_reuse_inverse_input=True, keep_solution_on_device=True,
    internal_load_workers=1, load_batch_size=1, postprocess_batch_size=1,
    postprocess_assembly='scalar', force_disk_cfield=False,
    linear_formulation='displacement', linear_preconditioning='symmetric_fused',
    linear_kernel_fusion=False,  # frozen baseline before the 2026-09-29 fusions
    linear_residual_smoothing=False,  # frozen baseline before the 2026-09-30 smoothing
    linear_fft_autotune=False, linear_energy_update=False,  # preserve this historical baseline
)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--material', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--defaults', action='store_true', help='Compare minimal GPU parameters too')
    p.add_argument('--compare-smoothing', action='store_true',
                   help='Compare current defaults with only residual smoothing disabled')
    return p


def run(args, variants=None):
    if args.output.exists() or args.repeats < 1:
        raise ValueError('Use a new output directory and positive repetition count')
    args.output.mkdir(parents=True)
    files = dict(phase=args.geometry/'phase.npy', orientation=args.geometry/'ori.npy',
                 material=args.material)
    hashes = {k: sha256(v) for k, v in files.items()}
    start = time.perf_counter()
    phase = np.load(files['phase']); ori = np.load(files['orientation'])
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(args.material.read_text())
    compiled = compile_cfield_geometry(phase, ori)
    preparation_s = time.perf_counter()-start
    common = dict(input_dir=str(args.geometry), phase_array=phase, ori_array=ori,
        compiled_cfield_geometry=compiled, fft_backend='cupy', solver_rtol=1e-5,
        solver_atol=0., solver_maxiter=1000, require_convergence=True, solver_verbose=False,
        **{k: material[k] for k in ('Em', 'nu_m', 'Ef_L', 'Ef_T', 'nu_LT', 'nu_TT', 'G_LT')})
    if variants is None:
        if args.compare_smoothing:
            variants = dict(previous=({'linear_residual_smoothing': False}, nullcontext),
                            defaults=({}, nullcontext))
        else:
            variants = dict(verified=(VERIFIED_SETTINGS, nullcontext))
            if args.defaults:
                variants['defaults'] = ({}, nullcontext)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    report = dict(shape=list(phase.shape), vf=float(phase.mean()), material=material,
        paths={k: str(v.resolve()) for k, v in files.items()}, input_sha256=hashes,
        geometry_preparation_s=preparation_s, gpu=device.decode() if isinstance(device, bytes) else device,
        cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        rtol=common['solver_rtol'], load_ids=list(range(6)),
        scope='Six unit Mandel strain loads, full Ceff, physical residual audits. '
              'Warm solves include material setup, solve, postprocessing and timing JSON. '
              'Geometry loading/compilation and input hashing are measured separately.',
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
            for p in (Path(__file__).resolve().parents[1]/'src').rglob('*.py')},
        benchmark_sha256=sha256(Path(__file__)),
        settings={name: settings for name, (settings, _) in variants.items()}, runs=[], warmups=[])
    reference = None
    def execute(name, rep):
        nonlocal reference
        settings, context = variants[name]
        timing = args.output/f'{name}_{rep}.json'
        with context():
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            with open(os.devnull, 'w') as sink, redirect_stdout(sink):
                C = solve_homogenization(dict(common, **settings, solver_timing_path=str(timing)))
            cp.cuda.get_current_stream().synchronize()
            wall = time.perf_counter()-start
        detail = json.loads(timing.read_text())
        infos = detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        if len(infos) != 6 or not all(i['true_residual_converged'] and
                i['true_norm_res_rel'] <= common['solver_rtol'] for i in infos):
            raise RuntimeError('Failed six-load physical residual audit')
        if reference is None:
            reference = C.copy()
        error = float(np.linalg.norm(C-reference)/np.linalg.norm(reference))
        if not np.isfinite(error) or error > 1e-6:
            raise RuntimeError(f'Ceff mismatch: {error}')
        row = dict(variant=name, repetition=rep, wall_s=wall,
            cg=sum(i['cg_iterations'] for i in infos), Ceff=C.tolist(), Ceff_relative_difference=error,
            max_physical_residual=max(i['true_norm_res_rel'] for i in infos),
            linear_formulation=detail['linear_formulation'])
        print(f'{name} {rep}: {wall:.6f}s, {row["cg"]} CG, residual={row["max_physical_residual"]:.3g}', flush=True)
        return row
    for name in variants:
        report['warmups'].append(execute(name, 'warmup'))
    names = list(variants)
    for rep in range(args.repeats):
        # Reverse alternating pairs; rotate larger suites to distribute drift.
        order = names[rep % len(names):]+names[:rep % len(names)]
        for name in order:
            report['runs'].append(execute(name, rep))
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    report['median_s'] = {name: float(np.median([r['wall_s'] for r in report['runs']
                                              if r['variant'] == name])) for name in names}
    report['inputs_unchanged'] = hashes == {k: sha256(v) for k, v in files.items()}
    if not report['inputs_unchanged']:
        raise RuntimeError('Input files changed during benchmark')
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['median_s']), flush=True)
    return report


if __name__ == '__main__':
    run(parser().parse_args())
