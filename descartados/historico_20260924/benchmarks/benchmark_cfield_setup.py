"""Paired G09 indexed-field setup and six-load solves against a pinned revision."""
import argparse
import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
                             quiet_solver, sync)
import pipeline.fft_solver as solver
import ffthompy.applications as applications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='a2154c1')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('Use a new output file and at least two repeats')
    relative = 'FFT/pipeline/fft_solver.py'
    source = subprocess.check_output(['git', 'show', f'{args.reference}:{relative}'], cwd=ROOT, text=True)
    tree = ast.parse(source)
    namespace = dict(vars(solver))
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                n.name in ('compile_cfield_geometry', '_build_cfield_gpu')]
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<reference>', 'exec'), namespace)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(MATERIAL.read_text())
    original = solver._build_cfield_gpu
    builds = {'before': namespace['_build_cfield_gpu'], 'after': original}
    layouts, compile_times = {}, {}
    for name, compile_fn in [('before', namespace['compile_cfield_geometry']),
                             ('after', solver.compile_cfield_geometry)]:
        start = time.perf_counter(); layouts[name] = compile_fn(phase, ori)
        compile_times[name] = time.perf_counter()-start
    configurations = {name: optimized_parameters(phase, ori, material, layout)
                      for name, layout in layouts.items()}
    # Isolate field-setup changes from the new default Krylov formulation.
    for config in configurations.values():
        config['linear_formulation'] = 'strain'
    report = dict(reference=args.reference, shape=list(phase.shape), repeats=args.repeats,
                  source_sha256=hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest(),
                  geometry_compile_s=compile_times, variants={})
    reference_payload = None
    samples = []
    active = 'before'
    audit_payload = True

    def measured(*a, **kw):
        nonlocal reference_payload
        sync(); start = time.perf_counter()
        out = builds[active](*a, **kw)
        sync(); samples.append(time.perf_counter()-start)
        if audit_payload and reference_payload is None:
            reference_payload = tuple(x.copy() for x in out[0])
        elif audit_payload:
            # Exact table/map equivalence: material actions are unchanged.
            for candidate, reference in zip(out[0], reference_payload):
                solver.cp.testing.assert_array_equal(candidate, reference)
        return out

    solver._build_cfield_gpu = measured
    curves = {}
    try:
        for active in builds:
            with quiet_solver():
                solver.solve_homogenization(configurations[active])
            report['variants'][active] = dict(wall_s=[], cfield_s=[])
        audit_payload = False
        for repeat in range(args.repeats):
            for active in (list(builds) if repeat % 2 == 0 else list(builds)[::-1]):
                samples.clear(); sync(); start = time.perf_counter()
                with quiet_solver():
                    curves[active] = solver.solve_homogenization(configurations[active])
                sync(); elapsed = time.perf_counter()-start
                v = report['variants'][active]
                v['wall_s'].append(elapsed); v['cfield_s'].append(samples[0])
                print(active, elapsed, samples[0], flush=True)
        # Independent true-residual audit, excluded from timings.
        original_linear = applications.linear_solver
        infos = []
        def observe(*a, **kw):
            result, info = original_linear(*a, **kw)
            infos.append(info)
            return result, info
        applications.linear_solver = observe
        report['true_residual_audits'] = {}
        try:
            for active in builds:
                infos.clear()
                with quiet_solver():
                    solver.solve_homogenization(dict(configurations[active], check_true_residual=True))
                report['true_residual_audits'][active] = dict(
                    loads=len(infos),
                    maximum=max(i['true_norm_res_rel'] for i in infos),
                    passed=len(infos) == 6 and all(i['true_residual_converged'] for i in infos))
        finally:
            applications.linear_solver = original_linear
    finally:
        solver._build_cfield_gpu = original
    error = np.linalg.norm(curves['after']-curves['before'])/np.linalg.norm(curves['before'])
    report['relative_Ceff_difference'] = float(error)
    report['indexed_payload_equal'] = True
    for v in report['variants'].values():
        v['median_wall_s'] = float(np.median(v['wall_s']))
        v['median_cfield_s'] = float(np.median(v['cfield_s']))
    report['timing_scope'] = 'Warm six-load solve, preloaded compiled geometry; payload/residual audits outside timing; no field export.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if error > 1e-6 or not all(a['passed'] for a in report['true_residual_audits'].values()):
        raise RuntimeError('Ceff or true-residual regression')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
