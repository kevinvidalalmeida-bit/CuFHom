#!/usr/bin/env python3
"""Paired G09 N240 regression against the pre-cleanup CG, loaded read-only.

Only the CG module and elasticity load initializer come from the pinned git
revision; FFT/material/projection code and configuration are shared. No git
checkout or geometry writes. Audits/field comparisons are outside timed runs.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types

import numpy as np
from profile_g09_fft import (
    ROOT, FFT_ROOT, G09_N240, MATERIAL, compile_cfield_geometry, cp,
    optimized_parameters, quiet_solver, solve_homogenization, sync,
)
import ffthompy.applications as applications

REFERENCE = 'a42304a7ecf4a638053413f2eef103ec79589305'
CORE = 'FFT/ffthompy_core/ffthompy/ffthompy'


def reference_implementation(revision):
    def source(path):
        return subprocess.check_output(
            ['git', 'show', f'{revision}:{CORE}/{path}'], cwd=ROOT, text=True)
    module = types.ModuleType('reference_cg')
    exec(compile(source('general/solver.py'), '<reference_cg>', 'exec'), vars(module))
    tree = ast.parse(source('applications.py'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'solve_load_elasticity')
    namespace = dict(vars(applications))
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 '<reference_load>', 'exec'), namespace)
    return module.linear_solver, namespace['solve_load_elasticity']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--reference', default=REFERENCE)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error('at least two repeats are required')
    original = applications.linear_solver, applications.solve_load_elasticity
    reference = reference_implementation(args.reference)
    phase = np.load(G09_N240/'phase.npy'); ori = np.load(G09_N240/'ori.npy')
    phase.setflags(write=False); ori.setflags(write=False)
    material = json.loads(MATERIAL.read_text())
    config = optimized_parameters(phase, ori, material, compile_cfield_geometry(phase, ori))
    fields, errors, infos, operator_calls = [], [], [], []
    audit = False
    save_fields = False

    @contextmanager
    def variant(mode):
        linear, load = reference if mode == 'before' else original
        def observe(*positional, **kwargs):
            count = 0
            afun = kwargs['Afun']
            def counted(x):
                nonlocal count
                count += 1
                return afun(x)
            kwargs['Afun'] = counted
            x, info = linear(*positional, **kwargs)
            infos.append(info); operator_calls.append(count)
            if audit:
                val = cp.asarray(x.val)
                if save_fields:
                    fields.append(val.copy())
                else:
                    ref = fields[len(errors)]
                    delta = val-ref
                    errors.append({
                        'relative_l2': float((cp.linalg.norm(delta)/cp.linalg.norm(ref)).item()),
                        'relative_linf': float((cp.max(cp.abs(delta))/cp.max(cp.abs(ref))).item()),
                    })
            return x, info
        applications.linear_solver = observe
        applications.solve_load_elasticity = load
        try:
            yield
        finally:
            applications.linear_solver, applications.solve_load_elasticity = original

    def solve(mode, **extra):
        infos.clear(); operator_calls.clear(); errors.clear()
        settings = dict(config, cupy_stable_reductions=mode == 'after_stable', **extra)
        with variant(mode), quiet_solver():
            return solve_homogenization(settings)

    modes = ('before', 'after', 'after_stable')
    for mode in modes:
        solve(mode)
        print('warmup', mode, flush=True)
    report = {
        'reference_revision': args.reference, 'grid': list(phase.shape),
        'solver_rtol': config['solver_rtol'], 'material': material,
        'timing': 'warm six-load solve; excludes geometry IO, audits and profiling',
        'variants': {m: {'wall_s': [], 'iterations': [], 'CG_operator_calls': [],
                         'Ceff': []} for m in modes},
        'core_sha256': {p: hashlib.sha256((ROOT/CORE/p).read_bytes()).hexdigest()
                        for p in ('general/solver.py', 'applications.py', 'tensors/operators.py')},
    }
    for repeat in range(args.repeats):
        for mode in (modes if repeat % 2 == 0 else modes[::-1]):
            sync(); started = time.perf_counter()
            c = solve(mode)
            sync(); wall = time.perf_counter()-started
            item = report['variants'][mode]
            item['wall_s'].append(wall)
            item['Ceff'].append(c.tolist())
            item['iterations'].append(sum(i['kit'] for i in infos))
            item['CG_operator_calls'].append(sum(operator_calls))
            print(mode, round(wall, 6), item['iterations'][-1], flush=True)
    reference_ceff = np.asarray(report['variants']['before']['Ceff'][0])
    for item in report['variants'].values():
        item['median_wall_s'] = float(np.median(item['wall_s']))
        item['max_relative_Ceff_difference'] = max(float(
            np.linalg.norm(np.asarray(c)-reference_ceff)/np.linalg.norm(reference_ceff))
            for c in item['Ceff'])

    audit = True
    report['audits'] = {}
    for label, mode in [('before_reference', 'before'), ('before_repeat', 'before'),
                        ('after', 'after'), ('after_stable', 'after_stable')]:
        save_fields = label == 'before_reference'
        solve(mode, check_true_residual=True, profile_cg_timing=True)
        report['audits'][label] = {
            'field_differences': list(errors),
            'max_true_relative_residual': max(i['true_norm_res_rel'] for i in infos),
            'all_true_converged': all(i['true_residual_converged'] for i in infos),
            'CG_operator_calls': sum(operator_calls),
            'direction_updates': sum(i['cg_profile']['update_p']['calls'] for i in infos),
        }
        print('audit', label, report['audits'][label], flush=True)
    output = FFT_ROOT/'benchmarks/results/g09_cg_cleanup.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    assert all(a['all_true_converged'] for a in report['audits'].values())
    assert all(v['max_relative_Ceff_difference'] < 1e-6 for v in report['variants'].values())
    noise = max(e['relative_l2'] for e in report['audits']['before_repeat']['field_differences'])
    for mode in ('after', 'after_stable'):
        assert max(e['relative_l2'] for e in report['audits'][mode]['field_differences']) < max(2*noise, 1e-6)
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    main()
