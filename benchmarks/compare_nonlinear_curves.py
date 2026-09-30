#!/usr/bin/env python3
"""Audit equivalent nonlinear paths and their measured times, never interpolate branches."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def compare_runs(reference, candidate):
    for key in ('fingerprint', 'configuration', 'backend', 'dtype'):
        if reference[key] != candidate[key]:
            raise ValueError('Not a same-problem latency comparison: '+key)
    for key in ('rtol', 'initial_increment', 'min_increment'):
        if not np.isclose(reference[key], candidate[key], rtol=1e-14, atol=0.):
            raise ValueError('Not a same-problem latency comparison: '+key)
    axis = reference['configuration']['axis']
    rows = [run['history'] for run in (reference, candidate)]
    if len(rows[0]) != len(rows[1]):
        raise ValueError('Different confirmed increments; compare matching branches explicitly')
    targets = [np.array([r.get('target_strain', r['strain'][axis]) for r in h]) for h in rows]
    if not np.allclose(*targets, rtol=0., atol=1e-9):
        raise ValueError('Different loading paths')
    stresses = [np.array([r['stress'] for r in h]) for h in rows]
    strains = [np.array([r['strain'] for r in h]) for h in rows]
    scale = float(reference['stress_scale_GPa'])
    error = float(np.max(np.abs(stresses[1]-stresses[0]))/scale)
    diagnostics = {}
    for label, run, h in zip(('reference', 'candidate'), (reference, candidate), rows):
        diagnostics[label] = dict(wall_seconds=run['wall_seconds'],
            successful_attempt_seconds=sum(r['seconds'] for r in h),
            cg_iterations=sum(r['cg_iterations'] for r in h),
            newton_iterations=sum(r['newton_iterations'] for r in h),
            rejected_steps=len(run['rejected_steps']),
            max_residual=max(r['residual'] for r in h),
            max_macro_residual=max(r['macro_residual'] for r in h),
            max_hill_mandel_error=max(r['hill_mandel_error'] for r in h),
            min_plastic_dissipation=min(r['plastic_dissipation'] for r in h),
            max_yield_violation=max(r['yield_violation'] for r in h),
            final_stress_MPa=h[-1]['stress'][axis]*1000,
            gpu_pool_reserved_bytes=run.get('gpu_pool_reserved_bytes'),
            cg_projection=run.get('cg_projection', 'double'),
            fused_cg=run.get('fused_cg', False),
            spectral_fusion=run.get('spectral_fusion', False),
            inner_forcing=run.get('inner_forcing', 'legacy'))
        for option, default in [('formulation', 'displacement'), ('preconditioner', 'matrix')]:
            diagnostics[label][option] = run.get(option, default)
    report = dict(shape=reference['configuration']['shape'], points=len(rows[0]),
        stress_scale_GPa=scale, threshold=.001, max_stress_error_over_scale=error,
        max_stress_error_MPa=error*scale*1000,
        max_mean_strain_difference=float(np.max(np.abs(strains[1]-strains[0]))),
        max_equivalent_plastic_mean_difference=max(abs(a['equivalent_plastic_mean']-
            b['equivalent_plastic_mean']) for a, b in zip(*rows)), runs=diagnostics,
        passed=bool(error <= .001 and all(np.isfinite(s).all() for s in stresses) and
            all(np.isfinite(e).all() for e in strains) and
            all(v['max_residual'] <= reference['rtol'] and
                v['max_macro_residual'] <= reference['rtol'] and
                v['max_hill_mandel_error'] <= 5*reference['rtol'] and
                v['max_yield_violation'] <= max(reference['rtol'],
                    5e-6 if reference['dtype'] == 'float32' else 1e-12) and
                v['min_plastic_dissipation'] >= 0 for v in diagnostics.values())),
        note='Curve equivalence at the same increments; not spatial or load-increment convergence, or experimental calibration.')
    for key in ('wall_seconds', 'successful_attempt_seconds'):
        a, b = (diagnostics[k][key] for k in ('reference', 'candidate'))
        if a is not None and b is not None and a > 0 and b > 0:
            report[key+'_speedup'] = a/b
            report[key+'_reduction_percent'] = 100*(1-b/a)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='metadata.json of reference')
    parser.add_argument('--candidate', type=Path, required=True, help='metadata.json of candidate')
    parser.add_argument('--output', type=Path, required=True, help='New directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    runs = [json.loads(p.read_text()) for p in (args.reference, args.candidate)]
    report = compare_runs(*runs)
    report['sources'] = {label: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for label, p in [('reference', args.reference), ('candidate', args.candidate)]}
    args.output.mkdir(parents=True)
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    axis = runs[0]['configuration']['axis']
    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True, layout='constrained')
    for run, label, style in zip(runs, ('Referencia', 'Candidato optimizado'), ('-', 'o')):
        axes[0].plot([r['strain'][axis]*100 for r in run['history']],
                     [r['stress'][axis]*1000 for r in run['history']], style,
                     markersize=3, fillstyle='none', label=label)
    difference = np.array([r['stress'] for r in runs[1]['history']]) - np.array(
        [r['stress'] for r in runs[0]['history']])
    axes[1].plot([r['strain'][axis]*100 for r in runs[0]['history']],
                 np.max(np.abs(difference), axis=1)*1000, '.-')
    axes[0].set(ylabel='Esfuerzo medio axial [MPa]',
                title='G09 — misma malla y física; plasticidad demostrativa')
    axes[0].legend()
    axes[1].set(xlabel='Deformación axial [%]', ylabel='Máx. diferencia |Δσᵢⱼ| [MPa]')
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(args.output/'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Curve-equivalence audit failed')


if __name__ == '__main__':
    main()
