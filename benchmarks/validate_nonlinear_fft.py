#!/usr/bin/env python3
"""Quantitative same-grid CPU/GPU and load-increment validation; no remeshing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from cufhom.pipeline.nonlinear_solver import solve_nonlinear_homogenization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'benchmarks/results/nonlinear_validation')
    parser.add_argument('--spectral-fusion', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--inner-forcing', choices=['legacy', 'ew', 'ew_target'])
    parser.add_argument('--coordinate-pcg', choices=['auto', 'classic', 'fused'], default='auto')
    parser.add_argument('--cupy-rfft-batch-mode', choices=['auto', 'full', 'component'], default='auto')
    parser.add_argument('--reuse-trial-response', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--formulation', choices=['strain', 'displacement'], default='displacement')
    parser.add_argument('--preconditioner', choices=['none', 'matrix'], default='matrix')
    args = parser.parse_args()
    if (args.output/'validation.json').exists():
        parser.error('Choose a new output directory')
    n = 12
    x, y, z = np.meshgrid(*[np.arange(n)-5.5]*3, indexing='ij')
    phase = ((x/3.7)**2 + (y/4.3)**2 + (z/3.2)**2 <= 1).astype(np.uint8)
    ori = np.zeros(phase.shape+(3,)); ori[:] = np.array([1., 1., .5])/1.5
    params = dict(phase_array=phase, ori_array=ori, return_fields=True,
                  coordinate_pcg=args.coordinate_pcg, reuse_trial_response=args.reuse_trial_response,
                  cupy_rfft_batch_mode=args.cupy_rfft_batch_mode,
                  formulation=args.formulation, preconditioner=args.preconditioner,
                  path=[0., .01, 0.], initial_increment=.00025)
    if args.spectral_fusion is not None:
        params['spectral_fusion'] = args.spectral_fusion
    if args.inner_forcing is not None:
        params['inner_forcing'] = args.inner_forcing
    runs = {}
    for name, extra in [('cpu64', {}), ('gpu32', dict(backend='cupy')),
                         ('half_increment', dict(initial_increment=.000125))]:
        print('Verifying '+name, flush=True)
        runs[name] = solve_nonlinear_homogenization(dict(params, **extra,
                                  output_dir=args.output/name))
    base = runs['cpu64']
    s0 = np.asarray([r['stress'] for r in base['history']])
    s1 = np.asarray([r['stress'] for r in runs['gpu32']['history']])
    s2 = np.asarray([r['stress'] for r in runs['half_increment']['history']])[::2]
    if s0.shape != s1.shape or s0.shape != s2.shape:
        raise RuntimeError('Unexpected cutbacks: compare branches at matching strains before acceptance')
    scale = base['stress_scale_GPa']
    report = dict(shape=list(phase.shape), fiber_fraction=float(phase.mean()),
                  stress_scale_GPa=scale, threshold=.001,
                  cpu_gpu_curve_error=float(np.max(np.abs(s1-s0))/scale),
                  half_increment_curve_error=float(np.max(np.abs(s2-s0))/scale),
                  gpu_final_strain_relative_L2=float(np.linalg.norm(runs['gpu32']['fields']['strain']-
                    base['fields']['strain'])/max(np.linalg.norm(base['fields']['strain']), 1e-12)),
                  residual_max={name: max(r['residual'] for r in run['history']) for name, run in runs.items()},
                  hill_mandel_max={name: max(r['hill_mandel_error'] for r in run['history']) for name, run in runs.items()},
                  wall_seconds={name: run['wall_seconds'] for name, run in runs.items()},
                  note='Synthetic inclusion, same grid. Not a temporal-convergence certificate for G09.')
    report['passed'] = (report['cpu_gpu_curve_error'] <= .001 and
                        report['half_increment_curve_error'] <= .001 and
                        all(report['residual_max'][k] <= v['rtol'] for k, v in runs.items()))
    (args.output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise SystemExit('Validation failed')


if __name__ == '__main__':
    main()
