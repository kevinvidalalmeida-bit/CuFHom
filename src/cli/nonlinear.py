"""Run nonlinear constitutive curves for a homogeneous, layered or supplied geometry."""
from pathlib import Path
import argparse
import json
import numpy as np

from cufhom.fft import solve_nonlinear_homogenization
from cufhom.fft.nonlinear.curves import (MANDEL_COMPONENTS, CONSTITUTIVE_COMPONENTS,
                                       loading_component, export_constitutive_curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['homogeneous', 'laminate', 'geometry'], default='homogeneous')
    parser.add_argument('--geometry', type=Path, help='geometry.npz or directory containing phase.npy and ori.npy')
    parser.add_argument('--material', type=Path, help='Material JSON for the user-supplied geometry')
    parser.add_argument('--component', choices=[*CONSTITUTIVE_COMPONENTS, 'all'], default='11',
                        help='Normal or shear experiment; all runs six independent virgin states')
    parser.add_argument('--backend', choices=['scipy', 'cupy'], default='scipy')
    parser.add_argument('--dtype', choices=['float32', 'float64'])
    parser.add_argument('--increment', type=float, default=.00025,
                        help='Increment of epsilon_ii or engineering gamma_ij (dimensionless)')
    parser.add_argument('--maximum', type=float, default=.01,
                        help='Final epsilon_ii or engineering gamma_ij; demo .01, not failure strain')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--globalization', choices=['auto', 'newton', 'ptc'], default='auto')
    parser.add_argument('--formulation', choices=['strain', 'displacement'], default='displacement',
                        help='displacement is the verified fast route; strain is the reference formulation')
    parser.add_argument('--preconditioner', choices=['none', 'matrix'], default='matrix',
                        help='matrix is the verified displacement route')
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--restart', type=Path)
    parser.add_argument('--start', type=float, default=0.,
                        help='Initial epsilon_ii or engineering gamma_ij, matching the checkpoint')
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument('--unload', action='store_true', help='Append unloading to zero strain (optional verification)')
    args = parser.parse_args()
    if args.component == 'all' and (args.restart or args.start != 0.):
        parser.error('--component all requires independent virgin states; restart a single component instead')
    components = CONSTITUTIVE_COMPONENTS if args.component == 'all' else (args.component,)
    output = args.output or Path.cwd()/'runs'/(
        'nonlinear_'+args.case+'_'+args.backend+('_cycle' if args.unload else '_loading')+
        ('' if args.component == '11' else '_'+args.component))
    run_dirs = {c: output/c if args.component == 'all' else output for c in components}
    if (output/'metadata.json').exists() or any((d/'metadata.json').exists() for d in run_dirs.values()):
        parser.error('Output already contains a run; choose another --output directory')
    if args.case == 'geometry':
        if args.geometry is None or args.material is None:
            parser.error('case geometry requires explicit --geometry and --material; no external campaign is assumed')
        if args.geometry.is_file():
            with np.load(args.geometry, allow_pickle=False) as data:
                phase, ori = data['phase'], data['ori']
        else:
            phase = np.load(args.geometry/'phase.npy', mmap_mode='r')
            ori = np.load(args.geometry/'ori.npy', mmap_mode='r')
        mat = json.loads(args.material.read_text())
        matrix, fiber = mat['matrix'], mat['fiber']
    else:
        phase = np.zeros((8, 8, 8), dtype=np.uint8)
        if args.case == 'laminate':
            phase[3:5, :, :] = 1
        ori = np.zeros(phase.shape+(3,)); ori[..., 0] = 1
        matrix = dict(E=3., nu=.3, yield_stress=.015, hardening=.03)
        fiber = dict(E_L=70., E_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.)
    parameters = dict(phase_array=phase, ori_array=ori, matrix=matrix, fiber=fiber,
                      backend=args.backend,
                      profile=args.profile, globalization=args.globalization,
                      formulation=args.formulation, preconditioner=args.preconditioner)
    if args.dtype:
        parameters['dtype'] = args.dtype
    if args.restart:
        parameters['restart_from'] = args.restart
    results = {}
    for component in components:
        axis = MANDEL_COMPONENTS.index(component)
        loading = loading_component(axis)
        scale = loading['strain_from_mandel_scale']
        count = [0]

        def progress(row):
            count[0] += 1
            print(json.dumps(dict(component=component, step=count[0],
                                  strain_quantity=loading['strain_quantity'],
                                  strain=row['strain'][axis]*scale,
                                  stress_quantity=loading['stress_quantity'],
                                  stress_GPa=row['stress'][axis]*loading['stress_from_mandel_scale'],
                                  newton=row['newton_iterations'], cg=row['cg_iterations'],
                                  seconds=row['seconds'], residual=row['residual'],
                                  plastic_fraction=row['plastic_fraction'])), flush=True)

        def rejected(row):
            physical = dict(row)
            for key in ('from_strain', 'target'):
                physical[key] *= scale
            print(json.dumps(dict(component=component, rejected=physical)), flush=True)

        run = dict(parameters, axis=axis,
                   path=[v/scale for v in [args.start, args.maximum]+([0.] if args.unload else [])],
                   initial_increment=args.increment/scale, pseudo_time_strain_scale=.01/scale,
                   output_dir=run_dirs[component], on_step=progress, on_reject=rejected)
        if args.checkpoint:
            run.update(checkpoint_path=run_dirs[component]/'checkpoint.npz', checkpoint_every=10)
        result = solve_nonlinear_homogenization(run)
        results[component] = result
        print(json.dumps(dict(component=component, output=str(run_dirs[component]),
                              wall_seconds=result['wall_seconds'], steps=len(result['history'])-1,
                              rejected=len(result['rejected_steps']),
                              max_residual=max(r['residual'] for r in result['history']))), flush=True)
    if args.component == 'all':
        export_constitutive_curves(results, output)
        print(json.dumps(dict(output=str(output), components=list(components),
                              wall_seconds=sum(r['wall_seconds'] for r in results.values()))), flush=True)


if __name__ == '__main__':
    main()
