"""Replay selected thesis materials without changing Sobol/enrichment ordering.

Includes six host snapshot exports, the campaign's voxel ordering and nearest
snapshot gather, and warm-start upload. Excludes Ritz assembly/monitor solves.
No existing campaign artifacts are modified or used as solved answers.
"""
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from profile_g09_fft import (ROOT, G09_N240, MATERIAL, optimized_parameters,
    quiet_solver, sync, cp, solve_homogenization, compile_cfield_geometry)

sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'scripts/cmame_interpretable_pipeline'))
import rom_reduced_operator as reduced
from constitutive_neighbors import nearest_constitutive_snapshot

DEFAULT_SEQUENCE = MATERIAL.parents[2]/'selected_sobol_sequence.csv'
PARAMETERS = ('Em', 'nu_m', 'Ef_L', 'Ef_T', 'G_LT', 'nu_LT', 'nu_TT')


def phases(material):
    coefficients = reduced._material_coefficients(material)
    return (np.einsum('q,qij->ij', coefficients[:2], reduced._isotropic_bases()),
            np.einsum('q,qij->ij', coefficients[2:], reduced._fiber_local_bases_axis0()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence', type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument('--geometry', type=Path, default=G09_N240)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--modes', nargs='+', default=['cold', 'previous', 'nearest'],
                        choices=['cold', 'previous', 'nearest', 'legacy_nearest'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or args.count < 2 or args.repeats < 1
            or not np.isfinite(args.rtol) or args.rtol <= 0
            or len(set(args.modes)) != len(args.modes) or 'cold' not in args.modes):
        parser.error('New output, positive repeats/rtol, count>=2 and unique modes including cold required')
    with args.sequence.open() as stream:
        records = list(csv.DictReader(stream))[:args.count]
    if len(records) != args.count:
        parser.error('Requested count exceeds the stored selected sequence')
    materials = [{k: float(row[k]) for k in PARAMETERS} for row in records]
    phase = np.load(args.geometry/'phase.npy'); ori = np.load(args.geometry/'ori.npy')
    phase.flags.writeable = ori.flags.writeable = False
    compiled = compile_cfield_geometry(phase, ori)
    config = optimized_parameters(phase, ori, materials[0], compiled)
    config.update(solver_rtol=args.rtol, cache_projection=True, check_true_residual=True)
    order = reduced.phase_orientation_voxel_order(phase, ori)
    inverse = np.empty_like(order); inverse[order] = np.arange(order.size)
    phase_pairs = [phases(m) for m in materials]
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    sources = ['FFT/pipeline/linear_coordinates.py', 'FFT/pipeline/linear_krylov.py',
        'FFT/pipeline/fft_solver.py', 'FFT/ffthompy_core/ffthompy/ffthompy/applications.py',
        'scripts/cmame_interpretable_pipeline/constitutive_neighbors.py',
        'FFT/benchmarks/benchmark_material_sequence.py']
    report = dict(sequence=str(args.sequence), sequence_sha256=hashlib.sha256(args.sequence.read_bytes()).hexdigest(),
        grid_shape=list(phase.shape), candidate_ids=[int(r['candidate_id']) for r in records],
        materials=materials, rtol=args.rtol, repeats=args.repeats,
        device=device.decode() if isinstance(device, bytes) else device,
        geometry_sha256={name:hashlib.sha256(memoryview(a).cast('B')).hexdigest()
                         for name,a in [('phase',phase),('ori',ori)]},
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        scope='Replay selected materials, same order and six loads. Warm kernels/plans and '
              'compiled immutable geometry. Includes field export, order/gather, upload, '
              'true-residual audit and timing metadata IO. Excludes adaptive monitor and Ritz work. '
              'Every mode exports/stores identical snapshot payloads. No cached Ceff answers.',
        variants={m:[] for m in args.modes})
    args.output.mkdir(parents=True)
    baseline = {}
    # Warm up both paths without retaining solved fields for the measured run.
    for formulation in (['auto', 'strain'] if 'legacy_nearest' in args.modes else ['auto']):
        with quiet_solver():
            solve_homogenization(dict(config, linear_formulation=formulation))
    sync()
    for repeat in range(args.repeats):
        modes = args.modes[repeat % len(args.modes):]+args.modes[:repeat % len(args.modes)]
        if repeat == 0:
            modes = ['cold']+[m for m in modes if m != 'cold']
        for mode in modes:
            ordered_history = []
            next_fields = np.empty((6, 6, phase.size), np.float32)
            run = dict(repeat=repeat, materials=[])
            for index, material in enumerate(materials):
                sync(); started = time.perf_counter()
                source, distance, initial = None, None, None
                selection_started = time.perf_counter()
                if mode != 'cold' and index:
                    source = index-1
                    if mode in ('nearest', 'legacy_nearest'):
                        source, distance = nearest_constitutive_snapshot(phase_pairs[index], phase_pairs[:index])
                    if source != index-1:
                        np.take(ordered_history[source], inverse, axis=2,
                                out=next_fields, mode='clip')
                    initial = next_fields
                selection_s = time.perf_counter()-selection_started
                ordered = np.empty_like(next_fields)
                export_seconds = [0.]
                def consume(load_id, field):
                    start = time.perf_counter()
                    value = field.reshape(6, phase.size)
                    next_fields[load_id] = value
                    np.take(value, order, axis=1, out=ordered[load_id], mode='clip')
                    export_seconds[0] += time.perf_counter()-start
                timing = args.output/f'{repeat}_{mode}_{index:03d}.json'
                parameters = dict(config, **material, initial_solution_fields=initial,
                    linear_formulation='strain' if mode == 'legacy_nearest' else 'auto',
                    solution_field_consumer=consume, solution_field_dtype='float32',
                    solver_timing_path=str(timing))
                with quiet_solver():
                    ceff = solve_homogenization(parameters)
                sync(); wall = time.perf_counter()-started
                metadata = json.loads(timing.read_text())
                infos = metadata['ffthompy_application_timing']['primaldual']['primal']['load_solver_infos']
                if mode == 'cold' and index not in baseline:
                    baseline[index] = ceff.copy()
                difference = float(np.linalg.norm(ceff-baseline[index])/np.linalg.norm(baseline[index]))
                entry = dict(position=index, candidate_id=int(records[index]['candidate_id']),
                    source_position=source, spectral_distance=distance, wall_s=wall,
                    selection_gather_s=selection_s, snapshot_copy_reorder_s=export_seconds[0],
                    Ceff=ceff.tolist(), Ceff_relative_difference=difference,
                    linear_formulation=metadata['linear_formulation'],
                    cg=sum(i['cg_iterations'] for i in infos), loads=infos,
                    comparison_passed=difference <= 1e-6,
                    equilibrium_passed=len(infos)==6 and all(i.get('true_residual_converged',False)
                        and i['true_norm_res_rel'] <= args.rtol for i in infos))
                run['materials'].append(entry)
                ordered_history.append(ordered)
                print(repeat, mode, index, 'source', source, 's', round(wall,3),
                      'cg',entry['cg'],'dC',f'{difference:.2e}',flush=True)
                if not entry['equilibrium_passed'] or not entry['comparison_passed']:
                    report['failure'] = entry
                    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
                    raise RuntimeError('Sequence failed physical/Ceff audit; see report.json')
            run['total_s'] = sum(r['wall_s'] for r in run['materials'])
            run['cg'] = sum(r['cg'] for r in run['materials'])
            report['variants'][mode].append(run)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            del ordered_history, ordered, parameters, initial, next_fields
            gc.collect()
    report['summary'] = {mode: dict(
        median_sequence_s=float(np.median([r['total_s'] for r in runs])),
        cg_per_sequence=[r['cg'] for r in runs],
        maximum_Ceff_relative_difference=max(m['Ceff_relative_difference'] for r in runs for m in r['materials']),
        equilibrium_passed=all(m['equilibrium_passed'] for r in runs for m in r['materials']))
        for mode,runs in report['variants'].items()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['summary'], indent=2),flush=True)


if __name__ == '__main__':
    main()
