"""Fixed continuous geometry vs resolution; optional bounded energy experiment.

Reads CSVs only; writes new small tables/reports, never full voxel fields.
No material solve, training or original-geometry modification.
"""
import argparse
import itertools
import math
from pathlib import Path
import sys
import time
from cufhom.runtime import progress, save_json


def flat_cylinder_interior(points, fiber, box):
    """Diagnostic only: strict membership in any periodic flat-cylinder image.

    All dimensions are physical. Does not modify SAM's capsule model. Testing
    finitely many sample points can prove intersection, never its absence.
    """
    import numpy as np
    c = fiber[:3] % box
    axis = fiber[3:6]
    half, radius = .5*fiber[6], .5*fiber[7]
    extent = half*np.abs(axis)+radius*np.sqrt(np.maximum(0., 1-axis**2))
    lo = np.ceil((-c-extent)/box).astype(int)
    hi = np.floor((box-c+extent)/box).astype(int)
    inside = np.zeros(len(points), dtype=bool)
    margin = 1e-10*max(1., box, fiber[6], fiber[7])
    for shift in itertools.product(*(range(a, b+1) for a, b in zip(lo, hi))):
        delta = points-c-box*np.array(shift)
        axial = delta@axis
        radial = delta-axial[:, None]*axis
        inside |= ((np.abs(axial) < half-margin)
                   & (np.einsum('ij,ij->i', radial, radial) < max(0., radius-margin)**2))
    return inside


def voxel_diagnostic(fibers, box, grid):
    import numpy as np
    from cufhom.pipeline.rve_generator import _fiber_voxel_indices
    scale = grid/box
    occupancy = np.zeros(grid**3, dtype=np.uint32)
    flat_occupancy = np.zeros(grid**3, dtype=np.uint32)
    owner = np.full(grid**3, -1, dtype=np.int32)
    for i, f in enumerate(fibers):
        ix, iy, iz = _fiber_voxel_indices(
            np.ascontiguousarray((f[:3] % box)*scale),
            np.ascontiguousarray(f[3:6]), f[6]*scale, .5*f[7]*scale, grid)
        flat = np.unique((ix*grid+iy)*grid+iz)
        occupancy[flat] += 1
        owner[flat] = i
        points = np.column_stack((flat//(grid*grid), (flat//grid) % grid, flat % grid))/scale
        interior = flat_cylinder_interior(points, f, box)
        flat_occupancy[flat[interior]] += 1
    used = occupancy > 0
    union = int(used.sum())
    owner = owner.reshape((grid,)*3)
    faces = 0
    for axis in range(3):
        other = np.roll(owner, 1, axis=axis)
        faces += int(((owner >= 0) & (other >= 0) & (owner != other)).sum())
    counts = np.bincount(owner[owner >= 0], minlength=len(fibers)).astype(float)
    a2 = np.einsum('n,ni,nj->ij', counts, fibers[:, 3:6], fibers[:, 3:6])/max(1, union)
    return dict(grid=grid, h_um=box/grid, vf=union/grid**3,
                multiply_occupied_sample_points=int((occupancy > 1).sum()),
                excess_memberships=int(occupancy.sum()-union),
                flat_cylinder_strict_shared_sample_points=int((flat_occupancy > 1).sum()),
                flat_cylinder_interior_sample_fraction=float((flat_occupancy > 0).sum()/grid**3),
                unlike_fiber_neighbor_faces=faces, A2_last_owner=a2.tolist(),
                note='Multiple capsule owners imply intersection (possibly boundary-only). '
                     'Strict flat-cylinder shared points prove interior intersection, '
                     'but zero points do not certify separation. Neighbor voxels do not prove overlap.')


def run(args):
    import numpy as np
    import numba
    from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers
    from cufhom.pipeline.sam_contact_energy import relax_contact_energy
    numba.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    target = np.diag([*args.a2, 1-sum(args.a2)])
    records = []
    for index, source in enumerate(args.input):
        fibers = np.atleast_2d(np.loadtxt(source, delimiter=',', skiprows=1))
        progress(f'Continuous audit {index+1}/{len(args.input)}: {source}')
        record = dict(source=str(source.resolve()), fiber_count=len(fibers),
                      nominal_cylinder_vf=float(np.sum(np.pi*fibers[:, 7]**2*fibers[:, 6]/4)/args.box**3),
                      capsule_volume_sum_fraction=float(np.sum(np.pi*fibers[:, 7]**2*fibers[:, 6]/4+
                                                               np.pi*fibers[:, 7]**3/6)/args.box**3),
                      initial_audit=audit_periodic_fibers(fibers, args.box, args.overlap_tolerance))
        # A sum of individual capsule volumes is an upper bound on the union,
        # not its measurement. Moving a fixed set cannot exceed this bound.
        upper = record['capsule_volume_sum_fraction']
        record['continuous_vf_feasible_by_volume_bound'] = bool(upper >= args.vf_target-args.vf_tolerance)
        if np.allclose(fibers[:, 6:], fibers[0, 6:]):
            record['minimum_count_for_vf_lower_tolerance'] = math.ceil(
                max(0., args.vf_target-args.vf_tolerance)/(upper/len(fibers)))
            record['nearest_count_for_nominal_capsule_target'] = int(
                np.floor(args.vf_target/(upper/len(fibers))+.5))
        variants = [('original', fibers)]
        if args.energy_iterations:
            progress('Experimental energy relaxation (geometry only)')
            start = time.perf_counter()
            out, info = relax_contact_energy(
                fibers, args.box, target, maxiter=args.energy_iterations,
                orientation_weight=args.orientation_weight,
                overlap_tolerance=args.overlap_tolerance, a2_tolerance=args.a2_tolerance,
                progress=lambda x: progress(f"iter={x['iteration']} overlap/d={x['overlap_relative']:.6g} "
                                           f"A2={x['A2_error']:.6g} energy={x['energy']:.6g}"))
            info['wall_s'] = time.perf_counter()-start
            record['energy'] = info
            np.savetxt(args.output/f'candidate_{index:02d}.csv', out, delimiter=',',
                       header='cx_um,cy_um,cz_um,ux,uy,uz,L_um,d_um', comments='')
            variants.append(('energy', out))
        record['rasterizations'] = {}
        for name, table in variants:
            rows = []
            for grid in args.grids:
                progress(f'Raster-only {name}, N={grid}; same fixed continuous fibers')
                row = voxel_diagnostic(table, args.box, grid)
                row['A2_error_relative'] = float(np.linalg.norm(np.array(row['A2_last_owner'])-target)/np.linalg.norm(target))
                row['vf_ok'] = abs(row['vf']-args.vf_target) <= args.vf_tolerance
                row['A2_ok'] = row['A2_error_relative'] <= args.voxel_a2_tolerance
                audit = record['initial_audit'] if name == 'original' else record['energy']['after']
                cont_a2 = np.linalg.norm(table[:, 3:6].T@table[:, 3:6]/len(table)-target)/np.linalg.norm(target)
                row['accepted_all'] = bool(audit['accepted'] and row['vf_ok'] and row['A2_ok']
                                           and cont_a2 <= args.a2_tolerance)
                rows.append(row)
            record['rasterizations'][name] = rows
        records.append(record)
        save_json(args.output/'diagnostic.json', dict(parameters=vars(args), cases=records))
    lines = ['# Geometría continua y voxelización: diagnóstico', '',
             'No FFT. Mismas fibras en todas las resoluciones. Candidatos experimentales, no certificados.', '',
             '| Caso | Ruta | N | Vf % | Puntos con ≥2 cápsulas | Interiores compartidos de cilindros planos | Contacto continuo válido | Aceptado todo |',
             '|---|---|---:|---:|---:|---:|---|---|']
    for i, record in enumerate(records):
        for name, rows in record['rasterizations'].items():
            audit = record['initial_audit'] if name == 'original' else record['energy']['after']
            for row in rows:
                lines.append(f"| {i} | {name} | {row['grid']} | {100*row['vf']:.5f} | "
                             f"{row['multiply_occupied_sample_points']} | {row['flat_cylinder_strict_shared_sample_points']} | "
                             f"{audit['accepted']} | {row['accepted_all']} |")
    lines += ['', 'Adyacencia entre vóxeles de fibras distintas no prueba penetración geométrica.',
              'Puntos con múltiples fibras sí pertenecen a la intersección continua; a baja resolución se pueden omitir.',
              'Volumen suma de cápsulas no es volumen unión si hay intersecciones.',
              'No se cambian el modelo de cápsulas, el número de fibras ni sus dimensiones.']
    for i, record in enumerate(records):
        lines += ['', f"Caso {i}: suma de volúmenes de cápsulas = {100*record['capsule_volume_sum_fraction']:.5f} %.",
                  f"Vf continua factible por esta cota superior: {record['continuous_vf_feasible_by_volume_bound']}."]
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    progress(f'Complete: {args.output}/REPORT.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--box', type=float, default=16.)
    p.add_argument('--grids', type=int, nargs='+', default=[48, 96, 144])
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--a2', type=float, nargs=2, default=[1/3, 1/3])
    p.add_argument('--energy-iterations', type=int, default=0)
    p.add_argument('--orientation-weight', type=float, default=1.)
    p.add_argument('--overlap-tolerance', type=float, default=1/120)
    p.add_argument('--a2-tolerance', type=float, default=.035)
    p.add_argument('--voxel-a2-tolerance', type=float, default=.06)
    p.add_argument('--vf-target', type=float, default=.4)
    p.add_argument('--vf-tolerance', type=float, default=.006)
    args = p.parse_args()
    if (args.box <= 0 or min(args.grids) < 2 or args.energy_iterations < 0
            or min(args.a2) < 0 or sum(args.a2) > 1 or not 0 < args.vf_target < 1):
        p.error('Invalid geometry/iteration settings')
    run(args)
