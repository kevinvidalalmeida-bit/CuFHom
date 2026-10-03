"""Campaign adapter for the canonical periodic geometry generator."""
from pathlib import Path
import numpy as np
from .geometry import GeometryRequest, generate_geometry
from .sam_shapes import fiber_volume


def generate_rve_main(parameters):
    p = parameters
    box, length, diameter = float(p['caja_um']), float(p['L_um']), float(p['d_um'])
    grid = int(round(box*float(p['resol'])))
    output = Path(p['output_dir'])
    request = GeometryRequest(
        output=output/'geometry', box=box, ar=length/diameter, diameter=diameter,
        grid=grid, vf=float(p['Vf_target']), a2=(float(p['a11']), float(p['a22'])),
        shape=p.get('shape', 'cylinder'), solver=p.get('solver', 'a4'),
        threads=int(p.get('sam_num_threads', 1)), seed=int(p['seed']),
        batch_size=max(1, round(float(p.get('sam_batch_vf', .01))*box**3 /
                                fiber_volume(length, diameter, p.get('shape', 'cylinder')))),
        a2_tolerance=float(p.get('sam_A2_tolerance', .01)),
        voxel_a2_tolerance=float(p.get('sam_voxel_A2_tolerance', .01)),
        vf_tolerance=float(p.get('sam_vf_tolerance', .005)),
        overlap_tolerance=float(p.get('sam_overlap_tolerance_relative', 1e-6)),
        verbose=bool(p.get('sam_verbose', False)))
    summary = generate_geometry(request)
    audit = summary['geometric_certificate'] or {}
    contacts = audit.get('audit', {})
    vf = summary.get('vf', float('nan'))
    if summary['accepted']:
        with np.load(request.output/'geometry.npz', allow_pickle=False) as geometry:
            np.save(output/'phase.npy', geometry['phase'])
            np.save(output/'ori.npy', geometry['ori'])
    error = vf-request.vf
    return dict(
        accepted=summary['accepted'], geometry_status=summary['status'],
        sam_t_gen_reported_s=summary['total_wall_s'], sam_n_fibers=summary['accepted_count'],
        sam_vf=vf, sam_vf_target=request.vf, sam_vf_error=error,
        sam_vf_error_percent_points=100*error, sam_vf_deficit=max(0., -error),
        sam_vf_tolerance=request.vf_tolerance,
        sam_A2_tolerance=request.a2_tolerance, sam_voxel_A2_tolerance=request.voxel_a2_tolerance,
        sam_overlap_tolerance_relative=request.overlap_tolerance, sam_vf_ok=summary.get('vf_ok', False),
        sam_overlap_ok=bool(contacts.get('accepted', False) and audit.get('gap_audit', contacts).get('accepted', False)),
        sam_final_overlap_relative=contacts.get('maximum_relative_overlap', float('nan')),
        sam_A2_error_rel=audit.get('A2_error', float('nan')),
        sam_voxel_A2_error_rel=summary.get('voxel_A2_error', float('nan')),
        sam_A2_ok=bool(summary.get('voxel_A2_ok', False) and audit.get('A2_error', float('inf')) <= request.a2_tolerance),
        sam_A4_error_rel=audit.get('A4_error', float('nan')),
        sam_voxel_A4_error_rel=summary.get('voxel_A4_error', float('nan')))
