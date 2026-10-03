"""Separate box capacity, orientation moments and isotropic packing feasibility.

The capacity witness has parallel fibers. It MUST NOT be reported as a solution
of the user's isotropic packing problem. Units are physical diameters.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from cufhom.pipeline.sam_paper_reference import (
    ContactNeighbors, a4_error, rotational_inertia,
)
from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers, segment_distance


ROOT = Path(__file__).resolve().parent
BOX, LENGTH, DIAMETER, GAP, TARGET = 60., 30., 1., .001, .60


def capacity_witness():
    """Triangular lattice of parallel diagonal lines on the cubic torus.

    Projecting B*Z^3 onto (1,1,1)^perp gives a triangular lattice, generated
    by v1,v2 below. Its shortest vector is B*sqrt(2/3). Subdivision by q
    therefore gives q^2 distinct periodic lines separated by at least p.
    Each line has period P=B*sqrt(3); m centers spaced by P/m accommodate
    segments whose capsule end-to-end length is LENGTH + DIAMETER + GAP.
    Distinct lines and neighbors on the same line consequently cannot overlap.
    """
    q = int(np.floor(BOX*np.sqrt(2/3)/(DIAMETER+GAP)))
    m = int(np.floor(BOX*np.sqrt(3)/(LENGTH+DIAMETER+GAP)))
    u = np.ones(3)/np.sqrt(3)
    v1 = BOX*(np.array([1., 0., 0.])-np.ones(3)/3)
    v2 = BOX*(np.array([0., 1., 0.])-np.ones(3)/3)
    i, j, k = np.indices((q, q, m)).reshape(3, -1)
    centers = (i[:, None]*v1/q+j[:, None]*v2/q
               +k[:, None]*BOX/m*np.ones(3)) % BOX
    single_volume = np.pi*DIAMETER**2*LENGTH/4
    count = int(np.ceil(TARGET*BOX**3/single_volume))
    chosen = np.random.default_rng(20261002).permutation(len(centers))[:count]
    fibers = np.column_stack((centers[chosen], np.tile(u, (count, 1)),
                              np.full(count, LENGTH), np.full(count, DIAMETER)))
    inflated = fibers.copy()
    inflated[:, 7] += GAP
    started = time.perf_counter()
    audit = audit_periodic_fibers(inflated, BOX, 0.)
    elapsed = time.perf_counter()-started
    assert audit['accepted'], audit
    assert len(centers) >= count
    result = dict(
        purpose='capacity only: parallel fibers, NOT isotropic packing',
        target_isotropic_problem_solved=False, count=count,
        cylinder_volume_fraction=count*single_volume/BOX**3,
        full_lattice_count=len(centers),
        full_lattice_cylinder_volume_fraction=len(centers)*single_volume/BOX**3,
        lattice_subdivisions=q, axial_centers_per_period=m,
        transverse_line_spacing=BOX*np.sqrt(2/3)/q,
        axial_spacing=BOX*np.sqrt(3)/m,
        required_transverse_spacing=DIAMETER+GAP,
        required_axial_spacing=LENGTH+DIAMETER+GAP,
        a2_relative_error=float(np.linalg.norm(u[:, None]*u-np.eye(3)/3)
                                /np.linalg.norm(np.eye(3)/3)),
        a4_relative_error=a4_error(fibers[:, 3:6]),
        gap_audit=audit, audit_seconds=elapsed,
    )
    np.savez_compressed(ROOT/'capacity_only_parallel.npz', fibers=fibers,
                        box=np.full(3, BOX), purpose=result['purpose'],
                        isotropic_packing_accepted=False)
    return result


def orientation_moments_only(count):
    """Four equal rings integrate polynomials through degree four exactly.

    Positions and collisions are deliberately absent: this checks whether the
    finite target count itself prevents A2/A4 isotropy, not simultaneous packing.
    """
    assert count % 4 == 0
    a, b = 1/3-np.sqrt(4/45), 1/3+np.sqrt(4/45)
    z = np.repeat([-np.sqrt(b), -np.sqrt(a), np.sqrt(a), np.sqrt(b)], count//4)
    phi = np.tile(2*np.pi*np.arange(count//4)/(count//4), 4)
    u = np.column_stack((np.sqrt(1-z*z)*np.cos(phi),
                         np.sqrt(1-z*z)*np.sin(phi), z))
    a2 = u.T@u/count
    return dict(purpose='orientation moments only: NO spatial packing', count=count,
                a2_relative_error=float(np.linalg.norm(a2-np.eye(3)/3)
                                        /np.linalg.norm(np.eye(3)/3)),
                a4_relative_error=a4_error(u))


def local_alignment(checkpoint):
    """Measure local director correlations at two explicit axis cutoffs."""
    with np.load(checkpoint, allow_pickle=False) as data:
        f = data['state']
        settings = json.loads(str(data['settings_json']))
        assert settings['box'] == BOX and settings['length'] == LENGTH
        assert settings['diameter'] == DIAMETER and settings['min_gap'] == GAP
        assert str(data['shape']) == 'cylinder'
    assert np.all(f[:, 6] == LENGTH) and np.all(f[:, 7] == DIAMETER)
    neighbors = ContactNeighbors(LENGTH, np.full(3, BOX), skin=.5,
                                 contact_diameter=DIAMETER+GAP)
    offsets, indices = neighbors.get(f[:, :3], f[:, 3:6])
    i = np.repeat(np.arange(len(f)), np.diff(offsets))
    j = indices
    keep = i < j
    i, j = i[keep], j[keep]
    delta = f[j, :3]-f[i, :3]
    delta -= np.rint(delta/BOX)*BOX
    u, v = f[i, 3:6]*LENGTH, f[j, 3:6]*LENGTH
    distance = np.full(len(i), np.inf)
    extent = .5*(np.abs(u)+np.abs(v))+1.501
    for shift in np.ndindex(3, 3, 3):
        other = delta+BOX*(np.array(shift)-1)
        possible = np.all(np.abs(other) <= extent+1e-12, axis=1)
        if possible.any():
            distance[possible] = np.minimum(distance[possible], segment_distance(
                other[possible], u[possible], v[possible]))
    cosine = np.abs(np.sum(f[i, 3:6]*f[j, 3:6], axis=1))
    p2 = .5*(3*cosine**2-1)
    samples = []
    for cutoff in (1.01, 1.1, 1.5):
        near = distance <= cutoff
        samples.append(dict(axis_distance_cutoff=cutoff, pair_count=int(near.sum()),
                            mean_p2=float(p2[near].mean()),
                            fraction_angle_below_20_degrees=float(
                                np.mean(cosine[near] >= np.cos(np.deg2rad(20))))))
    return dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(
        checkpoint.read_bytes()).hexdigest(), count=len(f),
        cylinder_volume_fraction=len(f)*np.pi*DIAMETER**2*LENGTH/(4*BOX**3),
        global_a4_relative_error=a4_error(f[:, 3:6]), samples=samples,
        reference_uniform_independent_directions=dict(mean_p2=0.,
            fraction_angle_below_20_degrees=float(1-np.cos(np.deg2rad(20)))),
        note='Pairs selected by proximity are not independent uniform samples; '
             'these statistics diagnose correlations, not a packing upper bound.')


def main():
    volume = np.pi*DIAMETER**2*LENGTH/4
    count = int(np.rint(TARGET*BOX**3/volume))
    eps = rotational_inertia(LENGTH, DIAMETER)
    old_eps = DIAMETER**2*(2/3+(LENGTH/DIAMETER)*(
        (LENGTH/DIAMETER)**3/12+(LENGTH/DIAMETER)**2/6
        +3*(LENGTH/DIAMETER)/16+1/15))
    result = dict(
        parameters=dict(box=BOX, length=LENGTH, diameter=DIAMETER, ar=LENGTH/DIAMETER,
                        min_gap=GAP, periodic=True, physical_shape='cylinder'),
        analytical_checks=dict(
            cube_volume=BOX**3, single_cylinder_volume=volume,
            nominal_target_count=count,
            nominal_target_volume_fraction=count*volume/BOX**3,
            cylinder_bounding_sphere_diameter=float(np.hypot(LENGTH, DIAMETER)),
            all_direction_self_image_axis_distance_lower_bound=BOX-LENGTH,
            capsule_with_gap_diameter=DIAMETER+GAP,
            inflated_capsules_occupied_volume_fraction=count*(
                np.pi*(DIAMETER+GAP)**2*LENGTH/4
                +np.pi*(DIAMETER+GAP)**3/6)/BOX**3,
            random_rod_asymptotic_scaling_range=[4/(LENGTH/DIAMETER),
                                               5.6/(LENGTH/DIAMETER)],
            random_rod_scaling_is_rigorous_upper_bound=False,
        ),
        published_comparison=dict(ar=33., box_over_diameter=70., length_over_box=33/70,
                                  min_gap_over_diameter=.001, count=7940,
                                  cylinder_volume_fraction=7940*np.pi*33/(4*70**3)),
        legacy_inertia_discrepancy=dict(current_paper_formula=eps,
                                       pre_fix_legacy_formula=old_eps, ratio=old_eps/eps),
        capacity_witness=capacity_witness(),
        orientation_moments_only=orientation_moments_only(count),
        local_alignment=local_alignment(ROOT.parent/'results'/
            'sam_dense_cylinder_ar30_candidates128_20261002'/'last_geometrically_valid.npz'),
        conclusion='Box capacity above 60% proved for a parallel arrangement. '
                   'A2/A4 realizability at target count proved separately. '
                   'Their simultaneous realization at Vf=60% in this exact box '
                   'has NOT been proved by these checks.',
    )
    repo = ROOT.parent.parent
    result['source_sha256'] = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), repo/'src/cufhom/pipeline/sam_generator.py',
                     repo/'src/cufhom/pipeline/sam_periodic_audit.py',
                     repo/'src/cufhom/pipeline/sam_paper_reference.py',
                     repo/'tests/test_sam_inertia.py')
    }
    result['capacity_witness']['npz_sha256'] = hashlib.sha256(
        (ROOT/'capacity_only_parallel.npz').read_bytes()).hexdigest()
    (ROOT/'feasibility.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
