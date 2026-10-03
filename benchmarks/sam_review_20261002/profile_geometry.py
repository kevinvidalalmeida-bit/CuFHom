"""Bounded SAM review: synthetic kernels and existing masters, no dense packing claim."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numba
import numpy as np

from cufhom.pipeline.sam_contact_energy import contact_energy
from cufhom.pipeline.sam_generator import SAMLiteGenerator
from cufhom.pipeline.sam_paper_reference import (
    ContactNeighbors, MultilevelContactNeighbors, a4_error, objective,
    rotational_inertia,
)
from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers

ROOT = Path(__file__).resolve().parents[2]


def timed(fn, repeats=3):
    fn()  # JIT / first call excluded
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - start)
    return value, dict(median_s=float(np.median(samples)), samples_s=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/results/performance_audit_20261001/fiber_geometries/manifest.json')
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a new output file to preserve evidence')
    numba.set_num_threads(1)
    result = dict(scope=__doc__, machine=platform.platform(), numba_threads=1,
                  kernels=[], masters=[], budgets=[])
    for n in (512, 2048, 5500):
        rng = np.random.default_rng(20261002 + n)
        box, length = np.full(3, 60.), 30.
        c = rng.uniform(0, 60., (n, 3))
        u = rng.normal(size=(n, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        dense, stats = timed(lambda: objective(c, u, length, box))
        row = dict(fibers=n, AR=length, box=60., synthetic_overlapping=True,
                   dense_objective=stats, overlap=dense[3], neighbors={})
        for cls in (ContactNeighbors, MultilevelContactNeighbors):
            cache = cls(length, box)
            def rebuild():
                fresh = cls(length, box)
                return fresh.get(c, u)
            _, build_stats = timed(rebuild)
            sparse, cached_stats = timed(lambda: objective(c, u, length, box, neighbors=cache))
            for a, b in zip(dense, sparse):
                np.testing.assert_array_equal(a, b)
            row['neighbors'][cls.__name__] = dict(
                build=build_stats, cached_objective=cached_stats,
                entries=int(len(cache.indices)), includes_self=True,
                matches_dense_bitwise=True)
        if n <= 2048:
            f = np.c_[c, u, np.full(n, length), np.ones(n)]
            audit, stats = timed(lambda: audit_periodic_fibers(f, box, 1e-6))
            np.testing.assert_allclose(audit['maximum_relative_overlap'], dense[3], atol=1e-12)
            row['independent_audit'] = stats
        result['kernels'].append(row)
        print(json.dumps(row), flush=True)

    manifest = json.loads(args.manifest.read_text())
    for case in manifest['cases']:
        path = ROOT / case['table']
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != case['master_sha256']:
            raise ValueError(f'Master changed: {path}')
        data = np.genfromtxt(path, delimiter=',', names=True)
        u = np.column_stack([data[name] for name in ('ux', 'uy', 'uz')])
        result['masters'].append(dict(case=case['case'], fibers=len(u),
            A2_relative=float(np.linalg.norm(u.T @ u / len(u) - np.eye(3)/3) / np.linalg.norm(np.eye(3)/3)),
            A4_relative=a4_error(u), source=str(path.relative_to(ROOT)), sha256=sha))

    for ar, box in ((30., 60.), (30., 64.), (33., 70.)):
        cylinder = np.pi * ar / 4
        capsule = cylinder + np.pi / 6
        nc, ns = [int(math.floor(.6 * box**3 / v + .5)) for v in (cylinder, capsule)]
        result['budgets'].append(dict(AR=ar, box=box, diameter=1., target_vf=.6,
            count_cylinders=nc, count_capsules=ns,
            default_count_ceiling_with_bootstrap_growth=48 + 39*4,
            capsule_vf_at_default_ceiling=(48 + 39*4)*capsule/box**3,
            minimum_stages_capsules_initial12_batch4=1+math.ceil((ns-12)/4),
            legacy_rotation_scale=SAMLiteGenerator.rotational_scale(ar, 1.),
            paper_rotation_scale=rotational_inertia(ar, 1.),
            scale_ratio=SAMLiteGenerator.rotational_scale(ar, 1.)/rotational_inertia(ar, 1.),
            audit_pair_count=nc*(nc+1)//2,
            audit_core_arrays_GiB=120*nc*(nc+1)//2/1024**3,
            voxels_per_diameter_at_N256=256/box,
            N_for_6_voxels_per_diameter=int(6*box)))
    files = [Path(__file__), *sorted((ROOT/'src/cufhom/pipeline').glob('sam_*.py')),
             ROOT/'src/cufhom/cli/geometry.py', ROOT/'src/cufhom/pipeline/rve_generator.py']
    result['source_sha256'] = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    result['contact_numba_options'] = contact_energy.targetoptions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
