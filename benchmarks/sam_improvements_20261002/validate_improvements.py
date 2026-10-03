"""Compare fixed inputs and certify outputs; elapsed times exclude JIT warmup."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import tracemalloc
from unittest.mock import patch

import numba
import numpy as np

from cufhom.pipeline import sam_paper_reference as paper
from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def timed(function, repeats=5):
    value = function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter()-start)
    return value, dict(median_s=float(np.median(samples)), samples_s=samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path to preserve evidence')
    numba.set_num_threads(1)
    name = 'cufhom.pipeline._sam_contact_reference'
    spec = importlib.util.spec_from_file_location(name, HERE/'contact_reference.py')
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    result = dict(numba_threads=1, kernels=[])
    for n in (512,2048,5500):
        rng = np.random.default_rng(20261002+n)
        c = rng.uniform(0,60,(n,3))
        u = rng.normal(size=(n,3)); u /= np.linalg.norm(u,axis=1)[:,None]
        box = np.full(3,60.)
        neighbors = paper.ContactNeighbors(30.,box)
        neighbors.get(c,u)
        def evaluate():
            return paper.objective(c,u,30.,box,neighbors=neighbors)
        actual,new = timed(evaluate)
        with patch.object(paper,'contact_energy',reference.contact_energy):
            expected,old = timed(evaluate)
        differences = []
        for index,(a,b) in enumerate(zip(actual,expected)):
            # Small segment-distance roundoff is amplified in force directions
            # near intersecting axes. Keep geometry/energy comparisons stricter.
            tolerance = 1e-8 if index in (1,2) else 2e-10
            np.testing.assert_allclose(a,b,rtol=1e-9,atol=tolerance)
            differences.append(float(np.max(np.abs(np.asarray(a)-b))))
        row = dict(fibers=n,synthetic_overlapping=True,cached_objective_new=new,
                   cached_objective_reference=old, speedup=old['median_s']/new['median_s'],
                   max_abs_differences=differences,neighbor_entries=len(neighbors.indices))
        result['kernels'].append(row)
        print(json.dumps(row),flush=True)
        if n == 5500:
            f = np.c_[c,u,np.full(n,30.),np.ones(n)]
            tracemalloc.start()
            start = time.perf_counter()
            audit = audit_periodic_fibers(f,box,1e-6)
            elapsed = time.perf_counter()-start
            _,peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            np.testing.assert_allclose(audit['maximum_relative_overlap'],actual[3],atol=1e-12)
            result['audit_5500'] = dict(wall_s=elapsed,traced_peak_bytes=peak,
                                         block_pairs=65536,result=audit)
    trial_path = HERE/'trial_ar30_vf15.npz'
    with np.load(trial_path,allow_pickle=False) as data:
        trial = data['state']
    # Warm both code paths with one iteration on the same trial.
    for optimizer in ('gradient','lbfgs'):
        paper.relax_paper_reference(trial,60.,np.eye(3)/3,maxiter=1,
                                    optimizer=optimizer,overlap_tolerance=1e-6)
    result['relaxation'] = dict(input_sha256=hashlib.sha256(trial_path.read_bytes()).hexdigest(),
                                fibers=len(trial),box=60.,AR=30.,min_gap=0.,solvers={})
    for optimizer in ('gradient','lbfgs'):
        start = time.perf_counter()
        final,info = paper.relax_paper_reference(trial,60.,np.eye(3)/3,maxiter=5000,
                                                optimizer=optimizer,overlap_tolerance=1e-6)
        info['wall_s'] = time.perf_counter()-start
        info['audit_repeated'] = audit_periodic_fibers(final,60.,1e-6)
        result['relaxation']['solvers'][optimizer] = info
        print(json.dumps(dict(optimizer=optimizer,iterations=info['iterations'],
                              evaluations=info['evaluations'],wall_s=info['wall_s'],
                              accepted=info['contact_A2_A4_accepted'],
                              overlap=info['after']['maximum_relative_overlap'])),flush=True)
    sources = [Path(__file__),HERE/'contact_reference.py',
               *sorted((ROOT/'src/cufhom/pipeline').glob('sam_*.py')),
               ROOT/'src/cufhom/cli/geometry.py']
    result['source_sha256'] = {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sources}
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
