"""Bounded kernel benchmark, not a packing campaign or convergence certificate."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from cufhom.pipeline.sam_paper_reference import (ContactNeighbors, MultilevelContactNeighbors,
                                         objective, sphere_retract)
from cufhom.runtime import save_json


def benchmark():
    rows = []
    for n, length, box in ((64, 4., 12.), (256, 16., 40.), (512, 33., 70.)):
        rng = np.random.default_rng(293+n)
        c = rng.uniform(0, box, (n, 3))
        u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        states = []
        for _ in range(20):
            c = np.ascontiguousarray((c+.025*rng.normal(size=c.shape)) % box)
            u = sphere_retract(u, .002*rng.normal(size=u.shape))
            states.append((c.copy(), u.copy()))
        # Include rejected long trial followed by a return to prior geometry.
        states += [(np.ascontiguousarray(c+2.), sphere_retract(u, .3*np.roll(u,1,axis=1))), states[-1]]
        domain = np.full(3, box)
        reference = [objective(cc, uu, length, domain) for cc, uu in states]
        for cls in (ContactNeighbors, MultilevelContactNeighbors):
            elapsed = []
            for repeat in range(4):
                cache = cls(length, domain)
                start = time.perf_counter()
                actual = [objective(cc, uu, length, domain, neighbors=cache) for cc, uu in states]
                seconds = time.perf_counter()-start
                for expected, observed in zip(reference, actual):
                    for a, b in zip(expected, observed):
                        np.testing.assert_array_equal(a, b)
                if repeat:  # exclude JIT and first pass, same trajectory
                    elapsed.append(seconds)
            rows.append(dict(fibers=n, AR=length, box=box, evaluations=len(states),
                             mode=cls.__name__, median_s=float(np.median(elapsed)),
                             rebuilds=cache.rebuilds,
                             level_rebuilds=getattr(cache, 'level_rebuilds', []),
                             matches_dense_bitwise=True))
    return dict(scope='objective evaluations on fixed synthetic trajectories; not full generation',
                repeats=3, excludes_first_pass=True, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path to preserve earlier measurements')
    result = benchmark()
    save_json(args.output, result)
    for row in result['rows']:
        print(row, flush=True)
