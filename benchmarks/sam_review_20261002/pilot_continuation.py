"""Time-bounded straight A4 packing probe; retain only independently certified states."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numba
import numpy as np

from cufhom.pipeline.sam_staged_insertion import Settings, certificate, generate_staged


class TimeBudget(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=60.)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    numba.set_num_threads(1)
    with np.load(a.initial) as data:
        last = data['state'].copy()
    target = np.eye(3)/3
    length, diameter, box = 30., 1., 60.
    volume = np.pi*diameter**2*length/4 + np.pi*diameter**3/6
    s = Settings(box=box, length=length, diameter=diameter, target_count=round(.6*box**3/volume),
                 initial_count=48, batch_size=224, candidates=8, attempts=2, max_stages=100,
                 relax_iterations=5000, orientation_weight=1., overlap_tolerance=1e-6,
                 a2_tolerance=.035, seed=20261002, solver='paper-a4', a4_tolerance=1e-4)
    request = dict(settings=asdict(s), initial=str(a.initial.resolve()), seconds=a.seconds,
                   nominal_vf_target=.6, model='capsules', purpose=__doc__)
    (a.output/'request.json').write_text(json.dumps(request, indent=2)+'\n')
    check = certificate(last, box, target, s.overlap_tolerance, s.a2_tolerance, s.a4_tolerance)
    if not check['accepted']:
        raise ValueError('Initial state failed independent audit')
    start = time.perf_counter()
    def checkpoint(table, event):
        nonlocal last
        last = table.copy()
        np.savez_compressed(a.output/'last_geometrically_valid.npz', state=last)
    checkpoint(last, {})
    with (a.output/'events.jsonl').open('w') as stream:
        def progress(event):
            event = dict(event, elapsed_s=time.perf_counter()-start)
            stream.write(json.dumps(event)+'\n')
            stream.flush()
            if event['event'] != 'relax_progress':
                print(json.dumps(event), flush=True)
            if event['elapsed_s'] >= a.seconds:
                raise TimeBudget
        try:
            last, summary = generate_staged(s, target, initial=last,
                                             progress=progress, checkpoint=checkpoint)
        except TimeBudget:
            summary = dict(status='wall_budget_exhausted', accepted_count=len(last), target_count=s.target_count)
    summary.update(geometric_certificate=certificate(last, box, target, s.overlap_tolerance,
                                                     s.a2_tolerance, s.a4_tolerance),
                   nominal_capsule_vf=len(last)*volume/box**3,
                   nominal_cylinder_vf=len(last)*np.pi*diameter**2*length/4/box**3,
                   accepted=False, vf_not_certified=True, voxel_checks_performed=False,
                   total_wall_s=time.perf_counter()-start)
    (a.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('events', 'orientation_target')}), flush=True)


if __name__ == '__main__':
    main()
