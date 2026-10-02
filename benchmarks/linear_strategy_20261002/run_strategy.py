"""Alternated six-load comparisons, preserving the original physical audit."""
import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
sys.path.insert(0, str(ROOT / 'descartados/auxiliary_geneo_20261002'))
from profile_g09_linear import run
from audit_linear_performance import inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True)
    p.add_argument('--route', choices=['fft', 'energy', 'auxiliary', 'combined', 'fine_geneo'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--geneo-threshold',type=float,default=.15)
    a = p.parse_args()
    os.environ['CUFHOM_GENEO_THRESHOLD']=str(a.geneo_threshold)
    if a.case in ('G08', 'G09'):
        with gzip.open(ROOT/'benchmarks/performance_audit_20261001/evidence.json.gz', 'rt') as f:
            evidence = json.load(f)
        paths = evidence[a.case+'/report.json']['input_sha256']
        geometry = next(Path(f).parent for f in paths if f.endswith('/phase.npy'))
        material = next(Path(f) for f in paths if f.endswith('/material.json'))
    elif a.case.startswith('F'):
        geometry = ROOT/f'benchmarks/results/performance_audit_20261001/fiber_geometries/sfrp_vf15_ar10/N{a.case[1:]}'
        material = ROOT/'benchmarks/performance_audit_20261001/material.json'
    else:
        import numpy as np
        n = int(a.case[1:])
        family = 'regions' if a.case.startswith('R') else 'compact'
        phase, ori, mat, _ = inputs(n, None, None, family)
        geometry = ROOT/'benchmarks/results/linear_strategy_20261002/inputs'/a.case
        geometry.mkdir(parents=True, exist_ok=True)
        np.save(geometry/'phase.npy', phase); np.save(geometry/'ori.npy', ori)
        material = geometry/'material.json'
        material.write_text(json.dumps(mat)+'\n')
    if a.route == 'fft':
        os.environ['CUFHOM_FFT_TUNING_CACHE'] = str(a.output.resolve()/'fft_cache.json')
        variants = dict(current=({'linear_fft_autotune': False}, nullcontext),
                        tuned=({'linear_fft_autotune': True}, nullcontext))
    elif a.route == 'energy':
        variants = dict(current=({}, nullcontext), energy=({'linear_energy_update': True}, nullcontext))
    elif a.route == 'auxiliary':
        from auxiliary_space import auxiliary_context, geneo_context
        variants = dict(current=({}, nullcontext), auxiliary=({}, auxiliary_context),
                        geneo=({}, geneo_context))
    elif a.route == 'fine_geneo':
        from auxiliary_space import fine_geneo_context
        variants = dict(current=({}, nullcontext), fine_geneo=({}, fine_geneo_context))
    else:
        os.environ['CUFHOM_FFT_TUNING_CACHE'] = str(a.output.resolve()/'fft_cache.json')
        variants = dict(current=({}, nullcontext), combined=(dict(linear_energy_update=True,linear_fft_autotune=True),nullcontext))
    baseline = dict(linear_energy_update=False,linear_fft_autotune=False)
    variants = {name:(dict(baseline,**settings), context) for name,(settings,context) in variants.items()}
    report=run(SimpleNamespace(geometry=geometry, material=material, output=a.output, repeats=a.repeats), variants)
    report['strategy_arguments']=dict(case=a.case,route=a.route,geneo_threshold=a.geneo_threshold,repeats=a.repeats)
    sources=[Path(__file__), ROOT/'descartados/auxiliary_geneo_20261002/auxiliary_space.py',
             ROOT/'descartados/auxiliary_geneo_20261002/metric_pcg.py']
    report['strategy_source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
