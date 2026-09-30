"""Measure fixed-grid specialization and launch geometry, without changing PCG."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark_gpu_hotspots import timed
from profile_g09_fft import cp
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from pipeline.nonlinear_reference import IsotropicReference
from pcg_geometry_candidate import installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('New output required')
    P = MixedProjector((240,)*3,(1.,1.,1.),[True,False,False,False,False,False],cp)
    Q = DisplacementCoordinates(P)
    M = IsotropicReference(P,2.5,1.15); M.set_shift(0.,np.dtype('float32'))
    baseline = FusedCoordinatePCG(Q)
    with installed():
        candidate = FusedCoordinatePCG(Q)
    rng = cp.random.RandomState(92)
    x,r,p,ap = [rng.standard_normal(Q.size*2,dtype=cp.float32).view(cp.complex64) for _ in range(4)]
    report = dict(scope='Synthetic 240^3 vectors; CUDA-event stage timings only.', stages={})
    for name,engine in [('baseline',baseline),('specialized',candidate)]:
        for blocks in ([16384] if name=='baseline' else [2048,4096,8192,16384]):
            engine.blocks=blocks; engine.partial=cp.empty((2,blocks),dtype=cp.float64)
            report['stages'][f'{name}_{blocks}_update'] = timed(lambda:engine.update(p,ap,0.,x,r,M))
            report['stages'][f'{name}_{blocks}_direction'] = timed(lambda:engine.direction(r,0.,p,M))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
