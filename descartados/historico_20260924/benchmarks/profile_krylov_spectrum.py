"""Lanczos/Ritz diagnostic from existing CG scalars; no additional FFT calls.

Estimates, not certified spectral bounds. Discard recurrence interpretation
if CG performed a reliable restart. Never changes stopping tolerances.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from profile_g09_fft import (ROOT, G09_N240, MATERIAL, cp, sync, quiet_solver,
    optimized_parameters, compile_cfield_geometry, solve_homogenization)
from pipeline import linear_coordinates as linear
from pipeline import linear_krylov


def tridiagonal(alpha, beta):
    alpha, beta = np.asarray(alpha), np.asarray(beta)
    if alpha.ndim!=1 or beta.shape!=alpha.shape or not np.isfinite([alpha,beta]).all() or (alpha<=0).any() or (beta<0).any():
        raise ValueError('Invalid CG recurrence')
    diagonal = 1/alpha
    diagonal[1:] += beta[:-1]/alpha[:-1]
    off = np.sqrt(beta[:-1])/alpha[:-1]
    return np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)


@contextmanager
def capture_spectrum():
    original_cg, original_update = linear.fused_cg, linear_krylov.fused_update
    records = []
    def cg(action, rhs, dot, *args, **kwargs):
        record = dict(alpha=[],beta=[],action_calls=0)
        previous = [None]
        def observed_dot(x,y):
            value = dot(x,y)
            if previous[0] is None:
                previous[0] = value
            return value
        def observed_action(x):
            record['action_calls'] += 1
            return action(x)
        def update(p,ap,alpha,x,r,partial):
            rr = original_update(p,ap,alpha,x,r,partial)
            record['alpha'].append(float(alpha))
            record['beta'].append(rr/previous[0])
            previous[0] = rr
            return rr
        with patch.object(linear_krylov,'fused_update',update):
            result, count = original_cg(observed_action,rhs,observed_dot,*args,**kwargs)
        record['iterations'] = count
        record['no_restarts'] = record['action_calls']==count+1
        if count and record['no_restarts']:
            values = np.linalg.eigvalsh(tridiagonal(record['alpha'],record['beta']))
            record.update(ritz=values.tolist(),ritz_ratio=float(values[-1]/values[0]))
        records.append(record)
        return result,count
    with patch.object(linear,'fused_cg',cg):
        yield records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output path')
    phase=np.load(G09_N240/'phase.npy'); ori=np.load(G09_N240/'ori.npy')
    phase.flags.writeable=ori.flags.writeable=False
    config=optimized_parameters(phase,ori,json.loads(MATERIAL.read_text()),compile_cfield_geometry(phase,ori))
    with quiet_solver():
        reference=solve_homogenization(config)
    with capture_spectrum() as records,quiet_solver():
        result=solve_homogenization(config)
    sync()
    difference=float(np.linalg.norm(result-reference)/np.linalg.norm(reference))
    if difference>1e-12 or not np.isfinite(difference):
        raise RuntimeError('Scalar observation changed the result')
    paths=['FFT/benchmarks/profile_krylov_spectrum.py','FFT/pipeline/linear_krylov.py','FFT/pipeline/linear_coordinates.py']
    report=dict(scope=__doc__,Ceff_difference=difference,loads=records,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for i,r in enumerate(records):
        print(i,r['iterations'],r.get('ritz_ratio'),r.get('ritz',[]),flush=True)


if __name__=='__main__':
    main()
