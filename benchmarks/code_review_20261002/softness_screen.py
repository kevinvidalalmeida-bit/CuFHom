"""FP32 cancellation screen for H=I+QR*(C-C0)QR; no timing claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from cufhom.pipeline.linear_energy import energy_action
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector,cp
from cufhom_core.tensors import fft
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'descartados/reference_split_20261002'))
from reference_split import action,prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output file')
    shape=(17,15,13);K,G=3.,2.
    base=np.zeros((6,6));base[:3,:3]=K-2*G/3
    np.fill_diagonal(base[:3,:3],K+4*G/3);base[3:,3:]=2*G*np.eye(3)
    cpu=SymmetricCoordinates(MixedProjector(shape,(2.,3.,4.),np.ones(6,bool),np),K,G)
    Q=SymmetricCoordinates(MixedProjector(shape,(2.,3.,4.),np.ones(6,bool),cp),K,G)
    fft.set_fft_backend('scipy')
    p64=cpu.adjoint(np.random.default_rng(325).normal(size=(6,)+shape))
    p=cp.asarray(p64,cp.complex64)
    # Sum the rounded input coordinates in CPU64 for an independent reference.
    rounded=cp.asnumpy(p).astype(np.complex128);expected_norm=np.vdot(rounded,rounded).real
    fft.set_fft_backend('cupy');fft.set_cupy_plan_mode('manual');fft.set_cupy_rfft_batch_mode('full')
    indices=np.triu_indices(6);rows=[]
    for eta in (1.,.1,.01,.001,.0001,.00001,.000001):
        matrices=np.asarray([base,eta*base])
        descriptor=SimpleNamespace(val=cp.ones(shape,cp.int32),material_table=
            cp.ascontiguousarray(cp.asarray(matrices[:,indices[0],indices[1]],cp.float32)))
        original,full=energy_action(Q,descriptor,p);original.spectrum=None
        split,reduced=action(Q,descriptor,prepare(Q,descriptor),p);split.spectrum=None
        reference=eta*expected_norm
        row=dict(eta=eta,reference_pap=reference,full_pap=full,split_pap=reduced,
                 full_relative_error=abs(full/reference-1),split_relative_error=abs(reduced/reference-1))
        rows.append(row);print(eta,row['full_relative_error'],row['split_relative_error'],flush=True)
    assert max(r['full_relative_error'] for r in rows)<2e-6
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    sources=list((root/'src').rglob('*.py'))+[Path(__file__),root/'descartados/reference_split_20261002/reference_split.py']
    args.output.write_text(json.dumps(dict(scope='Isotropic homogeneous soft phases, admissible complex64 coordinates, FP64 products; CPU64 independent exact eta*norm; no wall time claim.',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
