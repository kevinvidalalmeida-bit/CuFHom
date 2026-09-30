from pathlib import Path
from contextlib import redirect_stdout
import json,os
import numpy as np
import cupy as cp
from cufhom_blockcg import Ops,block_cg,trial
from cufhom.pipeline.fft_solver import solve_homogenization
out=Path('benchmarks/results/block_work_20260930');out.mkdir(exist_ok=True)
rng=np.random.default_rng(309)
for xp in (np,cp):
    ctype=np.complex128 if xp is np else np.complex64
    a=(rng.normal(size=(3,259))+1j*rng.normal(size=(3,259))).astype(ctype)
    b=(rng.normal(size=(6,259))+1j*rng.normal(size=(6,259))).astype(ctype)
    ops=Ops(xp)
    np.testing.assert_allclose(ops.gram(xp.asarray(a),xp.asarray(b)),(a.astype(complex).conj()@b.astype(complex).T).real,rtol=1e-12,atol=1e-12)
    coeff=rng.normal(size=(4,3));v=ops.combine(xp.asarray(a),coeff)
    got=np.asarray(v) if xp is np else cp.asnumpy(v)
    np.testing.assert_allclose(got,coeff@a,rtol=2e-7,atol=2e-7)
    # SPD with redundant and zero RHS, so the block rank must decrease safely.
    h=rng.normal(size=(100,100));h=h.T@h+5*np.eye(100)
    B=rng.normal(size=(4,100));B[2]=B[0]+B[1];B[3]=0
    hg=xp.asarray(h,dtype=np.float64 if xp is np else np.float32)
    bg=xp.asarray(B,dtype=hg.dtype)
    x,it,record=block_cg(lambda v:hg@v,bg,1e-10 if xp is np else 1e-5,200,xp)
    x=np.asarray(x) if xp is np else cp.asnumpy(x)
    error=np.linalg.norm(B-x@h.T,axis=1)/np.maximum(np.linalg.norm(B,axis=1),1.)
    assert np.max(error)<(2e-10 if xp is np else 2e-5),error
rows=[]
for case in ('laminate','compact'):
    shape=(10,10,10);phase=np.zeros(shape,np.uint8)
    if case=='laminate':phase[:4]=1
    else:
        coords=np.indices(shape)
        for center in ((2,3,4),(7,6,7)):
            phase[np.sum((coords-np.array(center)[:,None,None,None])**2,axis=0)<=5]=1
    ori=np.zeros(shape+(3,),np.float32);ori[:]=np.array([1,2,3])/np.sqrt(14)
    p=dict(input_dir='.',phase_array=phase,ori_array=ori,Em=3.,nu_m=.3,
        Ef_L=90.,Ef_T=10.,nu_LT=.25,nu_TT=.3,G_LT=6.,solver_rtol=1e-5,solver_verbose=False)
    with open(os.devnull,'w') as sink,redirect_stdout(sink):
        ref=solve_homogenization(dict(p,fft_backend='scipy',solver_rtol=1e-10))
        records=[];timing=out/f'{case}.json'
        with trial(records):
            result=solve_homogenization(dict(p,fft_backend='cupy',solver_timing_path=str(timing)))
    error=float(np.linalg.norm(result-ref)/np.linalg.norm(ref));assert error<2e-6,error
    data=json.loads(timing.read_text());infos=data['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert all(i['true_norm_res_rel']<=1e-5 for i in infos)
    rows.append(dict(case=case,Ceff_error=error,records=records,infos=infos))
(out/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='infos'} for r in rows],indent=2))
