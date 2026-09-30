from contextlib import redirect_stdout
from pathlib import Path
import os,json
import numpy as np
import cupy as cp
from cufhom_smoothed_late import smoothed as cr,trial
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction
from cufhom_core.tensors import Tensor
from cufhom.pipeline.fft_solver import solve_homogenization
rng=np.random.default_rng(930)
for xp in (np,cp):
    n=300;h=rng.normal(size=(n,n));h=h.T@h+5*np.eye(n)
    rhs=rng.normal(size=n)
    if xp is cp:
        a=cp.asarray(h,dtype=cp.float32);b=cp.asarray(rhs,dtype=cp.float32).view(cp.complex64)
        action=lambda x:(a@x.view(cp.float32)).view(cp.complex64)
        reducer=CupyWarpCGReduction(Tensor(name='b',val=b.view(cp.float32),N=(n,),order=0,Fourier=False))
        def dot(x,y):
            reducer.dot_kernel((reducer.blocks,),(256,),(x,y,reducer.partial,np.int64(n)))
            return float(cp.sum(reducer.partial))
    else:
        a=h;b=rhs;action=lambda x:a@x;dot=lambda x,y:float(x@y)
    tol=1e-10 if xp is np else 1e-5
    x,it=cr(action,b,dot,tol,500,xp)
    x=x if xp is np else cp.asnumpy(x).view(np.float32)
    assert np.linalg.norm(rhs-h@x)/np.linalg.norm(rhs)<(2e-10 if xp is np else 2e-5)
    reference=np.linalg.solve(h,rhs)
    relative_error=np.linalg.norm(x-reference)/np.linalg.norm(reference)
    # Residual tolerance controls a norm, not componentwise relative error at zeros.
    assert relative_error < 1.1*np.linalg.cond(h)*tol,relative_error
rows=[];out=Path('benchmarks/results/smoothed_late_work_20260930');out.mkdir(exist_ok=True)
for shape in ((5,7,5),(6,4,8)):
    phase=np.zeros(shape,np.uint8);phase[:2]=1
    ori=np.zeros(shape+(3,),np.float32);ori[:]=np.array([1,2,3])/np.sqrt(14)
    p=dict(input_dir='.',phase_array=phase,ori_array=ori,Em=3.,nu_m=.3,
        Ef_L=90.,Ef_T=10.,nu_LT=.25,nu_TT=.3,G_LT=6.,solver_rtol=1e-5,solver_verbose=False)
    with open(os.devnull,'w') as sink,redirect_stdout(sink):
        ref=solve_homogenization(dict(p,fft_backend='scipy',solver_rtol=1e-10))
        timing=out/f'{shape[0]}.json'
        with trial():C=solve_homogenization(dict(p,fft_backend='cupy',solver_timing_path=str(timing)))
    error=float(np.linalg.norm(C-ref)/np.linalg.norm(ref));assert error<2e-6,error
    infos=json.loads(timing.read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert all(i['true_norm_res_rel']<=1e-5 for i in infos)
    rows.append(dict(shape=shape,Ceff_error=error,infos=infos))
(out/'validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print('CPU and GPU dense solve and two complete small six-load physical cases passed')
