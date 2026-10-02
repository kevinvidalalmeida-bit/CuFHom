from contextlib import redirect_stdout
from types import SimpleNamespace
import os,json
from pathlib import Path
import numpy as np
import cupy as cp
from cufhom_acoustic_local import AcousticLocal,acoustic_matrices,trial
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom.pipeline.fft_solver import solve_homogenization
from cufhom_core.tensors import fft
fft.set_fft_backend('cupy');fft.set_cupy_plan_mode('manual')
rng=np.random.default_rng(1001)
rows=[]
for shape in ((5,7,5),(6,4,8)):
    Q=SymmetricCoordinates(MixedProjector(shape,(1.,1.3,.8),[True]*6,cp),4.,2.)
    A=rng.normal(size=(3,6,6));C=A.swapaxes(1,2)@A+3*np.eye(6)
    ids=rng.integers(0,3,size=shape,dtype=np.int32)
    table=C[:,np.triu_indices(6)[0],np.triu_indices(6)[1]]
    D=acoustic_matrices(C,4.,2.)
    assert np.linalg.eigvalsh(D).min()>0
    gpu=AcousticLocal(SimpleNamespace(val=cp.asarray(ids),material_table=cp.asarray(table,dtype=cp.float32)),Q,4.,2.,[])
    cpuQ=SymmetricCoordinates(MixedProjector(shape,(1.,1.3,.8),[True]*6,np),4.,2.)
    k,valid,weight=cpuQ._cpu_geometry();N=np.prod(shape)
    def coords(field):return (np.fft.rfftn(field,axes=(1,2,3))*Q.norm*weight*valid).ravel()
    def real(v):return np.fft.irfftn(v.reshape((3,)+Q.hshape)*Q.norm/weight,s=shape,axes=(1,2,3))*N
    v=coords(rng.normal(size=(3,)+shape));w=coords(rng.normal(size=(3,)+shape))
    inv=np.linalg.inv(D)[ids]
    expected=coords(np.einsum('...ij,j...->i...',inv,real(v)))
    output=cp.asnumpy(gpu.apply(cp.asarray(v,dtype=cp.complex64)))
    assert np.linalg.norm(output-expected)/np.linalg.norm(expected)<1e-6
    bv=gpu.apply(cp.asarray(v,dtype=cp.complex64));bw=gpu.apply(cp.asarray(w,dtype=cp.complex64))
    assert float(cp.vdot(cp.asarray(v),bv).real)>0
    left=float(cp.vdot(cp.asarray(v),bw).real);right=float(cp.vdot(bv,cp.asarray(w)).real)
    assert abs(left-right)<1e-5*max(abs(left),abs(right),1.)
    phase=np.zeros(shape,np.uint8);phase[:2]=1
    ori=np.empty(shape+(3,),np.float32);ori[:]=np.array([1,2,3])/np.sqrt(14)
    p=dict(input_dir='.',phase_array=phase,ori_array=ori,Em=3.,nu_m=.3,
        Ef_L=90.,Ef_T=10.,nu_LT=.25,nu_TT=.3,G_LT=6.,solver_rtol=1e-5,solver_verbose=False)
    out=Path('benchmarks/results/acoustic_local_validation_20261001');out.mkdir(exist_ok=True)
    with open(os.devnull,'w') as sink,redirect_stdout(sink):
        ref=solve_homogenization(dict(p,fft_backend='scipy',solver_rtol=1e-10))
        with trial(records:=[]):result=solve_homogenization(dict(p,fft_backend='cupy',solver_timing_path=str(out/f'{shape[0]}.json')))
    err=float(np.linalg.norm(result-ref)/np.linalg.norm(ref));assert err<2e-6
    info=json.loads((out/f'{shape[0]}.json').read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
    assert all(i['true_norm_res_rel']<=1e-5 for i in info)
    rows.append(dict(shape=shape,Ceff_error=err,records=records,infos=info))
(out/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
print('GPU/CPU independent Fourier maps, symmetry, positivity and six-load CPU64 references pass')
