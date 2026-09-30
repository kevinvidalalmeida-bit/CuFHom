from pathlib import Path
from contextlib import redirect_stdout
import json,os
import numpy as np
import cupy as cp
from cufhom_twolevel import FourierBasis, CoarseCorrection, trial
from cufhom.pipeline.nonlinear_coordinates import DisplacementCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom.pipeline.fft_solver import solve_homogenization

output=Path('benchmarks/results/twolevel_work_20260930')
# Independent CPU isometry on real admissible RFFT modes (including kz=0).
for shape in ((5,7,5),(6,4,8)):
    U=FourierBasis(shape,1,np)
    Q=DisplacementCoordinates(MixedProjector(shape,(1,1,1),[True]*6,np))
    cols=[]
    for j in range(U.rank):
        u=U.column(j,np.complex128)
        e=np.fft.irfftn(Q._map(u,False),s=shape,axes=(1,2,3))*np.prod(shape)
        h=np.fft.rfftn(e,axes=(1,2,3))
        back=Q._map(h,True)
        np.testing.assert_allclose(back,u,atol=2e-15)
        cols.append(e.ravel())
    np.testing.assert_allclose(np.array(cols)@np.array(cols).T,np.eye(U.rank),atol=3e-15)
# Dense independent SPD algebra, unrelated to a fibre or FFT constitutive tensor.
rng=np.random.default_rng(930)
shape=(3,3,3); basis=FourierBasis(shape,1,np)
n=2*basis.size
h=rng.normal(size=(n,n)); h=h.T@h+np.eye(n)
hg=cp.asarray(h)
r=cp.asarray(rng.normal(size=n)).view(cp.complex128)
action=lambda z: (hg@z.view(cp.float64)).view(cp.complex128)
U=np.array([basis.column(j,np.complex128).view(np.float64) for j in range(basis.rank)]).T
E=U.T@h@U; inverse=np.linalg.inv(E)
for kind in ('block','balanced'):
    correction=CoarseCorrection(action,r,shape,1,kind)
    if kind=='block': B=np.eye(n)+U@(inverse-np.eye(basis.rank))@U.T
    else:
        P=np.eye(n)-h@U@inverse@U.T
        B=P.T@P+U@inverse@U.T
    assert np.linalg.eigvalsh(B)[0]>0
    np.testing.assert_allclose(cp.asnumpy(correction(r)).view(np.float64),B@cp.asnumpy(r).view(np.float64),rtol=2e-12,atol=2e-12)
rows=[]
for case in ('laminate','compact'):
    shape=(10,10,10)
    phase=np.zeros(shape,np.uint8)
    if case=='laminate':phase[:4]=1
    else:
        coords=np.indices(shape)
        for center in ((2,3,4),(7,6,7)):
            phase[np.sum((coords-np.array(center)[:,None,None,None])**2,axis=0)<=5]=1
    ori=np.zeros(shape+(3,),np.float32);ori[:]=np.array([1,2,3])/np.sqrt(14)
    p=dict(input_dir='.',phase_array=phase,ori_array=ori,Em=3.,nu_m=.3,
        Ef_L=90.,Ef_T=10.,nu_LT=.25,nu_TT=.3,G_LT=6.,solver_rtol=1e-5,solver_verbose=False)
    with open(os.devnull,'w') as sink, redirect_stdout(sink):
        ref=solve_homogenization(dict(p,fft_backend='scipy',solver_rtol=1e-10))
        for kind,radius in (('base',0),('block',1),('block',2),('balanced',1)):
            from contextlib import nullcontext
            records=[]
            timing=output/f'{case}_{kind}_{radius}.json'
            with nullcontext() if kind=='base' else trial(shape,radius,kind,records):
                C=solve_homogenization(dict(p,fft_backend='cupy',solver_timing_path=str(timing)))
            d=json.loads(timing.read_text()); infos=d['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            error=float(np.linalg.norm(C-ref)/np.linalg.norm(ref))
            assert error<2e-6,error
            assert all(i['true_norm_res_rel']<=1e-5 for i in infos)
            rows.append(dict(case=case,kind=kind,radius=radius,relative_Ceff_error=error,
                cg=sum(i['cg_iterations'] for i in infos),coarse=records))
(output/'small_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
