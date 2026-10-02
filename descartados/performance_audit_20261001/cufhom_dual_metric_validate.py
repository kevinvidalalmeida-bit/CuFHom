from contextlib import redirect_stdout
from pathlib import Path
import os,json
import numpy as np
import cupy as cp
from cufhom_physical_metric import Metric
from cufhom_dual_metric import trial
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom.pipeline.fft_solver import solve_homogenization
from cufhom_core.tensors import fft
fft.set_fft_backend('cupy');fft.set_cupy_plan_mode('manual')
rng=np.random.default_rng(101);rows=[]
for shape in ((5,7,5),(6,4,8)):
 P=MixedProjector(shape,(1.,1.3,.8),[True]*6,cp);Q=SymmetricCoordinates(P,4.,2.);M=Metric(Q)
 stress=cp.asarray(rng.normal(size=(6,)+shape),dtype=cp.float32);residual=P(stress)
 z=Q.adjoint(residual)/np.prod(shape)
 got=M.norm(z)[1]*np.prod(shape)
 expected=float(cp.sum(residual.astype(cp.float64)**2))/np.prod(shape)
 assert abs(got/expected-1)<2e-6,(got,expected)
 phase=np.zeros(shape,np.uint8);phase[:2]=1;ori=np.empty(shape+(3,),np.float32);ori[:]=np.array([1,2,3])/np.sqrt(14)
 p=dict(input_dir='.',phase_array=phase,ori_array=ori,Em=3.,nu_m=.3,Ef_L=90.,Ef_T=10.,nu_LT=.25,nu_TT=.3,G_LT=6.,solver_verbose=False)
 out=Path('benchmarks/results/dual_metric_validation_20261001');out.mkdir(exist_ok=True)
 with open(os.devnull,'w') as sink,redirect_stdout(sink):
  ref=solve_homogenization(dict(p,fft_backend='scipy',solver_rtol=1e-10))
  with trial():C=solve_homogenization(dict(p,fft_backend='cupy',solver_rtol=1e-5,solver_timing_path=str(out/f'{shape[0]}.json')))
 error=float(np.linalg.norm(C-ref)/np.linalg.norm(ref));assert error<2e-6
 infos=json.loads((out/f'{shape[0]}.json').read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
 assert all(i['true_norm_res_rel']<=1e-5 for i in infos)
 rows.append(dict(shape=shape,metric_relative_error=abs(got/expected-1),Ceff_error=error,infos=infos))
(out/'report.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Physical metric identity checked against projected stress, CPU64 periodic references pass')
