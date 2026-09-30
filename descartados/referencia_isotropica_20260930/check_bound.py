"""Check a constitutive spectral bound for isotropic reference bulk/shear ratio."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from cufhom.pipeline.fft_solver import TI_stiffness_voigt,voigt_to_mandel,Enu_to_KG
from cufhom_core.mechanics.matcoef import ElasticTensor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--material',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise ValueError('Use a new output path')
m=json.loads(a.material.read_text())
K,G=Enu_to_KG(m['Em'],m['nu_m'])
Cm=ElasticTensor(bulk=K,mu=G).mandel
Cf=voigt_to_mandel(TI_stiffness_voigt(*(m[k] for k in ('Ef_L','Ef_T','nu_LT','nu_TT','G_LT'))))
u=np.array([1,1,1,0,0,0])/np.sqrt(3)
J=np.outer(u,u);D=np.eye(6)-J
C=np.stack([Cm,Cf])
def bound(ratio):
    root=J/np.sqrt(3*ratio)+D/np.sqrt(2)
    eigen=np.linalg.eigvalsh(root@C@root)
    if eigen.min()<=0:raise ValueError('SPD phases required')
    return float(eigen.max()/eigen.min())
result=minimize_scalar(lambda v:np.log(bound(np.exp(v))),bounds=(-2,4),method='bounded')
report=dict(scope='Upper bound from pointwise constitutive ratios; not a certified condition number of the actual FFT matrix. Isotropic reference is rotation invariant, so the two local phase tensors suffice for this material.',
    current_ratio=K/G,current_bound=bound(K/G),minimizer_ratio=float(np.exp(result.x)),
    optimized_bound=bound(np.exp(result.x)),search_log_ratio_bounds=[-2,4],
    scan=[dict(ratio=r,bound=bound(r)) for r in (.4,.7,1.,1.5,2.,3.,4.,6.,10.)])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
