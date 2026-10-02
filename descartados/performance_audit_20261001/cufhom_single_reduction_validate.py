import numpy as np
import cupy as cp
from cufhom_single_reduction import cg
rng=np.random.default_rng(1001)
for xp in (np,cp):
    q,_=np.linalg.qr(rng.normal(size=(257,257)))
    h=(q*np.geomspace(1,100,257))@q.T;b=rng.normal(size=257)
    a=xp.asarray(h,dtype=xp.float64 if xp is np else xp.float32)
    rhs=xp.asarray(b,dtype=a.dtype)
    dot=lambda x,y:float(xp.sum(x.astype(xp.float64)*y.astype(xp.float64)))
    for smooth in (False,True):
        tol=1e-10 if xp is np else 1e-5
        x,it=cg(lambda v:a@v,rhs,dot,tol,500,xp,residual_smoothing=smooth)
        v=x if xp is np else cp.asnumpy(x)
        assert np.linalg.norm(b-h@v)/np.linalg.norm(b)<(2e-10 if xp is np else 1.1e-5)
        assert np.linalg.norm(v-np.linalg.solve(h,b))/np.linalg.norm(np.linalg.solve(h,b))<100*tol
        print(xp.__name__,smooth,it)
print('Dense CPU64 and GPU32 pass with independent equation checks')
