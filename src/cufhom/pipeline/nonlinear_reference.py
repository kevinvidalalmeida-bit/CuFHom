"""SPD homogeneous reference inverse on the unchanged admissible FFT space."""
import numpy as np
from cufhom_core.tensors import fft as core_fft
from .nonlinear_projection import _CODE

_KERNELS = {}


class IsotropicReference:
    def __init__(self, projector, K, mu):
        if not np.isfinite([K, mu]).all() or min(K, mu) <= 0:
            raise ValueError('Reference bulk/shear moduli must be positive')
        self.P, self.K, self.mu = projector, K, mu
        self.shift = None

    def set_shift(self, shift, dtype):
        if self.shift == shift:
            return
        self.shift = shift
        C = np.eye(6)*(2*self.mu+shift)
        C[:3, :3] += self.K-2*self.mu/3
        free = self.P.free
        M = np.zeros((6, 6))
        M[np.ix_(free, free)] = np.linalg.inv(C[np.ix_(free, free)])
        self.mean = self.P.xp.asarray(M, dtype=dtype)
        self.a = 1/(2*self.mu+shift)
        self.b = 1/(self.K+4*self.mu/3+shift)

    def __call__(self, value):
        P, xp = self.P, self.P.xp
        with P.timers.measure('preconditioner'):
            h = core_fft.rfftn_cached(value, P.shape, (1, 2, 3))
            if xp is np:
                out = self.a*P._formula(h)
                x, y, z = P.frequencies
                q = x*x+y*y+z*z
                inv = 1/np.where(q > 0, q*q, 1)
                nn = (x*x, y*y, z*z, np.sqrt(2)*y*z, np.sqrt(2)*x*z, np.sqrt(2)*x*y)
                s = sum(n*v for n, v in zip(nn, h))*inv
                for c in range(6):
                    out[c] += (self.b-self.a)*nn[c]*s
                for axis, n in enumerate(P.shape):
                    if n % 2 == 0:
                        sl = [slice(None)]*4; sl[axis+1] = n//2
                        out[tuple(sl)] = 0
                out[:, 0, 0, 0] = self.mean @ h[:, 0, 0, 0]
                return core_fft.irfftn_cached(out, P.shape, (1, 2, 3))
            key = h.dtype.str
            if key not in _KERNELS:
                code = _CODE.replace('int free_mask)', 'int free_mask, const REAL* mean, REAL a, REAL b)')
                start = code.index(' if(s==0)')
                end = code.index(' if((n0%2', start)
                code = code[:start]+r'''
 if(s==0) {
   for(int c=0;c<6;c++) {
     REAL re=0,im=0;
     for(int d=0;d<6;d++){re+=mean[6*c+d]*x[d*count].x;im+=mean[6*c+d]*x[d*count].y;}
     y[c*count]=MAKE(scale*re,scale*im);
   }
   return;
 }
'''+code[end:]
                # Rename the complex local array to avoid scalar coefficient a.
                code = code.replace('a[6]', 'values[6]').replace('a[c]', 'values[c]')
                code = code.replace('   for(int c=0;c<6;c++) {\n     if(part)', '''
   REAL nn[6]={k0*k0*q,k1*k1*q,k2*k2*q,r2*k1*k2*q,r2*k0*k2*q,r2*k0*k1*q};
   for(int c=0;c<6;c++) {
     p[c]=a*p[c]+(b-a)*nn[c];
     if(part)''')
                name = 'float' if h.dtype == xp.complex64 else 'double'
                code = code.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
                _KERNELS[key] = xp.RawKernel(code, 'project')
            out = xp.empty_like(h)
            d = value.dtype.type
            count = h.size//6
            _KERNELS[key](((count+255)//256,), (256,),
                (h, out, *(np.int64(n) for n in P.shape), np.int64(h.shape[-1]),
                 *(d(1/L) for L in P.lengths), d(1/np.prod(P.shape)), np.int32(0),
                 self.mean, d(self.a), d(self.b)))
            return core_fft.irfftn_unscaled_cached(out, P.shape, (1, 2, 3))
