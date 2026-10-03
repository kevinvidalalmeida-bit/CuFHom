"""Shared admissible FFT space, mixed loading, coordinates and reference inverse."""
from contextlib import contextmanager
from types import SimpleNamespace
import time
from cufhom_core.tensors.objects import elasticity_direct_array
import numpy as np
from cufhom_core.tensors import fft as core_fft

_PROJECT_KERNELS = {}
_PROJECT_CODE = r'''
extern "C" __global__ void project(const COMPLEX* x, COMPLEX* y,
 long long n0, long long n1, long long n2, long long nz,
 REAL iy0, REAL iy1, REAL iy2, REAL scale, int free_mask) {
 const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const long long count=n0*n1*nz;
 if(s>=count)return;
 const long long i2=s%nz, i1=(s/nz)%n1, i0=s/(nz*n1);
 if(s==0) {
   for(int c=0;c<6;c++) {
     COMPLEX z=x[c*count];
     y[c*count]=MAKE((free_mask&(1<<c))?scale*z.x:0,
                     (free_mask&(1<<c))?scale*z.y:0);
   }
   return;
 }
 if((n0%2==0 && i0==n0/2) || (n1%2==0 && i1==n1/2) ||
    (n2%2==0 && i2==n2/2)) {
   for(int c=0;c<6;c++)y[c*count+s]=MAKE(0,0);
   return;
 }
 const REAL k0=(i0<=n0/2?i0:i0-n0)*iy0;
 const REAL k1=(i1<=n1/2?i1:i1-n1)*iy1;
 const REAL k2=i2*iy2;
 const REAL inv=1/(k0*k0+k1*k1+k2*k2);
 const REAL r2=(REAL)1.4142135623730950488;
 COMPLEX a[6], out[6];
 for(int c=0;c<6;c++)a[c]=x[c*count+s];
 for(int part=0;part<2;part++) {
   REAL h[6], p[6];
   for(int c=0;c<6;c++)h[c]=part?a[c].y:a[c].x;
   const REAL v0=h[0]*k0+(h[5]*k1+h[4]*k2)/r2;
   const REAL v1=h[1]*k1+(h[5]*k0+h[3]*k2)/r2;
   const REAL v2=h[2]*k2+(h[4]*k0+h[3]*k1)/r2;
   const REAL q=(k0*v0+k1*v1+k2*v2)*inv*inv;
   p[0]=2*k0*v0*inv-k0*k0*q;
   p[1]=2*k1*v1*inv-k1*k1*q;
   p[2]=2*k2*v2*inv-k2*k2*q;
   p[3]=r2*((k1*v2+k2*v1)*inv-k1*k2*q);
   p[4]=r2*((k0*v2+k2*v0)*inv-k0*k2*q);
   p[5]=r2*((k0*v1+k1*v0)*inv-k0*k1*q);
   for(int c=0;c<6;c++) {
     if(part)out[c].y=scale*p[c];else out[c].x=scale*p[c];
   }
 }
 for(int c=0;c<6;c++)y[c*count+s]=out[c];
}
'''


def project_normalized(cp, spectrum, shape, lengths, free):
    """Return P(spectrum)/N for a subsequent unnormalized cuFFT inverse."""
    dtype = np.dtype(spectrum.dtype)
    if dtype not in (np.dtype('complex64'), np.dtype('complex128')):
        raise ValueError('Expected complex64 or complex128 spectrum')
    if not spectrum.flags.c_contiguous:
        raise ValueError('Expected contiguous Mandel spectrum')
    expected = (6,)+tuple(shape[:-1])+(shape[-1]//2+1,)
    if spectrum.shape != expected:
        raise ValueError('Unexpected spectrum shape')
    real = np.float32 if dtype == np.dtype('complex64') else np.float64
    kernel = _PROJECT_KERNELS.get(dtype.str)
    if kernel is None:
        name = 'float' if real == np.float32 else 'double'
        code = _PROJECT_CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
        kernel = cp.RawKernel(code, 'project')
        _PROJECT_KERNELS[dtype.str] = kernel
    out = cp.empty_like(spectrum)
    count = spectrum.size//6
    kernel(((count+255)//256,), (256,),
        (spectrum, out, *(np.int64(n) for n in shape), np.int64(spectrum.shape[-1]),
         *(real(1/L) for L in lengths), real(1/np.prod(shape)),
         np.int32(sum(1 << i for i in free))))
    return out


_REFERENCE_KERNELS = {}


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
            if key not in _REFERENCE_KERNELS:
                code = _PROJECT_CODE.replace('int free_mask)', 'int free_mask, const REAL* mean, REAL a, REAL b)')
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
                _REFERENCE_KERNELS[key] = xp.RawKernel(code, 'project')
            out = xp.empty_like(h)
            d = value.dtype.type
            count = h.size//6
            _REFERENCE_KERNELS[key](((count+255)//256,), (256,),
                (h, out, *(np.int64(n) for n in P.shape), np.int64(h.shape[-1]),
                 *(d(1/L) for L in P.lengths), d(1/np.prod(P.shape)), np.int32(0),
                 self.mean, d(self.a), d(self.b)))
            return core_fft.irfftn_unscaled_cached(out, P.shape, (1, 2, 3))


_COORDINATE_KERNELS = {}
_PRECONDITIONERS = {}
_PRECONDITION_CODE = r'''
extern "C" __global__ void reference(const COMPLEX* x,COMPLEX* y,
 long long n0,long long n1,long long nz,REAL iy0,REAL iy1,REAL iy2,
 REAL a,REAL b,const REAL* mean,int free_mask) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;
 if(s==0) {
   for(int c=0;c<3;c++)y[c*count]=MAKE(0,0);
   int i=0;
   for(int c=0;c<6;c++)if(free_mask&(1<<c)) {
     REAL value=0;int j=0;
     for(int d=0;d<6;d++)if(free_mask&(1<<d))value+=mean[c*6+d]*x[3*count+j++].x;
     y[3*count+i++]=MAKE(value,0);
   }
   return;
 }
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 REAL k[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
 REAL inv=1/(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
 REAL re=0,im=0;
 for(int c=0;c<3;c++){re+=k[c]*x[c*count+s].x;im+=k[c]*x[c*count+s].y;}
 for(int c=0;c<3;c++)y[c*count+s]=MAKE(a*x[c*count+s].x+(b-a)*k[c]*re*inv,
                                                      a*x[c*count+s].y+(b-a)*k[c]*im*inv);
}
'''
_COORDINATE_CODE = r'''
extern "C" __global__ void coordinates(const COMPLEX* x, COMPLEX* y,
 long long n0,long long n1,long long n2,long long nz,
 REAL iy0,REAL iy1,REAL iy2,REAL norm,int free_mask,int adjoint) {
 const long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const long long count=n0*n1*nz;
 if(s>=count)return;
 const long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 if(s==0) {
   if(adjoint) {
     for(int c=0;c<3;c++)y[c*count]=MAKE(0,0);
     int j=0;
     for(int c=0;c<6;c++)if(free_mask&(1<<c))y[3*count+j++]=MAKE(norm*x[c*count].x,0);
   } else {
     int j=0;
     for(int c=0;c<6;c++) {
       y[c*count]=MAKE((free_mask&(1<<c))?norm*x[3*count+j++].x:0,0);
     }
   }
   return;
 }
 if((n0%2==0&&i0==n0/2)||(n1%2==0&&i1==n1/2)||(n2%2==0&&i2==n2/2)) {
   for(int c=0;c<(adjoint?3:6);c++)y[c*count+s]=MAKE(0,0);
   return;
 }
 REAL k[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
 REAL inv=1/sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]);
 for(int c=0;c<3;c++)k[c]*=inv;
 const REAL r2=(REAL)1.4142135623730950488;
 const REAL weight=(i2==0||(n2%2==0&&i2==n2/2))?1:r2;
 const REAL scale=adjoint?norm*weight:norm/weight;
 COMPLEX out[6];
 for(int part=0;part<2;part++) {
   REAL h[6],v[3],p[6];
   for(int c=0;c<(adjoint?6:3);c++)h[c]=part?x[c*count+s].y:x[c*count+s].x;
   if(adjoint) {
     v[0]=k[0]*h[0]+(k[1]*h[5]+k[2]*h[4])/r2;
     v[1]=k[1]*h[1]+(k[0]*h[5]+k[2]*h[3])/r2;
     v[2]=k[2]*h[2]+(k[0]*h[4]+k[1]*h[3])/r2;
   } else {for(int c=0;c<3;c++)v[c]=h[c];}
   REAL dot=k[0]*v[0]+k[1]*v[1]+k[2]*v[2];
   for(int c=0;c<3;c++)v[c]=r2*v[c]+(1-r2)*k[c]*dot;
   if(adjoint) {for(int c=0;c<3;c++)p[c]=v[c];}
   else {
     for(int c=0;c<3;c++)p[c]=k[c]*v[c];
     p[3]=(k[1]*v[2]+k[2]*v[1])/r2;
     p[4]=(k[0]*v[2]+k[2]*v[0])/r2;
     p[5]=(k[0]*v[1]+k[1]*v[0])/r2;
   }
   for(int c=0;c<(adjoint?3:6);c++) {
     if(part)out[c].x=(adjoint?1:-1)*scale*p[c];
     else out[c].y=(adjoint?-1:1)*scale*p[c];
   }
 }
 for(int c=0;c<(adjoint?3:6);c++)y[c*count+s]=out[c];
}
'''


class DisplacementCoordinates:
    def __init__(self, projector):
        self.P = projector
        self.xp = projector.xp
        self.shape = projector.shape
        self.hshape = self.shape[:-1]+(self.shape[-1]//2+1,)
        self.count = int(np.prod(self.hshape))
        self.size = 3*self.count+len(projector.free)
        self.norm = 1/np.sqrt(np.prod(self.shape))
        self.cached = None

    def _cpu_geometry(self):
        if self.cached is None:
            f = [np.fft.fftfreq(n)*n/L for n, L in zip(self.shape, self.P.lengths)]
            f[-1] = np.fft.rfftfreq(self.shape[-1])*self.shape[-1]/self.P.lengths[-1]
            k = np.asarray(np.meshgrid(*f, indexing='ij'))
            norm = np.sqrt(np.sum(k*k, axis=0))
            k /= np.where(norm > 0, norm, 1)
            valid = norm > 0
            for axis, n in enumerate(self.shape):
                if n % 2 == 0:
                    sl = [slice(None)]*3; sl[axis] = n//2
                    valid[tuple(sl)] = False
            weight = np.full(self.hshape[-1], np.sqrt(2.)); weight[0] = 1
            if self.shape[-1] % 2 == 0:
                weight[-1] = 1
            self.cached = k, valid, weight
        return self.cached

    def _map(self, x, adjoint):
        xp = self.xp
        out = xp.empty(self.size if adjoint else (6,)+self.hshape, dtype=x.dtype)
        if xp is not np:
            key = x.dtype.str
            if key not in _COORDINATE_KERNELS:
                name = 'float' if x.dtype == xp.complex64 else 'double'
                code = _COORDINATE_CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
                _COORDINATE_KERNELS[key] = xp.RawKernel(code, 'coordinates')
            d = x.real.dtype.type
            _COORDINATE_KERNELS[key](((self.count+255)//256,), (256,),
                (x, out, *(np.int64(n) for n in self.shape), np.int64(self.hshape[-1]),
                 *(d(1/L) for L in self.P.lengths), d(self.norm),
                 np.int32(sum(1<<i for i in self.P.free)), np.int32(adjoint)))
            return out
        k, valid, weight = self._cpu_geometry()
        r2 = np.sqrt(2.)
        if adjoint:
            v = np.asarray([k[0]*x[0]+(k[1]*x[5]+k[2]*x[4])/r2,
                            k[1]*x[1]+(k[0]*x[5]+k[2]*x[3])/r2,
                            k[2]*x[2]+(k[0]*x[4]+k[1]*x[3])/r2])
            v = r2*v+(1-r2)*k*np.sum(k*v, axis=0)
            out[:3*self.count] = (-1j*v*valid*self.norm*weight).ravel()
            out[3*self.count:] = x[self.P.free, 0, 0, 0].real*self.norm
        else:
            v = x[:3*self.count].reshape((3,)+self.hshape)
            v = r2*v+(1-r2)*k*np.sum(k*v, axis=0)
            for i in range(3):
                out[i] = k[i]*v[i]
            for a, i, j in ((3, 1, 2), (4, 0, 2), (5, 0, 1)):
                out[a] = (k[i]*v[j]+k[j]*v[i])/r2
            out *= 1j*valid*self.norm/weight
            out[:, 0, 0, 0] = 0
            out[self.P.free, 0, 0, 0] = x[3*self.count:].real*self.norm
        return out

    def adjoint(self, field):
        with self.P.timers.measure('coordinates_adjoint'):
            h = core_fft.rfftn_cached(field, self.shape, (1, 2, 3))
            return self._map(h, True)

    def strain(self, coordinates):
        with self.P.timers.measure('coordinates_strain'):
            h = self._map(coordinates, False)
            return core_fft.irfftn_unscaled_cached(h, self.shape, (1, 2, 3))

    def precondition(self, value, reference):
        xp = self.xp
        with self.P.timers.measure('preconditioner'):
            out = xp.empty_like(value)
            if xp is np:
                k, valid, _ = self._cpu_geometry()
                v = value[:3*self.count].reshape((3,)+self.hshape)
                out[:3*self.count] = (reference.a*v+(reference.b-reference.a)*k*np.sum(k*v, axis=0)).ravel()
                free = self.P.free
                out[3*self.count:] = reference.mean[np.ix_(free, free)] @ value[3*self.count:]
                return out
            key = value.dtype.str
            if key not in _PRECONDITIONERS:
                name = 'float' if value.dtype == xp.complex64 else 'double'
                code = _PRECONDITION_CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
                _PRECONDITIONERS[key] = xp.RawKernel(code, 'reference')
            d = value.real.dtype.type
            _PRECONDITIONERS[key](((self.count+255)//256,), (256,),
                (value, out, np.int64(self.shape[0]), np.int64(self.shape[1]), np.int64(self.hshape[-1]),
                 *(d(1/L) for L in self.P.lengths), d(reference.a), d(reference.b), reference.mean,
                 np.int32(sum(1<<i for i in self.P.free))))
            return out


class Timings:
    def __init__(self, xp, enabled):
        self.xp, self.enabled, self.seconds, self.calls = xp, enabled, {}, {}
        self.sampled_device_peak = 0

    def sync(self):
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()

    @contextmanager
    def measure(self, name):
        if self.enabled:
            self.sync()
            if self.xp is not np:
                free, total = self.xp.cuda.runtime.memGetInfo()
                self.sampled_device_peak = max(self.sampled_device_peak, total-free)
        start = time.perf_counter()
        yield
        if self.enabled:
            self.sync()
        self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter() - start
        self.calls[name] = self.calls.get(name, 0) + 1


class MixedProjector:
    """Orthogonal G + mean-free-components; even-grid Nyquist planes are zero."""
    def __init__(self, shape, lengths, strain_control, xp=np, timers=None, spectral_fusion=False):
        self.shape, self.lengths, self.xp = tuple(shape), tuple(lengths), xp
        self.free = np.flatnonzero(~np.asarray(strain_control, dtype=bool)).tolist()
        self.timers = timers or Timings(xp, False)
        self.descriptor = SimpleNamespace(N=self.shape, Y=self.lengths, fft_form='r',
                                          multype='elasticity_hg1_direct')
        self.frequencies = None
        self.spectral_fusion = bool(spectral_fusion and xp is not np)

    def _formula(self, h):
        xp = self.xp
        if self.frequencies is None:
            f = [np.fft.fftfreq(n)*n/L for n, L in zip(self.shape, self.lengths)]
            f[-1] = np.fft.rfftfreq(self.shape[-1])*self.shape[-1]/self.lengths[-1]
            self.frequencies = [xp.asarray(v.reshape(tuple(-1 if j == i else 1 for j in range(3))),
                                           dtype=h.real.dtype) for i, v in enumerate(f)]
        x, y, z = self.frequencies
        q2 = x*x + y*y + z*z
        inv = 1 / xp.where(q2 > 0, q2, 1)
        r2 = np.sqrt(2.)
        v = (h[0]*x + h[5]*y/r2 + h[4]*z/r2,
             h[5]*x/r2 + h[1]*y + h[3]*z/r2,
             h[4]*x/r2 + h[3]*y/r2 + h[2]*z)
        c = (x*v[0] + y*v[1] + z*v[2])*inv*inv
        out = xp.empty_like(h)
        for i, k in enumerate((x, y, z)):
            out[i] = 2*k*v[i]*inv - k*k*c
        for a, i, j in ((3, 1, 2), (4, 0, 2), (5, 0, 1)):
            ki, kj = (x, y, z)[i], (x, y, z)[j]
            out[a] = r2*((ki*v[j]+kj*v[i])*inv-ki*kj*c)
        for axis, n in enumerate(self.shape):
            if n % 2 == 0:
                sl = [slice(None)]*4
                sl[axis+1] = n//2
                out[tuple(sl)] = 0
        out[:, 0, 0, 0] = 0
        return out

    def __call__(self, value):
        with self.timers.measure('fft_projection'):
            h = core_fft.rfftn_cached(value, self.shape, (1, 2, 3))
            if self.spectral_fusion:
                g = project_normalized(self.xp, h, self.shape, self.lengths, self.free)
                return core_fft.irfftn_unscaled_cached(g, self.shape, (1, 2, 3))
            g = elasticity_direct_array(self.descriptor, h) if self.xp is not np else None
            if g is None:
                g = self._formula(h)
            for i in self.free:
                g[i, 0, 0, 0] = h[i, 0, 0, 0]
            return core_fft.irfftn_cached(g, self.shape, (1, 2, 3), overwrite_x=True)

