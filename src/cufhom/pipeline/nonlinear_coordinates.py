"""Isometric displacement coordinates for the existing spectral strain space.

Q(k)=i B(n)[B(n)*B(n)]^-1/2, n=k/|k|. RFFT coordinates include
sqrt(2) weights for interior last-axis frequencies and 1/sqrt(N) scaling.
Thus real(vdot(z,w)) equals the physical Mandel inner product of Qz,Qw.
Five (or the configured number of) free mean strains are separate coordinates.
This is a change of variables, not frequency truncation or mesh coarsening.
"""
import numpy as np
from cufhom_core.tensors import fft as core_fft

_KERNELS = {}
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
_CODE = r'''
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
            if key not in _KERNELS:
                name = 'float' if x.dtype == xp.complex64 else 'double'
                code = _CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
                _KERNELS[key] = xp.RawKernel(code, 'coordinates')
            d = x.real.dtype.type
            _KERNELS[key](((self.count+255)//256,), (256,),
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
