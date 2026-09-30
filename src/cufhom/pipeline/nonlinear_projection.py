"""Fused Fourier compatibility, mixed mean modes and inverse-FFT normalization.

No change to the spectral discretization: even-grid Nyquist planes stay zero.
Each thread owns all six Mandel components of one frequency. The projection
uses sigma*k and k.(sigma*k), not a stored or expanded 6x6 matrix.
"""
import numpy as np

_KERNELS = {}
_CODE = r'''
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
    kernel = _KERNELS.get(dtype.str)
    if kernel is None:
        name = 'float' if real == np.float32 else 'double'
        code = _CODE.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
        kernel = cp.RawKernel(code, 'project')
        _KERNELS[dtype.str] = kernel
    out = cp.empty_like(spectrum)
    count = spectrum.size//6
    kernel(((count+255)//256,), (256,),
        (spectrum, out, *(np.int64(n) for n in shape), np.int64(spectrum.shape[-1]),
         *(real(1/L) for L in lengths), real(1/np.prod(shape)),
         np.int32(sum(1 << i for i in free))))
    return out
