"""Experimental isometric equilibrium-stress coordinates, including Nyquist.

The dual complement of the current primal space must retain ALL six stress
components on excluded primal Nyquist planes. Mean stresses are prescribed.
No alternative discretization is introduced.
"""
import numpy as np
from pipeline.nonlinear_solver import cp
from ffthompy.tensors import fft

_KERNEL = None
_CODE = r'''
extern "C" __global__ void dual_map(const float2* x,float2* y,const int* extra,
 long long n0,long long n1,long long n2,long long nz,
 float iy0,float iy1,float iy2,float norm,float rt,float rp,float rv,int adjoint) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;
 if(s==0){for(int c=0;c<(adjoint?3:6);c++)y[c*count]=make_float2(0,0);return;}
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 int e=extra[s];
 float weight=(i2==0||(n2%2==0&&i2==n2/2))?1:1.4142135623730951f;
 float scale=adjoint?norm*weight:norm/weight;
 float B[18];
 if(e<0) {
   float n[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
   if(n[2]==0&&(n[1]<0||(n[1]==0&&n[0]<0)))for(int a=0;a<3;a++)n[a]=-n[a];
   float inv=rsqrtf(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);for(int a=0;a<3;a++)n[a]*=inv;
   float t[3];
   if(fabsf(n[2])<.9f){t[0]=n[1];t[1]=-n[0];t[2]=0;}
   else {t[0]=-n[2];t[1]=0;t[2]=n[0];}
   inv=rsqrtf(t[0]*t[0]+t[1]*t[1]+t[2]*t[2]);for(int a=0;a<3;a++)t[a]*=inv;
   float v[3]={n[1]*t[2]-n[2]*t[1],n[2]*t[0]-n[0]*t[2],n[0]*t[1]-n[1]*t[0]};
   for(int a=0;a<3;a++){B[3*a]=t[a]*t[a];B[3*a+1]=v[a]*v[a];B[3*a+2]=1.4142135623730951f*t[a]*v[a];}
   int first[3]={1,0,0},second[3]={2,2,1};
   for(int a=0;a<3;a++) {
     int b=first[a],c=second[a],row=3*(a+3);
     B[row]=1.4142135623730951f*t[b]*t[c];
     B[row+1]=1.4142135623730951f*v[b]*v[c];
     B[row+2]=t[b]*v[c]+t[c]*v[b];
   }
 }
 for(int part=0;part<2;part++) {
   float z[6]={0,0,0,0,0,0},out[6];
   if(e>=0) {
     for(int c=0;c<6;c++) {
       long long index=(adjoint||c<3)?c*count+s:3*count+3*(long long)e+c-3;
       z[c]=part?x[index].y:x[index].x;
     }
     float volume=(rv-rt)*(z[0]+z[1]+z[2])/3;
     for(int c=0;c<6;c++)out[c]=rt*z[c]+(c<3?volume:0);
   } else {
     if(adjoint)for(int c=0;c<3;c++)for(int a=0;a<6;a++)z[c]+=B[3*a+c]*(part?x[a*count+s].y:x[a*count+s].x);
     else for(int c=0;c<3;c++)z[c]=part?x[c*count+s].y:x[c*count+s].x;
     float plane=(rp-rt)*(z[0]+z[1])/2;
     for(int c=0;c<3;c++)z[c]=rt*z[c]+(c<2?plane:0);
     if(adjoint)for(int c=0;c<3;c++)out[c]=z[c];
     else for(int a=0;a<6;a++){out[a]=0;for(int c=0;c<3;c++)out[a]+=B[3*a+c]*z[c];}
   }
   for(int c=0;c<((!adjoint||e>=0)?6:3);c++) {
     long long index=(!adjoint||c<3)?c*count+s:3*count+3*(long long)e+c-3;
     if(part)y[index].y=scale*out[c];else y[index].x=scale*out[c];
   }
 }
}
'''


class DualCoordinates:
    def __init__(self, shape, lengths, bulk, shear):
        if not np.isfinite([bulk,shear]).all() or min(bulk,shear)<=0:
            raise ValueError('Dual reference must be SPD')
        self.shape,self.lengths=tuple(shape),tuple(lengths)
        self.hshape=self.shape[:-1]+(self.shape[-1]//2+1,)
        self.count=int(np.prod(self.hshape))
        mask=np.zeros(self.hshape,bool)
        for axis,n in enumerate(shape):
            if n%2==0:
                sl=[slice(None)]*3;sl[axis]=n//2;mask[tuple(sl)]=True
        selected=np.flatnonzero(mask.ravel())
        extra=np.full(self.count,-1,np.int32);extra[selected]=np.arange(len(selected),dtype=np.int32)
        self.extra=cp.asarray(extra)
        self.size=3*self.count+3*len(selected)
        self.rt=np.sqrt(2*shear)
        self.rp=1/np.sqrt(1/(6*shear)+2/(9*bulk))
        self.rv=np.sqrt(3*bulk)
        self.norm=1/np.sqrt(np.prod(shape))

    def _map(self,x,adjoint):
        global _KERNEL
        if x.dtype!=cp.complex64:
            raise ValueError('Dual prototype requires complex64')
        if _KERNEL is None:
            _KERNEL=cp.RawKernel(_CODE,'dual_map')
        out=cp.empty(self.size if adjoint else (6,)+self.hshape,cp.complex64)
        _KERNEL(((self.count+127)//128,),(128,),
            (x,out,self.extra,*(np.int64(n) for n in self.shape),np.int64(self.hshape[-1]),
             *(np.float32(1/L) for L in self.lengths),np.float32(self.norm),
             np.float32(self.rt),np.float32(self.rp),np.float32(self.rv),np.int32(adjoint)))
        return out

    def stress(self,x):
        h=self._map(x,False)
        return fft.irfftn_unscaled_cached(h,self.shape,(1,2,3))

    def adjoint(self,x):
        return self._map(fft.rfftn_cached(x,self.shape,(1,2,3)),True)
