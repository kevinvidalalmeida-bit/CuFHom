"""Research physical-metric minimum-residual smoothing of symmetric CG.

M(k)=R(k)^-2. Norms use the ORIGINAL strain-equation metric; weighted
smoothing computes cross products instead of assuming M-orthogonal CG residuals.
"""
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.nonlinear_solver import NonlinearConvergenceError,_cg_updates

_CODE=r'''
__device__ void frequency(long long s,long long n0,long long n1,long long n2,long long nz,
 double iy0,double iy1,double iy2,double* k,bool& valid){
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 valid=s!=0&&!(n0%2==0&&i0==n0/2)&&!(n1%2==0&&i1==n1/2)&&!(n2%2==0&&i2==n2/2);
 k[0]=(i0<=n0/2?i0:i0-n0)*iy0;k[1]=(i1<=n1/2?i1:i1-n1)*iy1;k[2]=i2*iy2;
 double t=valid?1/sqrt(k[0]*k[0]+k[1]*k[1]+k[2]*k[2]):0;
 for(int i=0;i<3;i++)k[i]*=t;}
__device__ void sums(double* v,double* out){
 __shared__ double warps[32];int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int j=0;j<4;j++){for(int d=16;d;d>>=1)v[j]+=__shfl_down_sync(0xffffffff,v[j],d);
 if(!lane)warps[8*j+warp]=v[j];}__syncthreads();if(!warp){
 for(int j=0;j<4;j++){double t=lane<8?warps[8*j+lane]:0;
 for(int d=16;d;d>>=1)t+=__shfl_down_sync(0xffffffff,t,d);
 if(!lane)out[j*gridDim.x+blockIdx.x]=t;}}}
__device__ void accumulate(float2* r,float2* sr,double* k,double t,double d,double* v){
 for(int part=0;part<2;part++){
 double a=0,b=0,ra=0,rb=0,c=0;
 for(int j=0;j<3;j++){double x=part?r[j].y:r[j].x,y=part?sr[j].y:sr[j].x;
 a+=x*x;b+=y*y;c+=x*y;ra+=k[j]*x;rb+=k[j]*y;}
 v[0]+=a;v[1]+=t*a+d*ra*ra;v[2]+=t*c+d*ra*rb;v[3]+=t*b+d*rb*rb;}}
extern "C" __global__ void metric(const float2* r,const float2* sr,double* out,
 long long n0,long long n1,long long n2,long long nz,double iy0,double iy1,double iy2,double t,double d){
 long long count=n0*n1*nz;double v[4]={0};
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;s+=(long long)gridDim.x*blockDim.x){
 double k[3];bool valid;frequency(s,n0,n1,n2,nz,iy0,iy1,iy2,k,valid);if(!valid)continue;
 float2 a[3],b[3];for(int j=0;j<3;j++){a[j]=r[j*count+s];b[j]=sr[j*count+s];}
 accumulate(a,b,k,t,d,v);}sums(v,out);}
extern "C" __global__ void update(const float2* p,const float2* ap,float alpha,float2* x,float2* r,
 float2* y,float2* sr,float weight,int smooth,double* out,
 long long n0,long long n1,long long n2,long long nz,double iy0,double iy1,double iy2,double t,double d){
 long long count=n0*n1*nz;double v[4]={0};
 for(long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;s<count;s+=(long long)gridDim.x*blockDim.x){
 double k[3];bool valid;frequency(s,n0,n1,n2,nz,iy0,iy1,iy2,k,valid);
 float2 a[3],b[3];for(int j=0;j<3;j++){long long index=j*count+s;
 float2 xi=x[index],ri=r[index],pi=p[index],ai=ap[index];
 if(smooth){float2 yi=y[index],si=sr[index];
 y[index]=make_float2(yi.x+weight*(xi.x-yi.x),yi.y+weight*(xi.y-yi.y));
 b[j]=make_float2(si.x+weight*(ri.x-si.x),si.y+weight*(ri.y-si.y));sr[index]=b[j];}
 x[index]=make_float2(xi.x+alpha*pi.x,xi.y+alpha*pi.y);
 a[j]=make_float2(ri.x-alpha*ai.x,ri.y-alpha*ai.y);r[index]=a[j];
 if(!smooth)b[j]=a[j];}
 if(valid)accumulate(a,b,k,t,d,v);}sums(v,out);}
'''
_MODULE=None;_BLEND=None
class Metric:
 def __init__(self,Q):
  global _MODULE
  if _MODULE is None:_MODULE=cp.RawModule(code=_CODE)
  self.Q=Q;self.blocks=min(16384,(Q.count+255)//256);self.partial=cp.empty((4,self.blocks),cp.float64)
  self.args=(*(np.int64(n) for n in Q.shape),np.int64(Q.hshape[-1]),
   *(np.float64(1/L) for L in Q.P.lengths),np.float64(1/Q.transverse**2),
   np.float64(1/Q.longitudinal**2-1/Q.transverse**2))
 def norm(self,r,s=None):
  _MODULE.get_function('metric')((self.blocks,),(256,),
   (r,r if s is None else s,self.partial,*self.args))
  return cp.asnumpy(cp.sum(self.partial,axis=1,dtype=cp.float64))
 def update(self,p,ap,alpha,x,r,y,sr,weight):
  _MODULE.get_function('update')((self.blocks,),(256,),
   (p,ap,np.float32(alpha),x,r,x if y is None else y,r if sr is None else sr,
    np.float32(weight),np.int32(y is not None),self.partial,*self.args))
  return cp.asnumpy(cp.sum(self.partial,axis=1,dtype=cp.float64))

def cg(action,rhs,dot,tolerance,maxiter,xp,precondition=None,action_and_dot=None,**kwargs):
 global _BLEND
 if xp is not cp or precondition is not None:raise ValueError('Pilot expects whitened GPU coordinates')
 captures=dict(zip(action.__code__.co_freevars,(c.cell_contents for c in action.__closure__)))
 M=Metric(captures['Q'])
 x=cp.zeros_like(rhs);r=rhs.copy();rr,physical,_,_=M.norm(r);limit=tolerance*tolerance*physical
 if rr==0:return x,0
 p=r.copy();direction=None;y=sr=None;weight=0.
 if _BLEND is None:_BLEND=cp.ElementwiseKernel('T x,T r,float32 w','T y,T s',
    'y += w*(x-y); s += w*(r-s);','cufhom_physical_metric_finish')
 for i in range(maxiter):
  if action_and_dot:ap,pap=action_and_dot(p,direction=direction);direction=None
  else:ap=action(p);pap=dot(p,ap)
  if not np.isfinite([rr,pap]).all() or min(rr,pap)<=0:raise NonlinearConvergenceError('Metric CG positivity')
  new,physical,cross,squared=M.update(p,ap,rr/pap,x,r,y,sr,weight)
  if y is None:
   estimate=physical
   if limit<physical<=100*limit:y=x.copy();sr=r.copy();weight=0.
  else:
   den=squared-2*cross+physical
   weight=float(np.clip((squared-cross)/den,0.,1.)) if den>0 else 0.
   estimate=(1-weight)**2*squared+2*weight*(1-weight)*cross+weight**2*physical
  if estimate<=limit:
   if y is not None:_BLEND(x,r,np.float32(weight),y,sr);x=y
   r=rhs-action(x);true=M.norm(r)
   if true[1]<=limit:return x,i+1
   rr=true[0];p=r.copy();direction=None;y=sr=None;weight=0.
   continue
  if action_and_dot:direction=(r,new/rr)
  else:_cg_updates()[1](r,np.float64(new/rr),p)
  rr=new
 raise NonlinearConvergenceError('Metric CG failed physical-metric residual')
@contextmanager
def trial():
 with patch.object(lc,'fused_cg',cg):yield
