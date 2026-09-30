"""Experimental additive block compliance, never imported by production.

B = B0 + weight Q* R* D R Q.  R is normalized block restriction, with
optional matrix/active-matrix support.  D is inverse mean tangent (Voigt)
or mean inverse tangent (Reuss).  Restrict/prolong implement the equivalent
average/constant pair, including partial and empty blocks.  The true fine
operator is untouched.  Fixed geometry and SPD blocks during each PCG.
"""
from contextlib import contextmanager
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from profile_g09_fft import cp
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction


_CODE = r'''
__device__ long long voxel(int block, int t, int edge, int n0,int n1,int n2) {
 int g1=(n1+edge-1)/edge,g2=(n2+edge-1)/edge;
 int x=(block/(g1*g2))*edge+t/(edge*edge);
 int y=((block/g2)%g1)*edge+(t/edge)%edge,z=(block%g2)*edge+t%edge;
 return x<n0&&y<n1&&z<n2 ? ((long long)x*n1+y)*n2+z : -1;
}
__device__ bool included(long long k,const int* ids,const float* dp,int support){
 return support==0 || (ids[k]==0 && (support==1 || dp[k]>0));
}
__device__ double reduce(double v){
 for(int j=16;j;j>>=1)v+=__shfl_down_sync(0xffffffff,v,j);return v;
}
extern "C" __global__ void assemble(const int* ids,const float* table,
 const double* inverse_table,const float* a,const float* b,const float* u,
 const float* dp,double* blocks,int* counts,int* bad,
 int n0,int n1,int n2,int edge,int support,int reuss,double K,double shift){
 long long N=(long long)n0*n1*n2;
 double sums[21]={0}; int count=0;
 for(int t=threadIdx.x;t<edge*edge*edge;t+=blockDim.x){
  long long k=voxel(blockIdx.x,t,edge,n0,n1,n2);
  if(k<0||!included(k,ids,dp,support))continue;
  count++;int id=ids[k]; double dir[6],rank=0,diag=0,vol=0;
  if(id==0){
   if(!reuss){
    diag=(double)a[k]+shift;vol=K-(double)a[k]/3;rank=b[k];
    for(int i=0;i<6;i++)dir[i]=u[i*N+k];
   }else{
    // Exact Sherman--Morrison even when stored n has small trace/norm error.
    double d=(double)a[k]+shift, bulk=3*K+shift,tr=0,ns=0;
    for(int i=0;i<3;i++)tr+=u[i*N+k];
    diag=1/d;vol=(1/bulk-diag)/3;
    for(int i=0;i<6;i++){
     dir[i]=diag*(double)u[i*N+k]+(i<3?vol*tr:0);
     ns+=(double)u[i*N+k]*dir[i];
    }
    double den=1+(double)b[k]*ns;
    if(!(d>0&&bulk>0&&den>0)){atomicExch(bad,1);den=1;}
    rank=-(double)b[k]/den;
   }
  }
  int h=0;
  for(int i=0;i<6;i++)for(int j=0;j<=i;j++){
   double v=id ? (reuss?inverse_table[id*36+i*6+j]:
                          (double)table[id*36+i*6+j]+(i==j?shift:0))
               : (i==j?diag:0)+(i<3&&j<3?vol:0)+rank*dir[i]*dir[j];
   sums[h++]+=v;
  }
 }
 __shared__ double partial[22*8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int h=0;h<21;h++){double v=reduce(sums[h]);if(!lane)partial[h*8+warp]=v;}
 double c=reduce((double)count);if(!lane)partial[21*8+warp]=c;
 __syncthreads();
 if(threadIdx.x==0){
  double total=0;for(int w=0;w<8;w++)total+=partial[21*8+w];
  counts[blockIdx.x]=(int)total;int h=0;
  for(int i=0;i<6;i++)for(int j=0;j<=i;j++){
   double v=0;for(int w=0;w<8;w++)v+=partial[h*8+w];h++;
   v=total?v/total:(i==j?1:0);
   blocks[blockIdx.x*36+i*6+j]=blocks[blockIdx.x*36+j*6+i]=v;
  }
 }
}
extern "C" __global__ void invert(const double* matrices,float* out,int* bad,int n,int inverse){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
 const double* a=matrices+36*k;double L[36]={0},inv[36]={0};
 // Cholesky is also the SPD audit for both averaging choices.
 for(int i=0;i<6;i++)for(int j=0;j<=i;j++){
  double v=a[i*6+j];for(int l=0;l<j;l++)v-=L[i*6+l]*L[j*6+l];
  if(i==j){if(!(v>0&&isfinite(v))){atomicExch(bad,1);return;}L[i*6+j]=sqrt(v);}
  else L[i*6+j]=v/L[j*6+j];
 }
 if(inverse){
  for(int j=0;j<6;j++){
   double y[6],x[6];
   for(int i=0;i<6;i++){double v=i==j?1:0;for(int l=0;l<i;l++)v-=L[i*6+l]*y[l];y[i]=v/L[i*6+i];}
   for(int i=5;i>=0;i--){double v=y[i];for(int l=i+1;l<6;l++)v-=L[l*6+i]*x[l];x[i]=v/L[i*6+i];}
   for(int i=0;i<6;i++)inv[i*6+j]=x[i];
  }
 }
 for(int i=0;i<6;i++)for(int j=0;j<6;j++)
  out[k*36+i*6+j]=(float)(inverse?(inv[i*6+j]+inv[j*6+i])/2:a[i*6+j]);
}
extern "C" __global__ void restrict_apply(const float* v,const float* D,
 const int* counts,const int* ids,const float* dp,float* coarse,
 int n0,int n1,int n2,int edge,int support){
 long long N=(long long)n0*n1*n2;double sums[6]={0};
 for(int t=threadIdx.x;t<edge*edge*edge;t+=blockDim.x){
  long long k=voxel(blockIdx.x,t,edge,n0,n1,n2);
  if(k<0||!included(k,ids,dp,support))continue;
  for(int i=0;i<6;i++)sums[i]+=v[i*N+k];
 }
 __shared__ double partial[6*8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int i=0;i<6;i++){double s=reduce(sums[i]);if(!lane)partial[i*8+warp]=s;}
 __syncthreads();
 if(threadIdx.x==0){
  double r[6];int count=counts[blockIdx.x];
  for(int i=0;i<6;i++){r[i]=0;for(int w=0;w<8;w++)r[i]+=partial[i*8+w];r[i]=count?r[i]/count:0;}
  for(int i=0;i<6;i++){double s=0;for(int j=0;j<6;j++)s+=D[blockIdx.x*36+i*6+j]*r[j];coarse[blockIdx.x*6+i]=s;}
 }
}
extern "C" __global__ void prolong(const float* coarse,const int* ids,
 const float* dp,float* out,int n0,int n1,int n2,int edge,int support){
 long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x,N=(long long)n0*n1*n2;
 if(k>=N)return;int x=k/(n1*n2),y=(k/n2)%n1,z=k%n2;
 int g1=(n1+edge-1)/edge,g2=(n2+edge-1)/edge;
 int block=((x/edge)*g1+y/edge)*g2+z/edge;
 bool active=included(k,ids,dp,support);
 for(int i=0;i<6;i++)out[i*N+k]=active?coarse[block*6+i]:0;
}
'''
_MODULE = None
_DOT = None


def device_dot(x, y):
    global _DOT
    if _DOT is None:
        _DOT = cp.ReductionKernel('complex64 x, complex64 y', 'float64 out',
            '(double)x.real()*y.real()+(double)x.imag()*y.imag()',
            'a+b', 'out=a', '0', 'spatial_pcg_dot', reduce_type='double')
    return _DOT(x, y)


class BlockCompliance:
    def __init__(self, Q, gpu, response, *, edge=8, support='matrix',
                 averaging='voigt', weight=1., shift=0.):
        global _MODULE
        if (Q.xp is not cp or gpu.dtype != np.float32 or Q.shape != gpu.shape
                or not isinstance(edge, int) or edge < 2
                or support not in ('all', 'matrix', 'plastic')
                or averaging not in ('voigt', 'reuss')
                or not np.isfinite([weight, shift]).all() or weight < 0 or shift < 0):
            raise ValueError('Unsupported block compliance configuration')
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_CODE)
        self.Q, self.gpu, self.response = Q, gpu, response
        self.edge, self.weight = edge, weight
        self.support = ('all', 'matrix', 'plastic').index(support)
        self.grid = tuple((n+edge-1)//edge for n in Q.shape)
        self.nblocks = int(np.prod(self.grid))
        self.geometry = (*(np.int32(n) for n in Q.shape), np.int32(edge), np.int32(self.support))
        self.counts = cp.empty(self.nblocks, dtype=cp.int32)
        self.blocks = cp.empty((self.nblocks, 6, 6), dtype=cp.float32)
        raw = cp.empty((self.nblocks, 6, 6), dtype=cp.float64)
        bad = cp.zeros(1, dtype=cp.int32)
        table = gpu.table.astype(cp.float64)+shift*cp.eye(6, dtype=cp.float64)
        inverse_table = cp.linalg.inv(table) if averaging == 'reuss' else table
        _MODULE.get_function('assemble')((self.nblocks,), (256,),
            (gpu.ids, gpu.table, inverse_table, response.a, response.b,
             response.direction, response.dp, raw, self.counts, bad, *self.geometry,
             np.int32(averaging == 'reuss'), np.float64(gpu.material.K), np.float64(shift)))
        _MODULE.get_function('invert')(((self.nblocks+127)//128,), (128,),
            (raw, self.blocks, bad, np.int32(self.nblocks), np.int32(averaging == 'voigt')))
        if int(bad[0]):
            raise ValueError('Non-SPD or nonfinite block tangent/compliance')
        self.coarse = cp.empty((self.nblocks, 6), dtype=cp.float32)
        self.restrict_kernel = _MODULE.get_function('restrict_apply')
        self.prolong_kernel = _MODULE.get_function('prolong')

    def local_action(self, field):
        """R* D R, owning output; response and input are never modified."""
        g, r = self.gpu, self.response
        self.restrict_kernel((self.nblocks,), (256,),
            (field, self.blocks, self.counts, g.ids, r.dp, self.coarse, *self.geometry))
        out = cp.empty_like(field)
        self.prolong_kernel(((g.n+255)//256,), (256,),
            (self.coarse, g.ids, r.dp, out, *self.geometry))
        return out

    def correction(self, residual):
        if self.weight == 0:
            return cp.zeros_like(residual)
        field = self.Q.strain(residual)
        out = self.Q.adjoint(self.local_action(field))
        out *= np.float32(self.weight)
        return out


class SpatialPCG(FusedCoordinatePCG):
    """Retain production PCG/true-residual audit and fused baseline updates."""
    def __init__(self, coordinates, spatial, reduction='warp'):
        if reduction not in ('warp', 'generic'):
            raise ValueError('Unknown reduction strategy')
        super().__init__(coordinates)
        self.spatial = spatial
        self.correction_value = None
        self.reduction_mode = reduction
        # The reduction constructor needs dtype/size metadata, not a field.
        descriptor = SimpleNamespace(dtype=np.dtype('float32'), size=2*coordinates.size)
        self.reduction = CupyWarpCGReduction(SimpleNamespace(N=coordinates.shape,
            val=descriptor), max_blocks=16384)

    def update(self, p, ap, alpha, x, r, reference):
        self.update_kernel((self.blocks,), (256,),
            (p, ap, np.float32(alpha), x, r, self.partial,
             *self.geometry, *self._reference_args(reference)))
        self.correction_value = self.spatial.correction(r)
        energies = cp.sum(self.partial, axis=1, dtype=cp.float64)
        if self.reduction_mode == 'generic':
            energies[1] += device_dot(r, self.correction_value)
        else:
            reduction = self.reduction
            reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                (r.view(cp.float32), self.correction_value.view(cp.float32),
                 reduction.partial, np.int64(r.size*2)))
            energies[1] += cp.sum(reduction.partial, dtype=cp.float64)
        return cp.asnumpy(energies)

    def direction(self, r, beta, p, reference):
        super().direction(r, beta, p, reference)
        p += self.correction_value

    @contextmanager
    def installed(self):
        original = self.Q.precondition
        def precondition(value, reference):
            return original(value, reference)+self.spatial.correction(value)
        with patch.object(self.Q, 'precondition', precondition):
            yield self
