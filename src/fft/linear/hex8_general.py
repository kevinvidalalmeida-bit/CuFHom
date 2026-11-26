"""Indexed, anisotropic, fully integrated periodic HEX8 on CPU or CUDA.

Spatial axes and lengths are X,Y,Z. Displacements are X,Y,Z; symmetric
tensors use orthonormal Mandel order 11,22,33,23,13,12. All nodal modes except
translations are retained, including even-grid Nyquist planes. This is a
separate HEX8 discretization, not a modified GaNi derivative.

CUDA uses one deterministic nodal gather, indexed element matrices, a stable
homogeneous FFT inverse, fused PCG vector traversals, and FP64 scalar products.
No global eight-Gauss-point strain/stress arrays or force atomics are needed.
"""

from itertools import product

import numpy as np

from .common import (
    FreshProgress,
    LinearSolveError,
    WorkingPrecisionError,
    device_label_counts,
    energy_report,
    energy_tolerance,
    working_material_check,
)

try:
    import cupy as cp
except ImportError:
    cp = None

_PAIRS = tuple((i, j) for i in range(6) for j in range(i, 6))
_NODES = np.array(list(product((0, 1), repeat=3)), dtype=np.int32)


def _material_table(stiffness, dtype):
    c = np.asarray(stiffness)
    if np.iscomplexobj(c):
        raise ValueError("Mandel stiffness matrices must be real")
    c = np.asarray(c, dtype=np.float64)
    if c.ndim == 2 and c.shape[1] == 21:
        full = np.zeros((len(c), 6, 6))
        for p, (i, j) in enumerate(_PAIRS):
            full[:, i, j] = full[:, j, i] = c[:, p]
        c = full
    if c.ndim != 3 or c.shape[1:] != (6, 6) or not len(c) or not np.isfinite(c).all():
        raise ValueError("stiffness must be a finite (P,6,6) or (P,21) Mandel table")
    scale = np.max(np.abs(c), axis=(1, 2))
    if np.any(np.max(np.abs(c - c.transpose(0, 2, 1)), axis=(1, 2)) > 1e-7 * scale):
        raise ValueError("Mandel stiffness matrices must be symmetric")
    with np.errstate(over="ignore", invalid="ignore"):
        c = ((c + c.transpose(0, 2, 1)) / 2).astype(dtype).astype(np.float64)
    if not np.isfinite(c).all():
        raise ValueError(
            "Mandel stiffness matrices must remain finite in the requested precision"
        )
    eigenvalues = np.linalg.eigvalsh(c)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues[:, 0] <= 0):
        raise ValueError(
            "Every stiffness matrix must remain positive definite in the requested precision"
        )
    return np.ascontiguousarray(c)


def _solver_parameters(rtol, atol, maxiter):
    message = "Nonnegative finite scalar tolerances, one positive tolerance, and a positive integer budget required"
    values = [np.asarray(value) for value in (rtol, atol, maxiter)]
    if any(value.ndim != 0 or value.dtype.kind not in "iuf" for value in values):
        raise ValueError(message)
    rtol, atol, budget = (value.item() for value in values)
    if (
        not all(np.isfinite(value) for value in (rtol, atol, budget))
        or rtol < 0
        or atol < 0
        or (rtol == 0 and atol == 0)
        or budget < 1
        or int(budget) != budget
    ):
        raise ValueError(message)
    return rtol, atol, int(budget)


def hex8_shape_gradients(shape, lengths):
    """Return B[g, Mandel, node, displacement] for 2x2x2 Gauss integration."""
    h = np.asarray(lengths) / np.asarray(shape)
    points = (1 + np.array([-1.0, 1.0]) / np.sqrt(3)) / 2
    b = np.zeros((8, 6, 8, 3))
    for g, t in enumerate(product(points, repeat=3)):
        for a, corner in enumerate(_NODES):
            d = [
                (2 * corner[i] - 1)
                / h[i]
                * np.prod([t[j] if corner[j] else 1 - t[j] for j in range(3) if j != i])
                for i in range(3)
            ]
            x, y, z = d
            b[g, :, a] = [
                [x, 0, 0],
                [0, y, 0],
                [0, 0, z],
                [0, z / np.sqrt(2), y / np.sqrt(2)],
                [z / np.sqrt(2), 0, x / np.sqrt(2)],
                [y / np.sqrt(2), x / np.sqrt(2), 0],
            ]
    return b


def _check_working_elements(ke, h, dtype):
    """Screen the actual rounded difference-form element on its strain space.

    Six isolated-element rigid modes are removed only for this diagnostic.
    Rounding may couple rotations to strains: this is not a proof of global
    SPD. PCG still checks every curvature and the final original residual.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        blocks = ke.astype(dtype).astype(np.float64).transpose(4, 0, 1, 2, 3).copy()
    if not np.isfinite(blocks).all():
        raise WorkingPrecisionError(
            "Working element table is nonfinite; use working=float64"
        )
    for a in range(8):
        blocks[:, a, a] = 0.0
        blocks[:, a, a] = -blocks[:, a].sum(axis=1)
    matrices = blocks.transpose(0, 1, 3, 2, 4).reshape(-1, 24, 24)
    xyz = _NODES * np.asarray(h)
    rigid = np.empty((24, 6))
    for a, point in enumerate(xyz):
        rigid[3 * a : 3 * a + 3, :3] = np.eye(3)
        rigid[3 * a : 3 * a + 3, 3:] = np.cross(np.eye(3), point).T
    basis = np.linalg.qr(rigid, mode="complete")[0][:, 6:]
    reduced = basis.T @ matrices @ basis
    reduced = (reduced + reduced.transpose(0, 2, 1)) / 2
    values = np.linalg.eigvalsh(reduced)
    if np.any(
        values[:, 0] <= 32 * np.finfo(float).eps * np.max(np.abs(values), axis=1)
    ):
        raise WorkingPrecisionError(
            "Working element lost positive definiteness on the strain space; use working=float64"
        )
    return dict(
        rounded_element_strain_space_checked=True,
        minimum_strain_eigenvalue=float(values[:, 0].min()),
        global_operator_certified=False,
    )


_CUDA_REDUCE = r"""
__device__ __forceinline__ double warp_sum(double x){for(int o=16;o;o>>=1)x+=__shfl_down_sync(0xffffffff,x,o);return x;}
__device__ __forceinline__ double warp_max(double x){for(int o=16;o;o>>=1)x=fmax(x,__shfl_down_sync(0xffffffff,x,o));return x;}
// Fixed-order block reductions over the linear thread id (block size a
// multiple of 32, at most 1024); linear thread 0 holds the result.
// Nonfinite magnitudes enter maxima as +inf, never as a NaN skip.
__device__ __forceinline__ int linear_thread(){return threadIdx.x+blockDim.x*(threadIdx.y+blockDim.y*threadIdx.z);}
__device__ double block_sum(double x){
 __shared__ double w[32];const int t=linear_thread(),warps=(blockDim.x*blockDim.y*blockDim.z)>>5;
 __syncthreads();x=warp_sum(x);
 if((t&31)==0)w[t>>5]=x;__syncthreads();
 x=t<warps?w[t]:0.;
 return t<32?warp_sum(x):0.;
}
__device__ double block_max(double x){
 __shared__ double w[32];const int t=linear_thread(),warps=(blockDim.x*blockDim.y*blockDim.z)>>5;
 __syncthreads();x=warp_max(x);
 if((t&31)==0)w[t>>5]=x;__syncthreads();
 x=t<warps?w[t]:0.;
 return t<32?warp_max(x):0.;
}
__device__ __forceinline__ double magnitude(double v){v=fabs(v);return v==v?v:__longlong_as_double(0x7ff0000000000000LL);}
"""

# Action kernels share one source per node class. MODE 0 writes K u, MODE 1
# also writes block partials of u.Ku and returns once the device PCG state is
# final, MODE 2 writes the fresh residual f - K u (OUT precision) with block
# maxima of |f - K u|; f is evaluated inline from the force table, so neither
# the load nor K u is stored. REAL is the arithmetic, XT the input precision.
_CUDA_INTERFACE = r"""
extern "C" __global__ void hex8_interface(const int* __restrict__ index,const REAL* __restrict__ ke,
 const long long* __restrict__ ids,const XT* __restrict__ u,OUT* __restrict__ out,double* __restrict__ partial,
 const int* __restrict__ state,const double* __restrict__ force,const double* __restrict__ macro,
 long long n,long long count,int ny,int nz,int nx,int phases){
#if MODE==1
 if(state[0])return;
#endif
 long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 double d=0;
 if(t<count){
 long long i=ids[t];
 int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
 REAL u0=TOREAL(u[i]),u1=TOREAL(u[n+i]),u2=TOREAL(u[2*n+i]),v0=0,v1=0,v2=0;
 #pragma unroll
 for(int a=0;a<8;a++){
  int ax=(a>>2)&1,ay=(a>>1)&1,az=a&1;
  int cx=x-ax;if(cx<0)cx+=nx;
  int cy=y-ay;if(cy<0)cy+=ny;
  int cz=z-az;if(cz<0)cz+=nz;
  int phase=index[((long long)cx*ny+cy)*nz+cz];
  #pragma unroll
  for(int b=0;b<8;b++){
   if(b==a)continue;
   int xx=x+((b>>2)&1)-ax;if(xx<0)xx+=nx;else if(xx>=nx)xx-=nx;
   int yy=y+((b>>1)&1)-ay;if(yy<0)yy+=ny;else if(yy>=ny)yy-=ny;
   int zz=z+(b&1)-az;if(zz<0)zz+=nz;else if(zz>=nz)zz-=nz;
   long long j=((long long)xx*ny+yy)*nz+zz;
   REAL d0=TOREAL(u[j])-u0,d1=TOREAL(u[n+j])-u1,d2=TOREAL(u[2*n+j])-u2;
   const REAL* k=ke+((a*8+b)*9)*(long long)phases+phase;
   v0+=k[0*phases]*d0+k[1*phases]*d1+k[2*phases]*d2;
   v1+=k[3*phases]*d0+k[4*phases]*d1+k[5*phases]*d2;
   v2+=k[6*phases]*d0+k[7*phases]*d1+k[8*phases]*d2;
  }
 }
#if MODE==2
 double f[3]={0,0,0};
 #pragma unroll
 for(int a=0;a<4;a++){
  int ax=(a>>2)&1,ay=(a>>1)&1,az=a&1;
  int x1=(x-ax+nx)%nx,y1=(y-ay+ny)%ny,z1=(z-az+nz)%nz;
  int x2=(x-(1-ax)+nx)%nx,y2=(y-(1-ay)+ny)%ny,z2=(z-(1-az)+nz)%nz;
  int p=index[((long long)x1*ny+y1)*nz+z1];
  int q=index[((long long)x2*ny+y2)*nz+z2];
  if(p==q)continue;
  #pragma unroll
  for(int c=0;c<3;c++)for(int j=0;j<6;j++){
   long long k=((a*3+c)*6+j)*(long long)phases;
   f[c]+=(force[k+p]-force[k+q])*macro[j];
  }
 }
 const double r0=f[0]-TODOUBLE(v0),r1=f[1]-TODOUBLE(v1),r2=f[2]-TODOUBLE(v2);
 out[i]=(OUT)r0;out[n+i]=(OUT)r1;out[2*n+i]=(OUT)r2;
 d=fmax(magnitude(r0),fmax(magnitude(r1),magnitude(r2)));
#else
 out[i]=v0;out[n+i]=v1;out[2*n+i]=v2;
#if MODE==1
#if FAST32
 d=(double)(u0*v0+u1*v1+u2*v2);
#else
 d=(double)u0*v0+(double)u1*v1+(double)u2*v2;
#endif
#endif
#endif
 }
#if MODE==1
 d=block_sum(d);if(threadIdx.x==0)partial[blockIdx.x]=d;
#elif MODE==2
 d=block_max(d);if(threadIdx.x==0)partial[blockIdx.x]=d;
#endif
}
"""

# Homogeneous nodes on the regular grid: the label test of the eight incident
# cells replaces any stored interior list; interface nodes are left to the
# gather above. The trilinear cell is centrally symmetric, so the homogeneous
# stencil has K_-o = K_o and K_-o = K_o^T: every block is symmetric, and
# +o/-o act on the sum of their differences. Per phase, table holds the 13
# half-space blocks as (00,01,02,11,12,22) padded to 8: 117 FMA per node.
# Interior nodes: a block of 32 (z) x 8 (y) nodes marches along x through
# INTERIOR_DEPTH planes with a rolling window of three displacement planes and
# two label planes (with a one-node periodic halo) in shared memory, so each
# value is read from DRAM once per block instead of up to 27 times.
INTERIOR_DEPTH = 16
_CUDA_INTERIOR = r"""
#if IS_DOUBLE==1
#define LOAD8(c,ptr) {const double2* q=(const double2*)(ptr);_Pragma("unroll")for(int w=0;w<4;w++){double2 e=__ldg(q+w);c[2*w]=e.x;c[2*w+1]=e.y;}}
#elif IS_DOUBLE==2
#define LOAD8(c,ptr) {const df4* q=(const df4*)(ptr);_Pragma("unroll")for(int w=0;w<2;w++){df4 e=q[w];c[4*w]=e.x;c[4*w+1]=e.y;c[4*w+2]=e.z;c[4*w+3]=e.w;}}
#else
#define LOAD8(c,ptr) {const float4* q=(const float4*)(ptr);_Pragma("unroll")for(int w=0;w<2;w++){float4 e=__ldg(q+w);c[4*w]=e.x;c[4*w+1]=e.y;c[4*w+2]=e.z;c[4*w+3]=e.w;}}
#endif
#define TZ 32
#define TY 8
#define HZ (TZ+2)
#define HP ((TY+2)*HZ)
// Halo tile (y0-1..y0+TY, z0-1..z0+TZ) of plane xp (periodic): displacements
// into slot s of U (when given) and labels into L (when given).
__device__ __forceinline__ void hex8_plane(const int* __restrict__ index,const XT* __restrict__ u,REAL (*U)[3][HP],
 int* L,int s,int xp,int y0,int z0,int nx,int ny,int nz,long long n){
 const int xx=(xp%nx+nx)%nx;
 for(int k=threadIdx.y*TZ+threadIdx.x;k<HP;k+=TZ*TY){
  int yy=y0+k/HZ-1,zz=z0+k%HZ-1;
  // Short axes can wrap more than once in this full cooperative halo tile.
  // Keep the single-wrap path for axes at least as large as the tile.
  if(ny>=TY){yy+=yy<0?ny:0;yy-=yy>=ny?ny:0;}
  else yy=(yy%ny+ny)%ny;
  if(nz>=TZ){zz+=zz<0?nz:0;zz-=zz>=nz?nz:0;}
  else zz=(zz%nz+nz)%nz;
  const long long j=(long long)xx*ny*nz+(long long)yy*nz+zz;
  if(U){U[s][0][k]=TOREAL(u[j]);U[s][1][k]=TOREAL(u[n+j]);U[s][2][k]=TOREAL(u[2*n+j]);}
  if(L)L[k]=index[j];
 }
}
extern "C" __global__ void __launch_bounds__(TZ*TY) hex8_interior(const int* __restrict__ index,
 const REAL* __restrict__ table,const XT* __restrict__ u,OUT* __restrict__ out,double* __restrict__ partial,
 const int* __restrict__ state,int nx,int ny,int nz){
#if MODE==1
 if(state[0])return;
#endif
 __shared__ REAL U[3][3][HP];  // displacement planes x-1, x, x+1 (rolling slots)
 __shared__ int L[2][HP];      // labels of planes x-1 and x (slot = plane parity)
 const int z0=blockIdx.x*TZ,y0=blockIdx.y*TY,x0=blockIdx.z*INTERIOR_DEPTH;
 const int z=z0+threadIdx.x,y=y0+threadIdx.y,c=(threadIdx.y+1)*HZ+threadIdx.x+1;
 const long long n=(long long)nx*ny*nz;
 double d=0;
 hex8_plane(index,u,U,L[(x0+1)&1],0,x0-1,y0,z0,nx,ny,nz,n);
 hex8_plane(index,u,U,L[x0&1],1,x0,y0,z0,nx,ny,nz,n);
 for(int x=x0;x<x0+INTERIOR_DEPTH&&x<nx;x++){
  const int S[3]={(x-x0)%3,(x-x0+1)%3,(x-x0+2)%3};  // slots of planes x-1, x, x+1
  __syncthreads();
  hex8_plane(index,u,U,0,S[2],x+1,y0,z0,nx,ny,nz,n);
  __syncthreads();
  const int* Lx=L[x&1];const int* Lm=L[(x+1)&1];
  const int p=Lx[c];
  if(z<nz&&y<ny&&Lm[c]==p&&Lx[c-HZ]==p&&Lx[c-1]==p&&Lm[c-HZ]==p&&Lm[c-1]==p&&Lx[c-HZ-1]==p&&Lm[c-HZ-1]==p){
   const REAL u0=U[S[1]][0][c],u1=U[S[1]][1][c],u2=U[S[1]][2][c];
   REAL v0=0,v1=0,v2=0;
   const REAL* k=table+(long long)p*104;
   #pragma unroll
   for(int o=0;o<13;o++){
    const int t=o+14,dx=t/9,dy=(t/3)%3,dz=t%3;
    const int jp=c+(dy-1)*HZ+dz-1,jm=c-(dy-1)*HZ-(dz-1);
    const REAL (*A)[HP]=U[S[dx]],(*B)[HP]=U[S[2-dx]];
    const REAL s0=(A[0][jp]-u0)+(B[0][jm]-u0),s1=(A[1][jp]-u1)+(B[1][jm]-u1),s2=(A[2][jp]-u2)+(B[2][jm]-u2);
    REAL kc[8];LOAD8(kc,k+8*o);
    v0+=kc[0]*s0+kc[1]*s1+kc[2]*s2;
    v1+=kc[1]*s0+kc[3]*s1+kc[4]*s2;
    v2+=kc[2]*s0+kc[4]*s1+kc[5]*s2;
   }
   const long long i=(long long)x*ny*nz+(long long)y*nz+z;
#if MODE==2
   out[i]=(OUT)(-TODOUBLE(v0));out[n+i]=(OUT)(-TODOUBLE(v1));out[2*n+i]=(OUT)(-TODOUBLE(v2));
   d=fmax(d,fmax(magnitude(TODOUBLE(v0)),fmax(magnitude(TODOUBLE(v1)),magnitude(TODOUBLE(v2)))));
#else
   out[i]=v0;out[n+i]=v1;out[2*n+i]=v2;
#if MODE==1
#if FAST32
   d+=(double)(u0*v0+u1*v1+u2*v2);
#else
   d+=(double)u0*v0+(double)u1*v1+(double)u2*v2;
#endif
#endif
#endif
  }
  __syncthreads();  // plane x+1 replaces the labels of plane x-1
  if(x+1<x0+INTERIOR_DEPTH&&x+1<nx)hex8_plane(index,u,0,L[(x+1)&1],0,x+1,y0,z0,nx,ny,nz,n);
 }
#if MODE==1
 d=block_sum(d);
#elif MODE==2
 d=block_max(d);
#endif
#if MODE
 if(threadIdx.x==0&&threadIdx.y==0)partial[blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z)]=d;
#endif
}
"""

_CUDA_CLASSIFY = r"""
extern "C" __global__ void hex8_general_classify(const int* index,unsigned char* mask,
 long long n,int ny,int nz,int nx){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 int p=index[i],z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);bool same=true;
 for(int a=1;a<8;a++){
  int xx=(x-((a>>2)&1)+nx)%nx,yy=(y-((a>>1)&1)+ny)%ny,zz=(z-(a&1)+nz)%nz;
  if(index[((long long)xx*ny+yy)*nz+zz]!=p)same=false;
 }
 mask[i]=same;
}
"""

_CUDA_RHS = r"""
extern "C" __global__ void hex8_general_rhs(const int* index,const double* force,
 const double* macro,OUT* out,double* partial,long long n,int ny,int nz,int nx,int phases){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 double m=0;
 if(i<n){
 int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
 double v[3]={0,0,0};
 #pragma unroll
 for(int a=0;a<4;a++){
  int ax=(a>>2)&1,ay=(a>>1)&1,az=a&1;
  int x1=(x-ax+nx)%nx,y1=(y-ay+ny)%ny,z1=(z-az+nz)%nz;
  int x2=(x-(1-ax)+nx)%nx,y2=(y-(1-ay)+ny)%ny,z2=(z-(1-az)+nz)%nz;
  int p=index[((long long)x1*ny+y1)*nz+z1];
  int q=index[((long long)x2*ny+y2)*nz+z2];
  if(p==q)continue;
  #pragma unroll
  for(int c=0;c<3;c++)for(int j=0;j<6;j++){
   long long k=((a*3+c)*6+j)*(long long)phases;
   v[c]+=(force[k+p]-force[k+q])*macro[j];
  }
 }
 out[i]=(OUT)v[0];out[n+i]=(OUT)v[1];out[2*n+i]=(OUT)v[2];
 m=fmax(magnitude(v[0]),fmax(magnitude(v[1]),magnitude(v[2])));
 }
 m=block_max(m);if(threadIdx.x==0)partial[blockIdx.x]=m;
}
"""

# m_j = f(e_j).u over the interface list (the load vanishes elsewhere): the
# volume-averaged Gauss stress of E + B u is Cbar E - sum_j m_j e_j.
_CUDA_MOMENTS = r"""
extern "C" __global__ void hex8_force_moments(const int* index,const double* force,const long long* ids,
 const XT* u,double* partial,long long n,long long count,int ny,int nz,int nx,int phases){
 long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 double m[6]={0,0,0,0,0,0};
 if(t<count){
  long long i=ids[t];
  int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
  const double v[3]={(double)u[i],(double)u[n+i],(double)u[2*n+i]};
  #pragma unroll
  for(int a=0;a<4;a++){
   int ax=(a>>2)&1,ay=(a>>1)&1,az=a&1;
   int x1=(x-ax+nx)%nx,y1=(y-ay+ny)%ny,z1=(z-az+nz)%nz;
   int x2=(x-(1-ax)+nx)%nx,y2=(y-(1-ay)+ny)%ny,z2=(z-(1-az)+nz)%nz;
   int p=index[((long long)x1*ny+y1)*nz+z1];
   int q=index[((long long)x2*ny+y2)*nz+z2];
   if(p==q)continue;
   for(int c=0;c<3;c++)for(int j=0;j<6;j++){
    long long k=((a*3+c)*6+j)*(long long)phases;
    m[j]+=(force[k+p]-force[k+q])*v[c];
   }
  }
 }
 for(int j=0;j<6;j++){const double s=block_sum(m[j]);if(threadIdx.x==0)partial[j*gridDim.x+blockIdx.x]=s;}
}
"""

_CUDA_STRESS = r"""
extern "C" __global__ void hex8_general_stress(const int* index,const REAL* u,
 const double* B,const double* C,const double* macro,double* partial,
 long long n,int ny,int nz,int nx,int phases){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;
 double stress[6]={0,0,0,0,0,0};
 if(i<n){
  int z=i%nz,y=(i/nz)%ny,x=i/((long long)ny*nz);
  double strain[6];for(int s=0;s<6;s++)strain[s]=macro[s];
  #pragma unroll
  for(int a=0;a<8;a++){
   int xx=(x+((a>>2)&1))%nx,yy=(y+((a>>1)&1))%ny,zz=(z+(a&1))%nz;
   long long j=((long long)xx*ny+yy)*nz+zz;
   double v[3]={(double)u[j],(double)u[n+j],(double)u[2*n+j]};
   for(int s=0;s<6;s++)for(int c=0;c<3;c++)strain[s]+=B[(s*8+a)*3+c]*v[c];
  }
  int p=index[i];
  for(int s=0;s<6;s++)for(int t=0;t<6;t++)stress[s]+=C[(s*6+t)*(long long)phases+p]*strain[t];
 }
 extern __shared__ double sums[];
 for(int s=0;s<6;s++)sums[s*blockDim.x+threadIdx.x]=stress[s];
 __syncthreads();
 for(int d=blockDim.x/2;d>0;d/=2){
  if(threadIdx.x<d)for(int s=0;s<6;s++)sums[s*blockDim.x+threadIdx.x]+=sums[s*blockDim.x+threadIdx.x+d];
  __syncthreads();
 }
 if(threadIdx.x==0)for(int s=0;s<6;s++)partial[s*gridDim.x+blockIdx.x]=sums[s*blockDim.x];
}
"""


# Reference inverse K0(k)^-1 evaluated per RFFT frequency in the working
# precision, so no 3x3 table is stored. Per axis, axes holds (A, B, m) with
# A = 2 sin(t)/h, B = A cos(t), m = cos^2(t)+sin^2(t)/3, t = pi k/n; the six
# products g = (G00,G11,G22,G01,G02,G12) of the integrated symbol enter the
# packed symbol K = (K00,K11,K22,K01,K02,K12) linearly through the constant
# 6x6 map M of the reference (host, FP64). K^-1 follows from its Cholesky
# factor in the same packed order; frequency 0 (translations) maps to zero.
_CUDA_SYMBOL = r"""
template<typename T>
__device__ __forceinline__ void reference_inverse(long long i,int ny,int nh,const T* __restrict__ axes,
 int ox,int oy,int oz,const T* __restrict__ M,T* w){
 if(i==0){for(int j=0;j<6;j++)w[j]=0;return;}
 const int kx=i/((long long)ny*nh),ky=(i/nh)%ny,kz=i%nh;
 const T* X=axes+3*(ox+kx);const T* Y=axes+3*(oy+ky);const T* Z=axes+3*(oz+kz);
 const T g[6]={X[0]*X[0]*Y[2]*Z[2],Y[0]*Y[0]*X[2]*Z[2],Z[0]*Z[0]*X[2]*Y[2],X[1]*Y[1]*Z[2],X[1]*Z[1]*Y[2],Y[1]*Z[1]*X[2]};
 T K[6];
 #pragma unroll
 for(int a=0;a<6;a++){T v=0;
  #pragma unroll
  for(int u=0;u<6;u++)v+=M[6*a+u]*g[u];K[a]=v;}
 const T a=sqrt(K[0]),b=K[3]/a,c0=K[4]/a,d=sqrt(K[1]-b*b),e=(K[5]-c0*b)/d,f=sqrt(K[2]-c0*c0-e*e);
 const T l00=1/a,l10=-b/(a*d),l11=1/d,l20=(b*e-c0*d)/(a*d*f),l21=-e/(d*f),l22=1/f;
 w[0]=l00*l00+l10*l10+l20*l20;w[1]=l11*l11+l21*l21;w[2]=l22*l22;
 w[3]=l10*l11+l20*l21;w[4]=l20*l22;w[5]=l21*l22;
}
__device__ bool packed_spd(const double* w){
 double a=w[0],b=w[1],c=w[2],d=w[3],e=w[4],f=w[5];
 double scale=fmax(a,fmax(b,c));bool ok=isfinite(a)&&isfinite(b)&&isfinite(c)&&isfinite(d)&&isfinite(e)&&isfinite(f)&&scale>0;
 if(ok){
  a/=scale;b/=scale;c/=scale;d/=scale;e/=scale;f/=scale;
  double p=b-d*d/a,q=c-e*e/a-(f-d*e/a)*(f-d*e/a)/p;
  ok=a>0&&p>0&&q>0&&isfinite(p)&&isfinite(q);
 }
 return ok;
}
// FP64 working precision stores the evaluation (FP64 division and square
// roots cost ~1.5 ms per 192^3 iteration on this class of GPU); FP32 working
// precision evaluates it per frequency and stores nothing.
extern "C" __global__ void hex8_store_symbol(const REAL* axes,int ox,int oy,int oz,const REAL* M,REAL* table,
 long long n,int ny,int nh){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
 REAL w[6];reference_inverse<REAL>(i,ny,nh,axes,ox,oy,oz,M,w);
 for(int j=0;j<6;j++)table[j*n+i]=w[j];
}
// Preparation: at every frequency the working-precision evaluation used by
// the PCG must be finite SPD and agree with the FP64 one to 1e-3 (relative
// to its largest entry): finite or even SPD alone would accept an inverse
// that rounding has destroyed for a nearly singular reference.
extern "C" __global__ void hex8_check_symbol(const REAL* axes,int ox,int oy,int oz,const REAL* M,
 const double* axes64,const double* M64,int* bad,long long n,int ny,int nh){
 long long i=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n||i==0)return;
 REAL w[6];double v[6],u[6],scale=0,error=0;
 reference_inverse<REAL>(i,ny,nh,axes,ox,oy,oz,M,w);
 reference_inverse<double>(i,ny,nh,axes64,ox,oy,oz,M64,v);
 for(int j=0;j<6;j++){u[j]=w[j];scale=fmax(scale,fabs(v[j]));}
 for(int j=0;j<6;j++)error=fmax(error,fabs(u[j]-v[j]));
 if(!packed_spd(u)||!packed_spd(v)||!(error<=1e-3*scale))atomicExch(bad,1);
}
"""

_CUDA_PCG = r"""
// Device scalars s: 0 rho=r.z, 1 p.Ap, 2 alpha, 3 beta, 4 max|r|, 5 rho at
// the last true residual. state: 0 running / 1 recursive residual below
// threshold / 2 lost curvature / 3 replace by a true residual; state[1] is
// the iteration that finished. Every kernel is a no-op once state is final.
// history[6k..6k+5] for iteration k: rho_k, p.Ap, recursive rho_k+1,
// max|r_k+1|, rho of a replacing true residual (restart/resume), unused.
extern "C" __global__ void hex8_pcg_alpha(const double* partial,int m,double* s,int* state,
 double* history,int iteration){
 if(state[0])return;
 double x=0;for(int i=threadIdx.x;i<m;i+=blockDim.x)x+=partial[i];
 x=block_sum(x);
 if(threadIdx.x==0){s[1]=x;history[6*iteration]=s[0];history[6*iteration+1]=x;
  if(!(x>0)||!isfinite(x)||!(s[0]>0)||!isfinite(s[0])||
     !(s[0]/x>0)||!isfinite(s[0]/x)||!((REAL)(s[0]/x)>0)||!isfinite((REAL)(s[0]/x)))state[0]=2;
  else{s[2]=s[0]/x;s[7]-=s[2]*s[0];}}
}
extern "C" __global__ void hex8_pcg_update(const REAL* p,const REAL* ap,REAL* x,REAL* r,
 const double* s,const int* state,double* partial,long long n){
 if(state[0])return;
 const double a=s[2];double m=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  x[i]=(REAL)((double)x[i]+a*(double)p[i]);
  const REAL v=(REAL)((double)r[i]-a*(double)ap[i]);r[i]=v;
  m=fmax(m,magnitude(v));
 }
 m=block_max(m);if(threadIdx.x==0)partial[blockIdx.x]=m;
}
// In-place K0^-1 multiply of the unscaled R2C spectrum, scaled for an
// unscaled C2R; r.z by Parseval with RFFT multiplicities, accumulated in FP64.
extern "C" __global__ void hex8_pcg_prec(const REAL* axes,int ox,int oy,int oz,const REAL* M,const REAL* table,
 COMPLEX* r,double* partial,const int* state,long long n,int ny,int nh,int even,REAL scale,double weight){
 if(state[0])return;
 double d=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  COMPLEX x=r[i],y=r[n+i],z=r[2*n+i];
  REAL w[6];
  if(table){for(int j=0;j<6;j++)w[j]=table[j*n+i];}
  else reference_inverse<REAL>(i,ny,nh,axes,ox,oy,oz,M,w);
  const REAL a=w[0],b=w[1],c=w[2],e4=w[3],e5=w[4],f=w[5];
  COMPLEX q0=MAKE_COMPLEX(a*x.x+e4*y.x+e5*z.x,a*x.y+e4*y.y+e5*z.y);
  COMPLEX q1=MAKE_COMPLEX(e4*x.x+b*y.x+f*z.x,e4*x.y+b*y.y+f*z.y);
  COMPLEX q2=MAKE_COMPLEX(e5*x.x+f*y.x+c*z.x,e5*x.y+f*y.y+c*z.y);
  const int k=i%nh;
  d+=(k==0||(even&&k==nh-1)?1.:2.)*((double)x.x*q0.x+(double)x.y*q0.y+(double)y.x*q1.x
     +(double)y.y*q1.y+(double)z.x*q2.x+(double)z.y*q2.y);
  r[i]=MAKE_COMPLEX(q0.x*scale,q0.y*scale);r[n+i]=MAKE_COMPLEX(q1.x*scale,q1.y*scale);
  r[2*n+i]=MAKE_COMPLEX(q2.x*scale,q2.y*scale);
 }
 d=block_sum(d);if(threadIdx.x==0)partial[blockIdx.x]=d*weight;
}
// After preconditioning: convergence of the updated residual, then beta.
// mode 0 iterates, 1 restarts (p=z), 2 resumes after a true-residual
// replacement (keeps p). Mixed precision replaces once rho < replace*s[5].
// s[7] is the running energy D - sum alpha_j rho_j of the iterate. energy > 0
// stops when the recursive bound rho/gamma_- <= energy*(s[7]-rho/gamma_-)
// instead of on max|r|; acceptance re-tests this bound with the fresh
// residual. s[6] keeps the recursive rho that triggered a replacement.
extern "C" __global__ void hex8_pcg_beta(const double* rho,int m,const double* rmax,int k,
 double* s,int* state,double threshold,int iteration,int mode,double replace,double* history,
 double gamma,double energy){
 if(state[0])return;
 double x=0,y=0;
 for(int i=threadIdx.x;i<m;i+=blockDim.x)x+=rho[i];
 for(int i=threadIdx.x;i<k;i+=blockDim.x)y=fmax(y,rmax[i]);
 x=block_sum(x);y=block_max(y);
 if(threadIdx.x)return;
 state[1]=iteration;
 if(mode)history[6*iteration+4]=x;else{history[6*iteration+2]=x;history[6*iteration+3]=y;}
 if(mode){
  if(!(x>0)||!isfinite(x)){state[0]=2;return;}
  if(mode==2)s[3]=x/s[0];
  s[0]=x;s[5]=x;return;
 }
 s[4]=y;
 if(energy>0?x/gamma<=energy*(s[7]-x/gamma):y<=threshold){state[0]=1;return;}
 if(!(x>0)||!isfinite(x)){state[0]=2;return;}
 if(x<replace*s[5]){state[0]=3;s[6]=x;return;}
 s[3]=x/s[0];s[0]=x;
}
extern "C" __global__ void hex8_pcg_direction(const REAL* z,REAL* p,const double* s,const int* state,long long n){
 if(state[0])return;
 const double b=s[3];
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
  p[i]=(REAL)((double)z[i]+b*(double)p[i]);
}
// A host-resident iterate's rows: x += alpha p with the arithmetic of
// hex8_pcg_update, unless the iteration had stopped when alpha was formed
// (go: state[0] then).
extern "C" __global__ void hex8_settle(const REAL* p,OTHER* x,const double* s,const int* go,long long n){
 if(go[0])return;
 const double a=s[2];
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
  x[i]=(OTHER)((double)x[i]+a*(double)p[i]);
}
// Mixed precision: fold the FP32 increment into the FP64 solution.
extern "C" __global__ void hex8_accumulate(REAL* w,OTHER* x,long long n){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x){
  x[i]+=(OTHER)w[i];w[i]=0;
 }
}
extern "C" __global__ void hex8_dot_partial(const REAL* a,const OTHER* b,double* partial,long long n){
 double d=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
  d+=(double)a[i]*(double)b[i];
 d=block_sum(d);if(threadIdx.x==0)partial[blockIdx.x]=d;
}
extern "C" __global__ void hex8_max_partial(const OTHER* a,double* partial,long long n){
 double m=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
  m=fmax(m,magnitude(a[i]));
 m=block_max(m);if(threadIdx.x==0)partial[blockIdx.x]=m;
}
// Per-component sums for the translation gauge: partial[c*gridDim.x+block].
extern "C" __global__ void hex8_component_sum(const REAL* a,double* partial,long long n){
 for(int c=0;c<3;c++){
  double d=0;
  for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
   d+=(double)a[c*n+i];
  d=block_sum(d);if(threadIdx.x==0)partial[c*gridDim.x+blockIdx.x]=d;
 }
}
// Fixed-order final sums (one block), so repeated calls are bitwise identical.
extern "C" __global__ void hex8_finish(const double* partial,int m,int count,int maximum,double* out){
 for(int c=0;c<count;c++){
  double x=0;
  for(int i=threadIdx.x;i<m;i+=blockDim.x)x=maximum?fmax(x,partial[c*m+i]):x+partial[c*m+i];
  x=maximum?block_max(x):block_sum(x);
  if(threadIdx.x==0)out[c]=x;
 }
}
extern "C" __global__ void hex8_subtract_means(REAL* a,const double* sums,long long n){
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(long long)gridDim.x*blockDim.x)
  for(int c=0;c<3;c++)a[c*n+i]-=(REAL)(sums[c]/(double)n);
}
"""


def _kernel(source, name, dtype, other=None):
    real, complex_, make = (
        ("float", "float2", "make_float2")
        if np.dtype(dtype) == np.float32
        else ("double", "double2", "make_double2")
    )
    if other is not None:
        source = source.replace(
            "OTHER", "float" if np.dtype(other) == np.float32 else "double"
        )
    return cp.RawKernel(
        source.replace("MAKE_COMPLEX", make)
        .replace("COMPLEX", complex_)
        .replace("REAL", real),
        name,
    )


_REDUCTION_THREADS = 256
# Mixed precision: residual-norm drop between FP64 true-residual replacements.
# 1e-4 leaves one replacement plus the final audit per load at rtol 1e-6:
# 4-11 % faster than 1e-3 on six 192^3 geometries (+0..3 iterations per six
# loads) and 7-11 % at 384^3; at rtol 1e-8, gyroid 128^3 takes 6 iterations
# (+3 %) more. No intermediate replacement fails the first audit (FP32 floor).
_RELIABLE_DROP = 1e-4


def _reduction_grid(n):
    from cufhom_core.general.cupy_warp_reduction import reduction_blocks

    return reduction_blocks(n, _REDUCTION_THREADS)


class Hex8Operator:
    """General finite-contrast SPD material table on a periodic Cartesian grid.

    index: integer (nx,ny,nz) cell labels. stiffness: (P,6,6) or sym21 (P,21).
    reference: homogeneous (6,6) SPD matrix; default unweighted arithmetic mean
    of table rows, matching the FANS phase-reference convention. The same
    reference must be passed explicitly when comparing different phase tables.
    backend: 'numpy' or 'cupy'. dtype: float32 or float64, the precision of
    the material, the solution and the fresh-residual target. working: the
    precision of the Krylov kernels. CUDA float64 problems default to
    'float32', a reliable-update PCG (FP64 solution, FP64 true-residual
    replacements, fresh FP64 acceptance) that reaches the FP64 problem at
    FP32 bandwidth; working='float64' runs every kernel in FP64. Otherwise the
    working precision is dtype. evaluation: arithmetic of the fresh residual
    on CUDA, 'df64' (double-float pairs of FP32 at the FP32 rate, ~2^-46.5
    per operation) or 'float64'. No void phases.
    """

    def __init__(
        self,
        index,
        stiffness,
        *,
        lengths=(1.0, 1.0, 1.0),
        reference=None,
        backend="cupy",
        dtype="float64",
        working=None,
        evaluation="df64",
    ):
        if evaluation not in ("df64", "float64"):
            raise ValueError("evaluation must be df64 or float64")
        self.evaluation = evaluation
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("HEX8 supports float32 and float64")
        if working is None:
            working = (
                "float32"
                if (backend, self.dtype) == ("cupy", np.dtype("float64"))
                else self.dtype
            )
        self.working = np.dtype(working)
        if self.working == np.float64:
            self.evaluation = "float64"
        if self.working != self.dtype and (self.working, self.dtype, backend) != (
            np.dtype("float32"),
            np.dtype("float64"),
            "cupy",
        ):
            raise ValueError(
                "A distinct working precision requires CUDA, float64 problems and float32 Krylov kernels"
            )
        labels = np.asarray(index)
        if labels.ndim != 3 or min(labels.shape) < 2 or labels.dtype.kind not in "iu":
            raise ValueError(
                "index must be a three-dimensional integer grid with at least two cells per axis"
            )
        self.C = _material_table(stiffness, self.dtype)
        self.working_diagnostics = working_material_check(self.C, self.working)
        if (
            labels.min() < 0
            or labels.max() >= len(self.C)
            or len(self.C) > np.iinfo(np.int32).max
        ):
            raise ValueError("Material indices are outside the stiffness table")
        self.shape, self.n, self.phases = labels.shape, labels.size, len(self.C)
        self.lengths = np.asarray(lengths, float)
        if (
            self.lengths.shape != (3,)
            or not np.isfinite(self.lengths).all()
            or np.any(self.lengths <= 0)
        ):
            raise ValueError("lengths must contain three positive finite X,Y,Z lengths")
        if backend not in ("numpy", "cupy") or (backend == "cupy" and cp is None):
            raise ValueError("backend must be numpy or available cupy")
        self.xp = np if backend == "numpy" else cp
        self.reference = _material_table(
            (self.C.mean(0) if reference is None else np.asarray(reference))[None],
            np.float64,
        )[0]
        from .common import reference_bounds

        self.spectral_bounds = reference_bounds(self.C, self.reference)
        self.gamma_minus = self.spectral_bounds["gamma_minus"]
        self.gamma_plus = self.spectral_bounds["gamma_plus"]
        self.B = hex8_shape_gradients(self.shape, self.lengths)
        ke = np.einsum("giau,pij,gjbv->abuvp", self.B, self.C, self.B) / (8 * self.n)
        homogeneous = np.zeros((27, 3, 3, self.phases))
        for a, na in enumerate(_NODES):
            for b, nb in enumerate(_NODES):
                dx, dy, dz = nb - na
                homogeneous[(dx + 1) * 9 + (dy + 1) * 3 + dz + 1] += ke[a, b]
        # Exact mean gradients: paired opposite corners cancel even for a
        # homogeneous anisotropic material, without testing a tiny numerical RHS.
        bbar = np.empty((6, 8, 3))
        h = self.lengths / np.array(self.shape)
        self.working_diagnostics.update(_check_working_elements(ke, h, self.working))
        for a, corner in enumerate(_NODES):
            x, y, z = (2 * corner - 1) / (4 * h)
            bbar[:, a] = [
                [x, 0, 0],
                [0, y, 0],
                [0, 0, z],
                [0, z / np.sqrt(2), y / np.sqrt(2)],
                [z / np.sqrt(2), 0, x / np.sqrt(2)],
                [y / np.sqrt(2), x / np.sqrt(2), 0],
            ]
        self.Bbar = bbar
        force = -np.einsum("iau,pij->aujp", bbar[:, :4], self.C) / self.n
        # Volume (Voigt) average of the cell stiffness: the energy of E alone.
        xp = self.xp
        self.index = xp.asarray(labels, dtype=xp.int32, order="C")
        # Labels were validated above; on CUDA they are counted on the device.
        counts = (
            np.bincount(labels.ravel(), minlength=self.phases)
            if xp is np
            else device_label_counts(self.index, self.phases)
        )
        self.voigt = np.einsum("p,pij->ij", counts / self.n, self.C)
        self.ke = xp.asarray(ke, dtype=self.working, order="C")
        self.force64 = xp.asarray(force, dtype=xp.float64, order="C")
        self.ke64 = (
            self.ke
            if self.working == np.float64
            else xp.asarray(ke, dtype=xp.float64, order="C")
        )
        if xp is np:
            self._build_numpy_inverse()
        else:
            # Half-space symmetric stencil blocks (P, 13, 8): the exact block
            # is symmetric and equal for +o and -o; average the four FP64
            # realizations so the stored operator is symmetric by construction.
            half = np.zeros((self.phases, 13, 8))
            for o in range(13):
                block = (homogeneous[14 + o] + homogeneous[12 - o]).transpose(2, 0, 1)
                block = (block + block.transpose(0, 2, 1)) / 4
                half[:, o, :6] = block[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
            self.interior_table = xp.asarray(half, dtype=self.working)
            self.interior_table64 = (
                self.interior_table
                if self.working == np.float64
                else xp.asarray(half, dtype=xp.float64)
            )
            if self.evaluation == "df64":
                from .evaluation import _real_pairs

                self.ke_df, self.interior_table_df = (
                    _real_pairs(ke, "df64"),
                    _real_pairs(half, "df64"),
                )
            _precompile(self)
            flags = cp.empty(self.n, dtype=cp.uint8)
            _classify_kernel()(
                ((self.n + 255) // 256,),
                (256,),
                (
                    self.index,
                    flags,
                    np.int64(self.n),
                    np.int32(self.shape[1]),
                    np.int32(self.shape[2]),
                    np.int32(self.shape[0]),
                ),
            )
            self.interface_nodes = cp.flatnonzero(1 - flags)
            del flags
            self.interior_count = self.n - int(self.interface_nodes.size)
            nx, ny, nz = self.shape
            self._interior_grid = (
                (nz + 31) // 32,
                (ny + 7) // 8,
                -(-nx // INTERIOR_DEPTH),
            )
            self._interface_blocks = (int(self.interface_nodes.size) + 127) // 128
            self._partial = cp.empty(
                max(
                    self._interface_blocks + int(np.prod(self._interior_grid)),
                    (self.n + 127) // 128,
                    6 * self._interface_blocks,
                ),
                cp.float64,
            )
            self._idle = cp.zeros(2, cp.int32)
            self._stress = _kernel(_CUDA_STRESS, "hex8_general_stress", self.dtype)
            self._bbar_gpu = cp.asarray(self.Bbar)
            self._c_gpu = cp.asarray(self.C.transpose(1, 2, 0).copy())
            axes, offsets, symbol_map = _reference_symbol(
                self.reference, self.shape, h, self.n
            )
            self.symbol = (
                cp.asarray(axes, dtype=self.working),
                *map(np.int32, offsets),
                cp.asarray(symbol_map, dtype=self.working),
            )
            self.nfreq = nx * ny * (nz // 2 + 1)
            invalid = cp.zeros((), dtype=cp.int32)
            _reduction_kernel("hex8_check_symbol", self.working)(
                ((self.nfreq + 255) // 256,),
                (256,),
                (
                    *self.symbol,
                    cp.asarray(axes),
                    cp.asarray(symbol_map),
                    invalid,
                    np.int64(self.nfreq),
                    np.int32(ny),
                    np.int32(nz // 2 + 1),
                ),
            )
            if int(invalid):
                raise WorkingPrecisionError(
                    "Reference HEX8 inverse is not finite positive definite in the requested precision"
                )
            self.symbol += (np.uint64(0),)
            if self.working == np.float64:
                self.store_symbol_table()

    def store_symbol_table(self):
        """Cache the FP64 reference-inverse table for subsequent preconditioning.

        FP64 working precision: store the reference inverse (6 values per
        frequency) instead of computing it in every preconditioning.
        """
        ny, nh = self.shape[1], self.shape[2] // 2 + 1
        table = cp.empty((6, self.nfreq), cp.float64)
        _reduction_kernel("hex8_store_symbol", self.working)(
            ((self.nfreq + 255) // 256,),
            (256,),
            (
                *self.symbol[:-1],
                table,
                np.int64(self.nfreq),
                np.int32(ny),
                np.int32(nh),
            ),
        )
        self.symbol = self.symbol[:-1] + (table,)

    def release_symbol_table(self):
        """Release the reference-inverse table and compute its entries on demand.

        Compute the reference inverse in the preconditioner (the same
        function per frequency, about 13 % slower in FP64) and free its table;
        True if there was one.
        """
        stored = not isinstance(self.symbol[-1], np.uint64)
        self.symbol = self.symbol[:-1] + (np.uint64(0),)
        return stored

    def _build_numpy_inverse(self):
        freq = [np.fft.fftfreq(n) * np.pi for n in self.shape]
        freq[-1] = np.fft.rfftfreq(self.shape[-1]) * np.pi
        t = np.array(np.meshgrid(*freq, indexing="ij"))
        s, c = np.sin(t), np.cos(t)
        m, h = c * c + s * s / 3, self.lengths / np.array(self.shape)
        g = np.empty((3, 3, *t.shape[1:]))
        for i in range(3):
            for j in range(3):
                g[i, j] = (
                    4 * s[i] ** 2 / h[i] ** 2 * m[(i + 1) % 3] * m[(i + 2) % 3]
                    if i == j
                    else 4 * s[i] * s[j] * c[i] * c[j] / (h[i] * h[j]) * m[3 - i - j]
                )
        rows = [[0, 4, 5], [1, 3, 5], [2, 3, 4]]
        ds = [[0, 2, 1], [1, 2, 0], [2, 1, 0]]
        symbol = np.zeros((*t.shape[1:], 3, 3))
        for a, b, j, k in product(range(3), repeat=4):
            symbol[..., a, b] += (
                self.reference[rows[a][j], rows[b][k]]
                * g[ds[a][j], ds[b][k]]
                * (1 if j == 0 else 1 / np.sqrt(2))
                * (1 if k == 0 else 1 / np.sqrt(2))
                / self.n
            )
        symbol[0, 0, 0] = np.eye(3)
        self.inverse = np.linalg.inv(symbol).astype(self.working)
        self.inverse[0, 0, 0] = 0
        if not np.isfinite(self.inverse).all():
            raise ValueError(
                "Reference HEX8 inverse lost positive definiteness in the requested precision"
            )
        stored = self.inverse.astype(np.float64)
        stored[0, 0, 0] = np.eye(3)
        try:
            np.linalg.cholesky(stored)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Stored reference HEX8 inverse is not positive definite in the requested precision"
            ) from error

    def action(self, u, *, audit=False, out=None):
        """K u; audit evaluates the rounded input material in FP64 afresh."""
        xp, dtype = self.xp, np.float64 if audit else self.working
        u = xp.ascontiguousarray(u, dtype=dtype)
        if u.shape != (3, *self.shape):
            raise ValueError("Expected three displacement components on the X,Y,Z grid")
        out = xp.empty_like(u) if out is None else out
        if (
            out.shape != u.shape
            or out.dtype != u.dtype
            or not out.flags.c_contiguous
            or xp.may_share_memory(u, out)
        ):
            raise ValueError(
                "action output must be contiguous, correctly typed and distinct from its input"
            )
        ke = self.ke64 if audit else self.ke
        if xp is np:
            out.fill(0)
            for a, na in enumerate(_NODES):
                phase = np.roll(self.index, tuple(na), (0, 1, 2))
                for b, nb in enumerate(_NODES):
                    if b == a:
                        continue
                    values = np.roll(u, tuple(na - nb), (1, 2, 3)) - u
                    out += np.einsum(
                        "ij...,j...->i...", np.take(ke[a, b], phase, axis=-1), values
                    )
        else:
            self._launch(0, dtype, u, out, self._partial, self._idle)
        return out

    def _launch(self, mode, real, u, out, partial, state, macro=None):
        """Interface gather and interior stencil; returns the partial count."""
        ke, table = (
            (self.ke_df, self.interior_table_df)
            if isinstance(real, str)
            else (self.ke64, self.interior_table64)
            if real == np.float64
            else (self.ke, self.interior_table)
        )
        written = 0
        if self._interface_blocks:
            _action_kernel("interface", mode, real, u.dtype, out.dtype)(
                (self._interface_blocks,),
                (128,),
                (
                    self.index,
                    ke,
                    self.interface_nodes,
                    u,
                    out,
                    partial,
                    state,
                    self.force64,
                    self._partial if macro is None else macro,
                    np.int64(self.n),
                    np.int64(self.interface_nodes.size),
                    np.int32(self.shape[1]),
                    np.int32(self.shape[2]),
                    np.int32(self.shape[0]),
                    np.int32(self.phases),
                ),
            )
            written = self._interface_blocks
        if self.interior_count:
            _action_kernel("interior", mode, real, u.dtype, out.dtype)(
                self._interior_grid,
                (32, 8, 1),
                (
                    self.index,
                    table,
                    u,
                    out,
                    partial[written:],
                    state,
                    *map(np.int32, self.shape),
                ),
            )
            written += int(np.prod(self._interior_grid))
        return written

    def rhs(self, macro):
        """FP64 volume-normalized nodal loading, with exact opposite-corner pairing."""
        macro = np.asarray(macro, float)
        if macro.shape != (6,) or not np.isfinite(macro).all():
            raise ValueError("macro must contain six finite native Mandel components")
        xp = self.xp
        out = xp.zeros((3, *self.shape), dtype=xp.float64)
        if xp is np:
            for a in range(4):
                p = np.roll(self.index, tuple(_NODES[a]), (0, 1, 2))
                q = np.roll(self.index, tuple(_NODES[7 - a]), (0, 1, 2))
                f = np.einsum("cjp,j->cp", self.force64[a], macro)
                out += f[:, p] - f[:, q]
        else:
            self._rhs_into(out, macro)
        return out

    def _rhs_into(self, out, macro):
        """CUDA: write the load in out's precision; return its FP64 max norm."""
        blocks = (self.n + 127) // 128
        _action_kernel("rhs", 0, np.float64, np.float64, out.dtype)(
            (blocks,),
            (128,),
            (
                self.index,
                self.force64,
                cp.asarray(macro, dtype=cp.float64),
                out,
                self._partial,
                np.int64(self.n),
                np.int32(self.shape[1]),
                np.int32(self.shape[2]),
                np.int32(self.shape[0]),
                np.int32(self.phases),
            ),
        )
        return float(_finish(self._partial[:blocks], 1, maximum=True)[0])

    def _residual_into(self, x, macro, out):
        """Evaluate the unrounded fresh nodal residual in high precision.

        CUDA: out = f - K x evaluated afresh in FP64 or double-float
        (evaluation; stored in out's precision); returns the max norm of the
        unrounded residual.
        """
        if type(self).action is not Hex8Operator.action:
            fresh = self.rhs(macro) - self.action(x, audit=True)
            cp.copyto(out, fresh, casting="same_kind")
            return _max_abs(fresh)
        real = "df64" if self.evaluation == "df64" else np.float64
        written = self._launch(
            2,
            real,
            x,
            out,
            self._partial,
            self._idle,
            cp.asarray(macro, dtype=cp.float64),
        )
        return float(_finish(self._partial[:written], 1, maximum=True)[0])

    def precondition(self, residual):
        """Apply the reference inverse to a validated nodal residual field."""
        xp = self.xp
        residual = xp.ascontiguousarray(residual, dtype=self.working)
        if residual.shape != (3, *self.shape):
            raise ValueError("Expected three residual components on the X,Y,Z grid")
        spectrum = xp.fft.rfftn(residual, axes=(1, 2, 3))
        if xp is np:
            spectrum = np.einsum("...ij,j...->i...", self.inverse, spectrum)
        else:
            grid = _reduction_grid(self.nfreq)
            _reduction_kernel("hex8_pcg_prec", self.working)(
                (grid,),
                (_REDUCTION_THREADS,),
                (
                    *self.symbol,
                    spectrum,
                    cp.empty(grid, cp.float64),
                    self._idle,
                    np.int64(self.nfreq),
                    np.int32(self.shape[1]),
                    np.int32(self.shape[2] // 2 + 1),
                    np.int32(0),
                    self.working.type(1),
                    np.float64(0),
                ),
            )
        return xp.fft.irfftn(spectrum, s=self.shape, axes=(1, 2, 3)).astype(
            self.working, copy=False
        )

    def mean_stress(self, u, macro):
        """FP64 mean stress; CUDA keeps strain and stress local to each thread."""
        xp = self.xp
        if xp is not np:
            u = xp.ascontiguousarray(u, dtype=self.dtype)
            blocks = (self.n + 127) // 128
            partial = xp.empty((6, blocks), dtype=xp.float64)
            self._stress(
                (blocks,),
                (128,),
                (
                    self.index,
                    u,
                    self._bbar_gpu,
                    self._c_gpu,
                    xp.asarray(macro, dtype=xp.float64),
                    partial,
                    np.int64(self.n),
                    np.int32(self.shape[1]),
                    np.int32(self.shape[2]),
                    np.int32(self.shape[0]),
                    np.int32(self.phases),
                ),
                shared_mem=6 * 128 * 8,
            )
            return cp.asnumpy(partial.sum(axis=1)) / self.n
        strain = xp.broadcast_to(
            xp.asarray(macro, dtype=xp.float64)[:, None, None, None], (6, *self.shape)
        ).copy()
        for a, node in enumerate(_NODES):
            values = xp.roll(u, tuple(-node), (1, 2, 3))
            strain += xp.einsum("ij,j...->i...", xp.asarray(self.Bbar[:, a]), values)
        # Chunking bounds the constitutive gather even for many orientation groups.
        total = xp.zeros(6, dtype=xp.float64)
        flat, labels = strain.reshape(6, -1), self.index.ravel()
        c = xp.asarray(self.C)
        for start in range(0, self.n, 262144):
            sl = slice(start, min(start + 262144, self.n))
            stress = xp.einsum("nij,jn->in", c[labels[sl]], flat[:, sl])
            total += stress.sum(axis=1, dtype=xp.float64)
        return np.asarray(total) / self.n if xp is np else cp.asnumpy(total) / self.n

    def load_work_stress(self, u, macro):
        """Mean Gauss stress through the load work, Cbar E - (f(e_j).u)_j.

        Exactly mean_stress in exact arithmetic (same B-bar quadrature); on
        CUDA it reads only interface nodes, where the load is nonzero.
        """
        macro = np.asarray(macro, float)
        if self.xp is np:
            return self.voigt @ macro - np.array(
                [
                    float(
                        np.einsum(
                            "i,i->", self.rhs(e).ravel(), u.ravel(), dtype=np.float64
                        )
                    )
                    for e in np.eye(6)
                ]
            )
        if not self._interface_blocks:
            return self.voigt @ macro
        partial = self._partial[: 6 * self._interface_blocks]
        _action_kernel("moments", 0, np.float64, u.dtype, np.float64)(
            (self._interface_blocks,),
            (128,),
            (
                self.index,
                self.force64,
                self.interface_nodes,
                u,
                partial,
                np.int64(self.n),
                np.int64(self.interface_nodes.size),
                np.int32(self.shape[1]),
                np.int32(self.shape[2]),
                np.int32(self.shape[0]),
                np.int32(self.phases),
            ),
        )
        return self.voigt @ macro - _finish(partial, 6)

    def dot(self, a, b):
        """FP64-accumulated product of two FP32/FP64 fields of equal size."""
        if self.xp is np:
            return float(np.einsum("i,i->", a.ravel(), b.ravel(), dtype=np.float64))
        a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
        if a.size != b.size:
            raise ValueError("dot requires fields of equal size")
        if a.dtype not in (cp.float32, cp.float64) or b.dtype not in (
            cp.float32,
            cp.float64,
        ):
            raise ValueError("dot requires FP32/FP64 fields")
        partial = cp.empty(_reduction_grid(a.size), cp.float64)
        _reduction_kernel("hex8_dot_partial", a.dtype, b.dtype)(
            (partial.size,), (_REDUCTION_THREADS,), (a, b, partial, np.int64(a.size))
        )
        return float(_finish(partial, 1)[0])

    def _action_partials(self, u, out, state, partial):
        """Launch out = K u (working precision) with block partials of u.Ku.

        Returns the number of partials written. Subclasses that replace the
        stiffness action keep the same PCG through an unfused dot.
        """
        if type(self).action is not Hex8Operator.action:
            self.action(u, out=out)
            blocks = min(partial.size, _reduction_grid(u.size))
            _reduction_kernel("hex8_dot_partial", u.dtype, out.dtype)(
                (blocks,), (_REDUCTION_THREADS,), (u, out, partial, np.int64(u.size))
            )
            return blocks
        return self._launch(1, self.working, u, out, partial, state)

    def _action_partial_capacity(self):
        if type(self).action is not Hex8Operator.action:
            return _reduction_grid(3 * self.n)
        return self._interface_blocks + int(np.prod(self._interior_grid))


_ACTION_KERNELS = {}


def _action_kernel(kind, mode, real, xt, out):
    """Compiled interface/interior/rhs kernel for (mode, arithmetic, input, output).

    real 'df64' is double-float arithmetic (pairs of FP32, evaluation.py) for
    the FP64-class fresh residual (MODE 2) at the FP32 rate.
    """
    df = isinstance(real, str)
    if df and (real != "df64" or mode != 2 or kind not in ("interface", "interior")):
        raise ValueError(
            "double-float arithmetic serves the fresh residual of the action kernels"
        )
    from .hex8_fused import _FAST32

    fast = int(_FAST32 and not df and np.dtype(real) == np.float32)
    key = (
        kind,
        mode,
        real if df else np.dtype(real).str,
        *(np.dtype(t).str for t in (xt, out)),
        fast,
    )
    if key not in _ACTION_KERNELS:
        from .precision import DF64_CUDA

        name = {np.dtype("float32"): "float", np.dtype("float64"): "double"}
        source = {
            "interface": _CUDA_INTERFACE,
            "interior": _CUDA_INTERIOR,
            "rhs": _CUDA_RHS,
            "moments": _CUDA_MOMENTS,
        }[kind]
        convert = (
            DF64_CUDA + "#define TOREAL(x) from_f64(x)\n#define TODOUBLE(x) to_f64(x)\n"
            if df
            else "#define TOREAL(x) ((REAL)(x))\n#define TODOUBLE(x) ((double)(x))\n"
        )
        code = (
            f"#define MODE {mode}\n#define FAST32 {fast}\n#define IS_DOUBLE {2 if df else int(np.dtype(real) == np.float64)}\n"
            f"#define INTERIOR_DEPTH {INTERIOR_DEPTH}\n"
            + _CUDA_REDUCE
            + (convert + source)
            .replace("REAL", "df" if df else name[np.dtype(real)])
            .replace("XT", name[np.dtype(xt)])
            .replace("OUT", name[np.dtype(out)])
        )
        if kind in ("interface", "interior"):
            # One name per mode, so profiles separate PCG products from FP64 residuals.
            name = "hex8_" + kind + ("", "_dot", "_residual")[mode]
            code = code.replace(" hex8_" + kind + "(", " " + name + "(")
        else:
            name = {"rhs": "hex8_general_rhs", "moments": "hex8_force_moments"}[kind]
        _ACTION_KERNELS[key] = cp.RawKernel(code, name)
    return _ACTION_KERNELS[key]


def _reference_symbol(reference, shape, h, n):
    """Per-axis (A, B, m) tables and the packed map K = M g of the reference."""
    tables, offsets, start = [], [], 0
    for axis, size in enumerate(shape):
        k = np.arange(size // 2 + 1 if axis == 2 else size)
        t = np.pi * k / size
        a = 2 * np.sin(t) / h[axis]
        tables.append(
            np.stack([a, a * np.cos(t), np.cos(t) ** 2 + np.sin(t) ** 2 / 3], axis=1)
        )
        offsets.append(start)
        start += len(k)
    rows = [[0, 4, 5], [1, 3, 5], [2, 3, 4]]  # Mandel strain rows of each displacement
    derivative = [[0, 2, 1], [1, 2, 0], [2, 1, 0]]
    weight = [1.0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
    packed = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    index = {pair: u for u, pair in enumerate(packed)}
    symbol_map = np.zeros((6, 6))
    for out, (a, b) in enumerate(packed):
        for j in range(3):
            for k in range(3):
                p, q = sorted((derivative[a][j], derivative[b][k]))
                symbol_map[out, index[(p, q)]] += (
                    reference[rows[a][j], rows[b][k]] * weight[j] * weight[k] / n
                )
    return np.concatenate(tables).ravel(), offsets, symbol_map.ravel()


_REDUCTION_KERNELS = {}


def _reduction_kernel(name, dtype, other=None):
    other = dtype if other is None else other
    key = (name, np.dtype(dtype).str, np.dtype(other).str)
    if key not in _REDUCTION_KERNELS:
        _REDUCTION_KERNELS[key] = _kernel(
            _CUDA_REDUCE + _CUDA_SYMBOL + _CUDA_PCG, name, dtype, other
        )
    return _REDUCTION_KERNELS[key]


_CLASSIFY = []


def _classify_kernel():
    if not _CLASSIFY:
        _CLASSIFY.append(cp.RawKernel(_CUDA_CLASSIFY, "hex8_general_classify"))
    return _CLASSIFY[0]


def _precompile(op):
    """Compile the CUDA solve kernels concurrently before their first launches.

    Compile every kernel a CUDA solve uses, concurrently (NVRTC releases
    the GIL), instead of one after another at their first launches; the
    generated FFT kernels of hex8_fused join the same pool when they apply.
    """
    from concurrent.futures import ThreadPoolExecutor

    from . import hex8_fused

    w, d, f64, f32 = op.working, op.dtype, np.dtype("float64"), np.dtype("float32")
    kernels = [_reduction_kernel("hex8_check_symbol", w), _classify_kernel()]
    kernels += [_action_kernel(kind, 1, w, w, w) for kind in ("interface", "interior")]
    residual = "df64" if op.evaluation == "df64" else f64
    kernels += [
        _action_kernel(kind, 2, residual, d, w) for kind in ("interface", "interior")
    ]
    kernels += [
        _action_kernel("rhs", 0, f64, f64, w),
        _action_kernel("moments", 0, f64, d, f64),
    ]
    kernels += [
        _reduction_kernel(name, *types)
        for name, types in (
            ("hex8_pcg_beta", (w, w)),
            ("hex8_finish", (f64, f64)),
            ("hex8_dot_partial", (d, w)),
            ("hex8_dot_partial", (f32, w)),
            ("hex8_component_sum", (d, d)),
            ("hex8_accumulate", (w, d)),
        )
    ]
    jobs = [lambda kernel=kernel: kernel.attributes for kernel in kernels]
    jobs += hex8_fused.compile_jobs(op)
    jobs[0]()  # the first compile fills CuPy's NVRTC probe cache
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda job: job(), jobs[1:]))


def _finish(partial, count, maximum=False):
    """Deterministic one-block reduction of count rows of block partials."""
    out = cp.empty(count, cp.float64)
    _reduction_kernel("hex8_finish", np.float64, np.float64)(
        (1,),
        (_REDUCTION_THREADS,),
        (
            partial,
            np.int32(partial.size // count),
            np.int32(count),
            np.int32(maximum),
            out,
        ),
    )
    return out.get()


class _CudaPCG:
    """Device-resident PCG of one CUDA operator, shared by its six loads.

    alpha, beta and the convergence test stay on the device. The host reads
    the two-integer state of the previous iteration while the next one is
    queued, so the GPU never waits for a host decision; once final, every
    PCG kernel of a queued iteration returns immediately. Fresh FP64 audits
    and restarts keep the semantics of the NumPy realization.
    """

    def __init__(self, op, *, library_fft=True):
        import cupyx

        self.op, dt = op, op.working
        shape = (3, *op.shape)
        self.n3 = 3 * op.n
        self.r, self.p, self.ap = (cp.empty(shape, dt) for _ in range(3))
        # K p is consumed by the residual update before z = K0^-1 r is formed.
        self.z = self.ap
        # Mixed precision: FP32 increment since the last FP64 replacement.
        self.mixed = dt != op.dtype
        self.increment = cp.zeros(shape, dt) if self.mixed else None
        self.replace = np.float64(_RELIABLE_DROP**2 if self.mixed else 0.0)
        nh = op.shape[2] // 2 + 1
        self.nf = op.shape[0] * op.shape[1] * nh
        self.grid, self.fgrid = _reduction_grid(self.n3), _reduction_grid(self.nf)
        self.pap = cp.empty(op._action_partial_capacity(), cp.float64)
        self.s = cp.zeros(8, cp.float64)
        self.state = cp.zeros(2, cp.int32)
        self.idle = cp.zeros(2, cp.int32)
        self.go = cp.zeros(
            1, cp.int32
        )  # state[0] when alpha formed (host-resident iterate)
        self.history = None
        self.host = cupyx.empty_pinned((2, 2), np.int32)
        self.events = [
            cp.cuda.Event(block=False, disable_timing=True) for _ in range(2)
        ]
        k = {
            name: _reduction_kernel(name, dt)
            for name in (
                "hex8_pcg_alpha",
                "hex8_pcg_update",
                "hex8_pcg_prec",
                "hex8_pcg_beta",
                "hex8_pcg_direction",
            )
        }
        self.k = k
        # Generated FFTs own their spectrum and reductions. Do not construct
        # cuFFT plans/work areas only to discard them in FusedCudaPCG.
        self.forward = self.backward = self.spectrum = None
        self.rmax = self.rho = self.prec_args = self.measure_args = None
        if library_fft:
            from cupyx.scipy.fft import get_fft_plan

            complex_ = cp.complex64 if dt == np.float32 else cp.complex128
            self.spectrum = cp.empty((3, op.shape[0], op.shape[1], nh), complex_)
            self.forward = get_fft_plan(self.r, axes=(1, 2, 3), value_type="R2C")
            self.backward = get_fft_plan(
                self.spectrum, shape=op.shape, axes=(1, 2, 3), value_type="C2R"
            )
            self.rmax = cp.empty(self.grid, cp.float64)
            self.rho = cp.empty(self.fgrid, cp.float64)
            self.prec_args = (
                *op.symbol,
                self.spectrum,
                self.rho,
                self.state,
                np.int64(self.nf),
                np.int32(op.shape[1]),
                np.int32(nh),
                np.int32(op.shape[2] % 2 == 0),
                dt.type(1 / op.n),
                np.float64(1 / op.n),
            )
            self.measure_args = self.prec_args[:8] + (self.idle,) + self.prec_args[9:]
        self.gamma, self.energy = np.float64(op.gamma_minus), np.float64(0.0)

    def reset_history(self, maxiter):
        if self.history is None or self.history.size < 6 * (maxiter + 1):
            self.history = cp.empty(6 * (maxiter + 1), cp.float64)
        self.history.fill(np.nan)

    def read_history(self, last):
        h = self.history[: 6 * (last + 1)].reshape(-1, 6).get()
        return dict(
            rho=h[:, 0].tolist(),
            pAp=h[:, 1].tolist(),
            rho_next=h[:, 2].tolist(),
            residual_inf=h[:, 3].tolist(),
            rho_replaced=h[:, 4].tolist(),
        )

    def energy_norm(self):
        """r.K0^-1 r of the current residual buffer (FP64 accumulation)."""
        self.forward.fft(self.r, self.spectrum, cp.cuda.cufft.CUFFT_FORWARD)
        self.k["hex8_pcg_prec"]((self.fgrid,), (_REDUCTION_THREADS,), self.measure_args)
        return float(_finish(self.rho, 1)[0])

    def _precondition(self, iteration, threshold, mode):
        self.forward.fft(self.r, self.spectrum, cp.cuda.cufft.CUFFT_FORWARD)
        self.k["hex8_pcg_prec"]((self.fgrid,), (_REDUCTION_THREADS,), self.prec_args)
        self.backward.fft(self.spectrum, self.z, cp.cuda.cufft.CUFFT_INVERSE)
        self.k["hex8_pcg_beta"](
            (1,),
            (_REDUCTION_THREADS,),
            (
                self.rho,
                np.int32(self.fgrid),
                self.rmax,
                np.int32(self.grid),
                self.s,
                self.state,
                np.float64(threshold),
                np.int32(iteration),
                np.int32(mode),
                self.replace,
                self.history,
                self.gamma,
                self.energy,
            ),
        )

    def prepare(self, iteration, threshold, mode):
        """Precondition a fresh residual and return its reference-energy product.

        z = K0^-1 r and rho = r.z of a fresh residual in r; returns rho
        (also G0 of the fresh residual). mode 1 restarts, 2 resumes.
        """
        self.state.fill(0)
        self._precondition(iteration, threshold, mode)
        return float(self.history[6 * iteration + 4])

    def proceed(self, mode):
        """Restart p = z, or resume (reliable update) p = z + beta p."""
        if mode == 1:
            cp.copyto(self.p, self.z)
        else:
            self.k["hex8_pcg_direction"](
                (self.grid,),
                (_REDUCTION_THREADS,),
                (self.z, self.p, self.s, self.state, np.int64(self.n3)),
            )

    def start(self, iteration, threshold):
        self.prepare(iteration, threshold, 1)
        self.proceed(1)

    def release(self):
        """Release the Krylov workspace during a fresh host-solution evaluation.

        Give the memory of z (= Ap) to a whole host-resident solution
        during a fresh evaluation; restore() reallocates it.
        """
        self.ap = self.z = None

    def restore(self):
        self.ap = cp.empty((3, *self.op.shape), self.op.working)
        self.z = self.ap

    def released_bytes(self):
        return 3 * self.op.n * self.op.working.itemsize

    def flush(self, x):
        """Mixed precision: x += increment in FP64, increment = 0."""
        _reduction_kernel("hex8_accumulate", self.increment.dtype, x.dtype)(
            (self.grid,), (_REDUCTION_THREADS,), (self.increment, x, np.int64(self.n3))
        )

    def iterate(self, x, first, last, threshold):
        """Queue iterations first+1..last; return (state, finishing iteration)."""
        k, n3 = self.k, np.int64(self.n3)
        x = self.increment if self.mixed else x
        if not isinstance(x, cp.ndarray):
            raise ValueError("A host-resident HEX8 iterate needs the fused PCG")
        polled = None
        for iteration in range(first + 1, last + 1):
            m = self.op._action_partials(self.p, self.ap, self.state, self.pap)
            k["hex8_pcg_alpha"](
                (1,),
                (_REDUCTION_THREADS,),
                (
                    self.pap,
                    np.int32(m),
                    self.s,
                    self.state,
                    self.history,
                    np.int32(iteration),
                ),
            )
            k["hex8_pcg_update"](
                (self.grid,),
                (_REDUCTION_THREADS,),
                (self.p, self.ap, x, self.r, self.s, self.state, self.rmax, n3),
            )
            self._precondition(iteration, threshold, 0)
            k["hex8_pcg_direction"](
                (self.grid,),
                (_REDUCTION_THREADS,),
                (self.z, self.p, self.s, self.state, n3),
            )
            slot = iteration % 2
            self.state.get(out=self.host[slot], blocking=False)
            self.events[slot].record()
            if polled is not None:
                self.events[polled % 2].synchronize()
                if self.host[polled % 2, 0]:
                    break
            polled = iteration
        state = self.state.get()
        return int(state[0]), int(state[1])


def _subtract_means(x):
    n = x[0].size
    grid = _reduction_grid(n)
    partial = cp.empty(3 * grid, cp.float64)
    _reduction_kernel("hex8_component_sum", x.dtype)(
        (grid,), (_REDUCTION_THREADS,), (x, partial, np.int64(n))
    )
    sums = cp.asarray(_finish(partial, 3))
    _reduction_kernel("hex8_subtract_means", x.dtype)(
        (grid,), (_REDUCTION_THREADS,), (x, sums, np.int64(n))
    )


def _max_abs(a):
    partial = cp.empty(_reduction_grid(a.size), cp.float64)
    _reduction_kernel("hex8_max_partial", np.float64, a.dtype)(
        (partial.size,), (_REDUCTION_THREADS,), (a, partial, np.int64(a.size))
    )
    return float(_finish(partial, 1, maximum=True)[0])


def solve_hex8_load(operator, macro, *, rtol=1e-6, atol=0.0, maxiter=3000):
    """PCG with fresh FP64 nodal Linf acceptance, never a relaxed FP32 floor.

    On a rejected candidate the residual is replaced and CG restarts. The
    iteration budget includes restarts; nonconvergence raises an exception.
    The returned residual is the fresh one, in the Krylov working precision.
    """
    rtol, atol, maxiter = _solver_parameters(rtol, atol, maxiter)
    if operator.xp is not np:
        info = _solve_cuda(operator, macro, rtol, atol, maxiter)
        info["residual"] = info["residual"].copy()  # the PCG reuses its buffer
        return info
    xp = operator.xp
    rhs64 = operator.rhs(macro)
    norm = float(xp.max(xp.abs(rhs64)))
    if not np.isfinite(norm):
        raise RuntimeError("HEX8 nodal loading is nonfinite")
    threshold = max(atol, rtol * norm)
    x = xp.zeros((3, *operator.shape), dtype=operator.dtype)
    if norm <= threshold:
        return dict(
            solution=x,
            residual=rhs64,
            iterations=0,
            restarts=0,
            audits=1,
            recursive_residual_inf=norm,
            replacements=0,
            evaluation="float64",
            residual_inf=norm,
            rhs_inf=norm,
            threshold=threshold,
            accepted=True,
            **_energy_bound(
                operator, operator.dot(rhs64, operator.precondition(rhs64))
            ),
        )
    r = rhs64.astype(operator.dtype)
    z = operator.precondition(r)
    p = z.copy()
    rho = operator.dot(r, z)
    ap = xp.empty_like(x)
    restarts, audits, progress = 0, 0, FreshProgress()
    for iteration in range(1, maxiter + 1):
        operator.action(p, out=ap)
        pap = operator.dot(p, ap)
        if not np.isfinite(pap + rho) or pap <= 0 or rho <= 0:
            raise LinearSolveError(
                "HEX8 PCG lost positive curvature before physical acceptance",
                status="lost_curvature",
                iterations=iteration,
            )
        alpha = rho / pap
        if (
            not np.isfinite(alpha)
            or alpha <= 0
            or not np.isfinite(operator.dtype.type(alpha))
            or operator.dtype.type(alpha) <= 0
        ):
            raise LinearSolveError(
                "HEX8 PCG unsafe step", status="unsafe_step", iterations=iteration
            )
        x += alpha * p
        r -= alpha * ap
        if float(xp.max(xp.abs(r))) <= threshold or iteration == maxiter:
            recursive_max = float(xp.max(xp.abs(r)))
            x -= x.mean(axis=(1, 2, 3), keepdims=True)
            fresh = rhs64 - operator.action(x, audit=True)
            audits += 1
            value = float(xp.max(xp.abs(fresh)))
            if np.isfinite(value) and value <= threshold:
                return dict(
                    solution=x,
                    residual=fresh,
                    iterations=iteration,
                    restarts=restarts,
                    audits=audits,
                    recursive_residual_inf=recursive_max,
                    replacements=0,
                    evaluation="float64",
                    residual_inf=value,
                    rhs_inf=norm,
                    threshold=threshold,
                    accepted=True,
                    **_energy_bound(
                        operator, operator.dot(fresh, operator.precondition(fresh))
                    ),
                )
            progress.check(value, iteration)
            r = fresh.astype(operator.dtype)
            z = operator.precondition(r)
            p = z.copy()
            rho = operator.dot(r, z)
            restarts += 1
            continue
        z = operator.precondition(r)
        new = operator.dot(r, z)
        p *= new / rho
        p += z
        rho = new
    raise LinearSolveError(
        f"HEX8 failed fresh residual: {value:.6g} > {threshold:.6g} after {maxiter} updates ({restarts} restarts)",
        status="maxiter",
        iterations=maxiter,
    )


def _solve_cuda(
    operator, macro, rtol, atol, maxiter, energy_rtol=None, residency="device"
):
    """Run PCG with device scalars and the NumPy loop acceptance criterion.

    Neither the load nor an FP64 residual is stored: the load is written in
    the working precision into r, and every fresh FP64 residual f - K x is
    evaluated by one fused kernel straight into r; any such evaluation that
    passes acceptance is accepted. The returned residual is that PCG buffer,
    valid until the next solve with this operator.

    energy_rtol: goal-oriented alternative to the nodal Linf criterion. CG
    stops when the recursive bound rho/gamma_minus on the energy error of
    this load is below energy_rtol times its running energy; a candidate is
    accepted only if the fresh-residual bound G0/gamma_minus (FP64 residual,
    reference inverse) is below energy_rtol*(C~ - G0/gamma_minus) with the
    candidate energy C~ = <sigma>.E - u.r. The corresponding exact-arithmetic
    inequality bounds field-energy error. The computed test excludes residual,
    inverse and assembly rounding and is not a floating-point certificate;
    it does not bound the nodal residual. rtol and atol are then not used.

    residency 'host' (mixed: the FP64 solution) or 'iterate' (the solution of
    the fixed configurations, which is the Krylov iterate) keeps the solution
    in pinned host memory (coordinates._HostVector) with its leading planes
    that fit on the device; every fresh evaluation gathers it on the device
    in the memory of z and the preconditioner spectrum, and also forms the
    candidate's mean stress through the load moments and u.r (returned as
    mean_stress, solution_dot).
    """
    macro = np.asarray(macro, float)
    if macro.shape != (6,) or not np.isfinite(macro).all():
        raise ValueError("macro must contain six finite native Mandel components")
    energy = energy_rtol is not None
    if energy and (not np.isfinite(energy_rtol) or not 0 < energy_rtol < 1):
        raise ValueError("energy_rtol must lie in (0, 1)")
    pcg = getattr(operator, "_pcg", None)
    if pcg is None:
        # Generated FFTs fused with the PCG vectors where supported (hex8_fused).
        from .hex8_fused import fused_pcg

        pcg = operator._pcg = fused_pcg(operator) or _CudaPCG(operator)
    norm = operator._rhs_into(pcg.r, macro)
    if not np.isfinite(norm):
        raise RuntimeError("HEX8 nodal loading is nonfinite")
    threshold = -1.0 if energy else max(atol, rtol * norm)
    host, stored = residency != "device", {}
    if host:
        from .coordinates import _HostVector

        n0, n1, n2 = operator.shape
        x = _HostVector(
            3 * n0, n1 * n2, operator.dtype, _hex8_resident_rows(operator, pcg)
        )
    else:
        x = cp.zeros((3, *operator.shape), dtype=operator.dtype)

    def fresh(evaluate):
        """evaluate(u) on the whole solution on the device."""
        if not host:
            return evaluate(x)
        pool = cp.get_default_memory_pool()
        pcg.release()
        pool.free_all_blocks()  # z and the spectrum leave no split blocks for u
        u = cp.empty((3, *operator.shape), operator.dtype)
        try:
            x.gather(u)
            out = evaluate(u)
            x.scatter(u)
            stored.update(
                mean_stress=operator.load_work_stress(u, macro),
                solution_dot=operator.dot(u, pcg.r),
            )
        finally:
            del u
            pool.free_all_blocks()
            pcg.restore()
        return out

    def evaluate(u):
        if pcg.mixed:
            pcg.flush(u)
        _subtract_means(u)
        value = operator._residual_into(u, macro, pcg.r)
        # Exact energy of the candidate: <sigma>.E - u.r (C~ = C_stress - u.r).
        return value, float(macro @ operator.load_work_stress(u, macro)) - operator.dot(
            u, pcg.r
        ) if energy else None

    common = dict(
        rhs_inf=norm,
        threshold=threshold,
        accepted=True,
        criterion="energy" if energy else "nodal_linf",
        energy_rtol=energy_rtol,
        residency=residency,
        evaluation="float64"
        if type(operator).action is not Hex8Operator.action
        else operator.evaluation,
        preconditioner_fft="cufft" if type(pcg) is _CudaPCG else "generated",
    )
    if norm <= threshold or norm == 0:
        if energy:
            current = float(macro @ operator.voigt @ macro)
            common.update(energy=current, energy_threshold=energy_rtol * current)
        if host:
            x.fill_zero()
            fresh(lambda u: None)
        return dict(
            solution=x,
            residual=pcg.r,
            iterations=0,
            restarts=0,
            audits=1,
            replacements=0,
            recursive_residual_inf=norm,
            residual_inf=norm,
            **common,
            **stored,
            **_energy_bound(operator, pcg.energy_norm()),
        )
    if pcg.mixed:
        pcg.increment.fill(0)
    pcg.reset_history(maxiter)
    pcg.energy = np.float64(energy_rtol if energy else 0.0)
    pcg.replace = np.float64(_RELIABLE_DROP**2 if pcg.mixed else 0.0)
    pcg.s[7] = float(macro @ operator.voigt @ macro)
    pcg.start(0, threshold)
    iteration, restarts, audits, replacements = 0, 0, 0, 0
    events, previous, progress = [], np.inf, FreshProgress()
    while True:
        status, iteration = pcg.iterate(x, iteration, maxiter, threshold)
        if status == 2:
            raise LinearSolveError(
                "HEX8 PCG lost positive curvature or unsafe step before physical acceptance",
                status="lost_curvature",
                iterations=iteration,
            )
        replacements += status == 3
        value, current = fresh(evaluate)
        if not np.isfinite(value):
            raise LinearSolveError(
                "Nonfinite fresh HEX8 residual",
                status="nonfinite",
                iterations=iteration,
            )
        audits += 1
        # The fresh residual replaces the recursive one. Its preconditioned
        # norm is the fresh energy bound G0; compared with the recursive rho it
        # measures the drift, which decides below between keeping the search
        # direction (reliable update) and restarting from p = z.
        bound = _energy_bound(operator, pcg.prepare(iteration, threshold, 2))
        recursive = float(pcg.history[6 * iteration + 2])
        if energy:
            limit = energy_rtol * (current - bound["energy_error_upper"])
            common.update(energy=current, energy_threshold=limit)
            accepted = bound["energy_error_upper"] <= limit
        else:
            accepted = value <= threshold
        if np.isfinite(value) and accepted:
            return dict(
                solution=x,
                residual=pcg.r,
                iterations=iteration,
                restarts=restarts,
                audits=audits,
                recursive_residual_inf=float(pcg.history[6 * iteration + 3]),
                replacements=replacements,
                residual_inf=value,
                events=events,
                history=pcg.read_history(iteration),
                **common,
                **bound,
                **stored,
            )
        if iteration >= maxiter:
            raise LinearSolveError(
                f"HEX8 failed fresh residual: {value:.6g} > {threshold:.6g} after {maxiter} updates ({restarts} restarts)",
                status="maxiter",
                iterations=iteration,
            )
        # Fresh residuals that stop improving are at the attainable accuracy of
        # the working precision: stop instead of auditing the budget away.
        progress.check(value, iteration)
        drift = abs(bound["energy_G0"] / recursive - 1) if recursive > 0 else np.inf
        if status == 3 and drift > 1e-2:
            # Recursion drifted between replacements: replace ten times as often.
            pcg.replace = np.float64(max(pcg.replace, 1e-4))
        # Keeping p is valid only for a small replacement (mixed precision gives
        # ~1e-4); an FP32 recursion that drifted far from the FP64 residual
        # (a failed FP32 audit) restarts, as does a candidate without progress.
        mode = 2 if drift <= 0.25 and value <= 0.5 * previous else 1
        pcg.proceed(mode)
        if mode == 1:
            restarts += 1
        events.append(
            (
                iteration,
                "replacement" if status == 3 else "resume" if mode == 2 else "restart",
            )
        )
        previous = value if np.isfinite(value) else np.inf


def _hex8_spectrum(op):
    """Estimate the preconditioner workspace and identify generated FFT use.

    Bytes of the preconditioner spectrum and whether it is generated (fused
    PCG); the cuFFT spectrum adds one vector for its plans' work areas.
    """
    from .hex8_fused import fused_hex8_config, fused_hex8_supported
    from .kernel_cache import device_properties

    n0, n1, n2 = op.shape
    w, limit = op.working.itemsize, device_properties()["sharedMemPerBlockOptin"]
    if fused_hex8_supported(op.shape, op.working, limit):
        cfg = fused_hex8_config(op.shape, op.working, limit)
        return 3 * n0 * n1 * cfg["tiles"] * cfg["T"] * 2 * w, True
    return 3 * n0 * n1 * (n2 // 2 + 1) * 2 * w + 3 * op.n * w, False


def hex8_working_bytes(op, residency="device"):
    """Estimate device bytes needed by a HEX8 load besides its operator.

    Device bytes of a HEX8 load beside the operator: the PCG workspace
    (r, p, Ap = z, the preconditioner spectrum and, mixed, the FP32 increment)
    and the solution. On the host levels the solution is on the device only
    during fresh evaluations, in the memory of z and the generated spectrum,
    beside two staging slabs.
    """
    from .coordinates import _SLAB_BYTES

    n3, w, d = 3 * op.n, op.working.itemsize, op.dtype.itemsize
    spectrum, fused = _hex8_spectrum(op)
    pcg = 3 * n3 * w + spectrum + (n3 * w if op.working != op.dtype else 0)
    if residency == "device":
        return pcg + n3 * d
    released = n3 * w + (spectrum if fused else 0)
    return max(pcg, pcg - released + n3 * d) + 2 * _SLAB_BYTES


def hex8_residency(op):
    """Select the first supported HEX8 residency that fits the device.

    The first residency of a HEX8 load that fits the device, or None:
    'device'; mixed 'host', the FP64 solution in pinned host memory; fixed
    'iterate', the solution, which is the Krylov iterate, there (it needs the
    fused PCG). The leading planes that fit stay on the device.
    """
    from .common import device_free_bytes

    free = device_free_bytes() - (512 << 20)
    for level in ("device", "host" if op.working != op.dtype else "iterate"):
        if level == "iterate" and not _hex8_spectrum(op)[1]:
            return None
        if hex8_working_bytes(op, level) < free:
            return level
    return None


def _hex8_resident_rows(op, pcg):
    """Estimate the leading solution planes that can remain device-resident.

    Planes (component, x0) of a host-resident solution kept on the device:
    the most that fit beside the PCG workspace and staging slabs, and beside
    the whole solution gathered in the memory z and the spectrum release.
    """
    from .common import device_free_bytes
    from .coordinates import _SLAB_BYTES

    n0, n1, n2 = op.shape
    row, whole = n1 * n2 * op.dtype.itemsize, 3 * op.n * op.dtype.itemsize
    free = device_free_bytes() - (512 << 20) - 2 * _SLAB_BYTES
    return int(
        max(0, min(3 * n0, free / row, (free + pcg.released_bytes() - whole) / row))
    )


def _energy_bound(op, g0):
    """Fresh-residual bracket of this load's energy error r^T K^+ r.

    G0 = r^T K0^+ r with the reference preconditioner; gamma_- C0 <= C_p <=
    gamma_+ C0 gives G0/gamma_+ <= error <= G0/gamma_-. gamma are numerical
    generalized eigenvalues, not interval certificates; diagnostic only.
    """
    if not np.isfinite(g0) or g0 < 0:
        raise LinearSolveError(
            "Invalid fresh preconditioned energy indicator",
            status="invalid_preconditioner",
        )
    return dict(
        energy_G0=g0,
        energy_error_upper=g0 / op.gamma_minus,
        energy_error_lower=g0 / op.gamma_plus,
    )


_HOST_CHUNK = 1 << 24


def _retain_host(u, dtype):
    """Pinned host copy of a device solution, rounded on the device in chunks."""
    import cupyx

    host = cupyx.empty_pinned(u.shape, dtype)
    flat, out = u.reshape(-1), host.reshape(-1)
    for start in range(0, flat.size, _HOST_CHUNK):
        flat[start : start + _HOST_CHUNK].astype(dtype).get(
            out=out[start : start + _HOST_CHUNK]
        )
    return host


def _stored_dot(op, stored, field):
    """FP64 dot of a retained (device or pinned host) solution with a device field."""
    if op.xp is np or isinstance(stored, cp.ndarray):
        return op.dot(stored, field)
    flat, other, total = stored.reshape(-1), field.reshape(-1), 0.0
    stage = cp.empty(min(_HOST_CHUNK, flat.size), flat.dtype)
    for start in range(0, flat.size, _HOST_CHUNK):
        width = min(_HOST_CHUNK, flat.size - start)
        stage[:width].set(flat[start : start + width])
        total += op.dot(stage[:width], other[start : start + width])
    return total


def homogenize_hex8(
    index,
    stiffness,
    *,
    lengths=(1.0, 1.0, 1.0),
    reference=None,
    backend="cupy",
    dtype="float64",
    rtol=1e-6,
    atol=0.0,
    maxiter=3000,
    storage="auto",
    return_fields=False,
    operator_type=None,
    working=None,
    energy_rtol=None,
    evaluation="df64",
):
    """Six-load full energy matrix with native physical audits.

    C_stress is the Gauss-averaged stress of each accepted solution. The
    energy matrix uses, for b <= a, C[a,b] = C_stress[b,a] - u_b . r_a with
    the fresh residual r_a = f_a - K u_a; in exact arithmetic this is the
    field energy (A+BU)^T W C (A+BU), symmetric, with an error R^T K^+ R
    quadratic in the residuals. Only earlier solutions are retained (rounded
    to FP32, which perturbs the tiny product u_b . r_a at ~1e-7 relative);
    residuals are never retained. Retained solutions use GPU memory when the
    budget allows, otherwise host memory. With storage='auto' a load that
    does not fit the device keeps its solution in pinned host memory
    (hex8_residency); the PCG workspace stays on the device. Output fields,
    if requested, are returned on the host at full precision. Timing belongs
    to the caller. No hidden fallback to GaNi.
    """
    if storage not in ("auto", "device", "host"):
        raise ValueError("storage must be auto, device or host")
    rtol, atol, maxiter = _solver_parameters(rtol, atol, maxiter)
    energy_rtol = energy_tolerance(energy_rtol)
    if energy_rtol is not None and backend != "cupy":
        raise ValueError("The energy criterion is implemented by the CUDA PCG")
    operator_type = Hex8Operator if operator_type is None else operator_type
    if not isinstance(operator_type, type) or not issubclass(
        operator_type, Hex8Operator
    ):
        raise ValueError("operator_type must implement the Hex8Operator contract")
    op = operator_type(
        index,
        stiffness,
        lengths=lengths,
        reference=reference,
        backend=backend,
        dtype=dtype,
        working=working,
        evaluation=evaluation,
    )
    xp = op.xp
    keep = op.dtype if return_fields else np.dtype(np.float32)
    required = 5 * 3 * op.n * keep.itemsize
    host = storage == "host" or xp is np
    residency, HostVector = "device", type(None)
    if xp is not np and storage == "auto":
        from .common import device_free_bytes
        from .coordinates import _HostVector as HostVector

        residency = hex8_residency(op)
        if residency != "device" and op.release_symbol_table():
            # Computing the FP64 reference inverse costs less than streaming
            # the solution: drop its table first, and keep it if that alone
            # does not bring the load back to the device.
            cp.get_default_memory_pool().free_all_blocks()
            lighter = hex8_residency(op)
            if lighter == "device" or residency is None:
                residency = lighter
            else:
                op.store_symbol_table()
        if residency is None:
            raise MemoryError(
                "The HEX8 load does not fit the available device/host residency budget; "
                "choose a different configuration or grid explicitly"
            )
        # Retained solutions use what the load leaves, with one vector of
        # slack for pool rounding (as many as fit); the host levels retain on
        # the host.
        working = hex8_working_bytes(op, residency) + 3 * op.n * op.working.itemsize
        slots = (
            0
            if residency != "device"
            else int(
                max(
                    0,
                    min(
                        required // (3 * op.n * keep.itemsize),
                        (device_free_bytes() - (256 << 20) - working)
                        // (3 * op.n * keep.itemsize),
                    ),
                )
            )
        )
        host = slots == 0
    else:
        slots = 0 if host else 6
    retained, infos = [], []
    csigma, correction = np.empty((6, 6)), np.zeros((6, 6))
    for a, macro in enumerate(np.eye(6)):
        try:
            info = (
                _solve_cuda(op, macro, rtol, atol, maxiter, energy_rtol, residency)
                if xp is not np
                else solve_hex8_load(op, macro, rtol=rtol, atol=atol, maxiter=maxiter)
            )
        except Exception as error:
            # The PCG workspace owns the operator too. Break that cycle on
            # rejected solves, as on success, before propagating the failure.
            op._pcg = None
            if isinstance(error, LinearSolveError):
                error.diagnostics.update(
                    load_id=a,
                    problem_precision=op.dtype.name,
                    working_precision=op.working.name,
                    accepted=False,
                )
            raise
        u, r = info.pop("solution"), info.pop("residual")
        if (
            "mean_stress" in info
        ):  # formed in the last evaluation of a host-resident solution
            csigma[:, a], product = info.pop("mean_stress"), info.pop("solution_dot")
        else:
            csigma[:, a] = (
                op.load_work_stress(u, macro)
                if xp is not np
                else op.mean_stress(u, macro)
            )
            product = op.dot(u, r)
        for b, ub in enumerate(retained):
            correction[a, b] = correction[b, a] = _stored_dot(op, ub, r)
        correction[a, a] = product
        if a < 5 or return_fields:
            if isinstance(u, HostVector):
                retained.append(u.rounded(keep).reshape(3, *op.shape))
            elif len(retained) >= slots and xp is not np:
                retained.append(_retain_host(u, keep))
            else:
                retained.append(u if u.dtype == keep else u.astype(keep))
        del u  # the next load allocates its solution only after this one is gone
        infos.append(info)
    # The cached PCG references the operator: break the cycle so that its
    # device workspace is released with this call, not at a later gc pass.
    op._pcg = None
    if not np.isfinite(csigma).all() or not np.isfinite(correction).all():
        raise RuntimeError("HEX8 mean stress or energy correction is nonfinite")
    lower = np.tril(csigma.T - correction)
    ceff = lower + np.tril(lower, -1).T
    eigenvalues = np.linalg.eigvalsh(ceff)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0:
        raise RuntimeError(
            "HEX8 effective energy matrix is nonfinite or not positive definite"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = np.max(np.abs(ceff))
        symmetry_defect = float(
            np.linalg.norm(csigma / scale - csigma.T / scale)
            / np.linalg.norm(ceff / scale)
        )
    if not np.isfinite(symmetry_defect):
        raise RuntimeError("HEX8 stress symmetry defect is nonfinite")
    # E = C - C* is PSD: |E_ab| <= sqrt(E_aa E_bb) and ||E||_2 <= trace(E).
    energy_bound = energy_report(
        ceff, op.spectral_bounds, [i.get("energy_G0") for i in infos]
    )
    energy_bound.update(gamma_minus=op.gamma_minus, gamma_plus=op.gamma_plus)
    result = dict(
        C=ceff,
        C_stress=csigma,
        energy_correction=correction,
        infos=infos,
        energy_error_bound=energy_bound,
        formulation="HEX8",
        implementation="CuFHom-HEX8",
        dtype=op.dtype.name,
        working=op.working.name,
        storage="host"
        if host
        else "device"
        if len(retained) <= slots
        else "device+host",
        retained_device=min(slots, len(retained)),
        residency=residency,
        shape=op.shape,
        symbol_table=op.xp is not np and not isinstance(op.symbol[-1], np.uint64),
        reference=op.reference.copy(),
        gamma_minus=op.gamma_minus,
        working_diagnostics=op.working_diagnostics,
        spectral_bounds=dict(op.spectral_bounds),
        symmetry_defect=symmetry_defect,
    )
    result["operator_type"] = type(op).__name__
    if return_fields:
        result["solutions"] = [
            np.asarray(u) if isinstance(u, np.ndarray) else cp.asnumpy(u)
            for u in retained
        ]
    return result
