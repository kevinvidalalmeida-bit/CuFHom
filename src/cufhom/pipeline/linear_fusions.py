"""CUDA source transformations for the two validated linear coordinate fusions.

The shared coordinate formula owns the mean/Nyquist conventions. Early returns
stay inside a device function so every thread reaches the reduction barrier.
"""


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise RuntimeError('Coordinate kernel changed: review linear fusions')
    return code.replace(old, new)


def direction_source(code):
    code = replace_once(code, 'const COMPLEX* x', 'COMPLEX* x')
    code = replace_once(code, 'REAL longitudinal) {',
        'REAL longitudinal, const COMPLEX* residual, REAL beta) {')
    return replace_once(code, 'if(s>=count)return;', r'''if(s>=count)return;
 // Update ALL coordinates before handling zero/Nyquist modes. Each thread
 // owns the same frequency in all three components; no inter-thread read.
 for(int c=0;c<3;c++) {
   const long long index=c*count+s;
   x[index]=MAKE(residual[index].x+beta*x[index].x,
                 residual[index].y+beta*x[index].y);
 }
''')


def adjoint_dot_source(code):
    code = replace_once(code, 'extern "C" __global__ void coordinates',
                         '__device__ __forceinline__ void coordinate_one')
    code = replace_once(code, 'REAL longitudinal) {',
                         'REAL longitudinal, const COMPLEX* p, double& energy) {')
    code = replace_once(code,
        'for(int c=0;c<(adjoint?3:6);c++)y[c*count+s]=out[c];', r'''
 for(int c=0;c<3;c++) {
   const long long index=c*count+s;
   y[index]=out[c];
   energy+=(double)p[index].x*out[c].x+(double)p[index].y*out[c].y;
 }
''')
    return code+r'''
extern "C" __global__ void adjoint_dot(const COMPLEX* x, COMPLEX* y,
 long long n0,long long n1,long long n2,long long nz,
 REAL iy0,REAL iy1,REAL iy2,REAL norm,int free_mask,int adjoint,
 REAL transverse,REAL longitudinal,const COMPLEX* p,double* partial) {
 double energy=0;
 coordinate_one(x,y,n0,n1,n2,nz,iy0,iy1,iy2,norm,free_mask,1,transverse,longitudinal,p,energy);
 for(int offset=16;offset;offset>>=1)energy+=__shfl_down_sync(0xffffffff,energy,offset);
 __shared__ double warps[8];
 const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)warps[warp]=energy;
 __syncthreads();
 if(warp==0) {
   energy=lane<8?warps[lane]:0;
   for(int offset=16;offset;offset>>=1)energy+=__shfl_down_sync(0xffffffff,energy,offset);
   if(lane==0)partial[blockIdx.x]=energy;
 }
}
'''
