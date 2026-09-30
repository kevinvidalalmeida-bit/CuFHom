"""Isolated linear experiments. Nothing here is imported by production solvers.

Use candidate() only in a single-threaded benchmark/test. Its temporary patches
reuse the production original-equation audit and restore all entry points.
"""
from contextlib import contextmanager, ExitStack
import time
from unittest.mock import patch
import numpy as np
from pipeline import linear_coordinates as linear
from pipeline.nonlinear_coordinates import DisplacementCoordinates, _CODE
from pipeline.nonlinear_solver import _cg, _cg_updates, NonlinearConvergenceError, cp

_KERNELS = {}


class SymmetricCoordinates(DisplacementCoordinates):
    """QR and RQ*, R=M^-1/2, with R fused into the coordinate kernel."""
    def __init__(self, projector, bulk, shear):
        super().__init__(projector)
        if self.P.free:
            raise ValueError('Experiment supports fixed means only')
        self.transverse = np.sqrt(1/(2*shear))
        self.longitudinal = np.sqrt(1/(bulk+4*shear/3))

    def root(self, value):
        k, _, _ = self._cpu_geometry()
        v = value.reshape((3,)+self.hshape)
        return (self.transverse*v+(self.longitudinal-self.transverse)*k*np.sum(k*v, axis=0)).ravel()

    def _map(self, x, adjoint):
        if self.xp is np:
            return self.root(super()._map(x, True)) if adjoint else super()._map(self.root(x), False)
        out = cp.empty(self.size if adjoint else (6,)+self.hshape, dtype=x.dtype)
        key = ('root', x.dtype.str)
        if key not in _KERNELS:
            code = _CODE.replace('int adjoint)', 'int adjoint, REAL transverse, REAL longitudinal)')
            old = 'v[c]=r2*v[c]+(1-r2)*k[c]*dot;'
            assert code.count(old) == 1
            code = code.replace(old, 'v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;')
            name = 'float' if x.dtype == cp.complex64 else 'double'
            code = code.replace('COMPLEX', name+'2').replace('REAL', name).replace('MAKE', 'make_'+name+'2')
            _KERNELS[key] = cp.RawKernel(code, 'coordinates')
        d = x.real.dtype.type
        _KERNELS[key](((self.count+255)//256,), (256,),
            (x, out, *(np.int64(n) for n in self.shape), np.int64(self.hshape[-1]),
             *(d(1/L) for L in self.P.lengths), d(self.norm), np.int32(0), np.int32(adjoint),
             d(self.transverse), d(self.longitudinal)))
        return out


_UPDATE = r'''
extern "C" __global__ void update_rr(const float* p, const float* ap,
 float alpha, float* x, float* r, double* partial, long long count) {
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     i<count;i+=(long long)gridDim.x*blockDim.x) {
   x[i]+=alpha*p[i];
   float ri=r[i]-alpha*ap[i]; r[i]=ri;
   sum+=(double)ri*(double)ri;
 }
 for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
 __shared__ double sums[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(lane==0)sums[warp]=sum;
 __syncthreads();
 if(warp==0) {
   sum=lane<8?sums[lane]:0;
   for(int offset=16;offset;offset>>=1)sum+=__shfl_down_sync(0xffffffff,sum,offset);
   if(lane==0)partial[blockIdx.x]=sum;
 }
}
'''


def fused_update(p, ap, alpha, x, r, partial):
    if 'update' not in _KERNELS:
        _KERNELS['update'] = cp.RawKernel(_UPDATE, 'update_rr')
    _KERNELS['update']((partial.size,), (256,),
        (p, ap, np.float32(alpha), x, r, partial, np.int64(r.nbytes//4)))
    return float(cp.sum(partial, dtype=cp.float64))


def fused_cg(action, rhs, dot, tolerance, maxiter, xp, project=None, fused=False,
             precondition=None, initial=None):
    """Classical PCG, replacing only x/r update + norm by one FP64-sum kernel."""
    if xp is not cp or rhs.dtype not in (cp.complex64, cp.float32) or project is not None or initial is not None:
        raise ValueError('Fused experiment requires zero-start complex64/float32 GPU vectors')
    x = cp.zeros_like(rhs); r = rhs.copy()
    rr = dot(r, r); limit = tolerance*tolerance*rr
    if rr == 0:
        return x, 0
    partial = cp.empty(min(16384, (rhs.nbytes//4+255)//256), dtype=cp.float64)
    z = precondition(r) if precondition else r
    rz = dot(r, z) if precondition else rr
    p = z.copy()
    for i in range(maxiter):
        ap = action(p); pap = dot(p, ap)
        if not np.isfinite([pap, rz]).all() or min(pap, rz) <= 0:
            raise NonlinearConvergenceError('Fused CG lost positivity')
        new = fused_update(p, ap, rz/pap, x, r, partial)
        if not np.isfinite(new):
            raise NonlinearConvergenceError('Nonfinite fused residual')
        if new <= limit:
            r = rhs-action(x)
            new = dot(r, r)
            if new <= limit:
                return x, i+1
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            p = z.copy()
        else:
            z = precondition(r) if precondition else r
            next_rz = dot(r, z) if precondition else new
            _cg_updates()[1](z, np.float64(next_rz/rz), p)
        rz = next_rz
    raise NonlinearConvergenceError('Fused CG failed true residual')


_ANISO_BUILD = r'''
extern "C" __global__ void build_inverse(const double* C, float* M,
 long long n0,long long n1,long long n2,long long nz,double iy0,double iy1,double iy2) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 if(s==0||(n0%2==0&&i0==n0/2)||(n1%2==0&&i1==n1/2)||(n2%2==0&&i2==n2/2)) {
   for(int a=0;a<6;a++)M[a*count+s]=0;return;
 }
 double n[3]={(i0<=n0/2?i0:i0-n0)*iy0,(i1<=n1/2?i1:i1-n1)*iy1,i2*iy2};
 double inv=1/sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
 for(int a=0;a<3;a++)n[a]*=inv;
 const double r2=1.4142135623730950488;
 double S[18];
 for(int j=0;j<3;j++) {
   double v[3];
   for(int a=0;a<3;a++)v[a]=r2*(a==j)+(1-r2)*n[a]*n[j];
   for(int a=0;a<3;a++)S[3*a+j]=n[a]*v[a];
   S[9+j]=(n[1]*v[2]+n[2]*v[1])/r2;
   S[12+j]=(n[0]*v[2]+n[2]*v[0])/r2;
   S[15+j]=(n[0]*v[1]+n[1]*v[0])/r2;
 }
 double CS[18];
 for(int a=0;a<6;a++)for(int j=0;j<3;j++) {
   double sum=0;for(int b=0;b<6;b++)sum+=C[6*a+b]*S[3*b+j];
   CS[3*a+j]=sum;
 }
 double H[9];
 for(int i=0;i<3;i++)for(int j=0;j<3;j++) {
   double sum=0;for(int a=0;a<6;a++)sum+=S[3*a+i]*CS[3*a+j];H[3*i+j]=sum;
 }
 double a=H[0],b=H[1],c=H[2],d=H[4],e=H[5],f=H[8];
 double det=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
 M[s]=(d*f-e*e)/det; M[count+s]=(c*e-b*f)/det;
 M[2*count+s]=(b*e-c*d)/det; M[3*count+s]=(a*f-c*c)/det;
 M[4*count+s]=(b*c-a*e)/det; M[5*count+s]=(a*d-b*b)/det;
}
extern "C" __global__ void apply_inverse(const float* M,const float2* x,float2* y,long long count) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(s>=count)return;
 float a=M[s],b=M[count+s],c=M[2*count+s],d=M[3*count+s],e=M[4*count+s],f=M[5*count+s];
 float2 x0=x[s],x1=x[count+s],x2=x[2*count+s];
 y[s]=make_float2(a*x0.x+b*x1.x+c*x2.x,a*x0.y+b*x1.y+c*x2.y);
 y[count+s]=make_float2(b*x0.x+d*x1.x+e*x2.x,b*x0.y+d*x1.y+e*x2.y);
 y[2*count+s]=make_float2(c*x0.x+e*x1.x+f*x2.x,c*x0.y+e*x1.y+f*x2.y);
}
'''


def mean_stiffness(A):
    weights = cp.bincount(A.val.ravel(), minlength=A.material_table.shape[0])
    packed = cp.asnumpy(weights.astype(cp.float64) @ A.material_table.astype(cp.float64))/A.val.size
    i, j = np.triu_indices(6)
    C = np.zeros((6, 6)); C[i, j] = packed; C[j, i] = packed
    if not np.isfinite(C).all() or np.linalg.eigvalsh(C).min() <= 0:
        raise ValueError('Mean reference is not SPD')
    return C


class AnisotropicCoordinates(DisplacementCoordinates):
    def __init__(self, projector, C, cache):
        super().__init__(projector)
        if self.P.free or self.xp is not cp:
            raise ValueError('Anisotropic experiment requires fixed means on GPU')
        if 'inverse' not in cache:
            cp.cuda.Stream.null.synchronize(); started = time.perf_counter()
            if 'aniso_build' not in _KERNELS:
                _KERNELS['aniso_build'] = cp.RawKernel(_ANISO_BUILD, 'build_inverse')
                _KERNELS['aniso_apply'] = cp.RawKernel(_ANISO_BUILD, 'apply_inverse')
            self.inverse = cp.empty((6, self.count), dtype=cp.float32)
            _KERNELS['aniso_build'](((self.count+127)//128,), (128,),
                (cp.asarray(C), self.inverse, *(np.int64(n) for n in self.shape),
                 np.int64(self.hshape[-1]), *(np.float64(1/L) for L in self.P.lengths)))
            if not bool(cp.isfinite(self.inverse).all()):
                raise ValueError('Nonfinite anisotropic inverse')
            cp.cuda.Stream.null.synchronize()
            cache.update(inverse=self.inverse, reference_build_s=time.perf_counter()-started,
                         reference_bytes=self.inverse.nbytes, reference_C=C.tolist())
        self.inverse = cache['inverse']

    def precondition(self, value, reference):
        out = cp.empty_like(value)
        _KERNELS['aniso_apply'](((self.count+255)//256,), (256,),
            (self.inverse, value, out, np.int64(self.count)))
        return out


@contextmanager
def candidate(mode, reference_ratio=None):
    """One context per six-load solve: no hidden cross-solve reference caching."""
    if mode not in ('baseline', 'production', 'symmetric', 'fused_updates', 'combined', 'anisotropic',
                    'polarization', 'polarization_macro', 'polarization_local'):
        raise ValueError(mode)
    if reference_ratio is not None:
        if mode != 'combined' or not np.isfinite(reference_ratio) or not 0 < reference_ratio < 1.5:
            raise ValueError('Reference ratio requires combined and 0 < ratio < 1.5')
    records, cache = [], {}
    original_factory = linear.make_linear_solver
    def factory(**kwargs):
        if mode.startswith('polarization'):
            from linear_polarization_candidate import make_polarization_solver
            solve = make_polarization_solver(**kwargs, cache=cache, cg=fused_cg,
                                            macro=mode != 'polarization', local=mode == 'polarization_local')
        else:
            solve = original_factory(**kwargs, preconditioning=None if mode == 'production' else 'matrix')
        def run(**parameters):
            with ExitStack() as stack:
                if mode in ('symmetric', 'combined'):
                    bulk = kwargs['bulk']
                    if reference_ratio is not None:
                        bulk = 2*kwargs['shear']/reference_ratio-4*kwargs['shear']/3
                        cache['reference_ratio'] = reference_ratio
                    stack.enter_context(patch.object(linear, 'DisplacementCoordinates',
                        lambda P: SymmetricCoordinates(P, bulk, kwargs['shear'])))
                    def cg(*args, **kw):
                        kw['precondition'] = None
                        inner = fused_cg if mode == 'combined' else _cg
                        return inner(*args, **kw)
                    stack.enter_context(patch.object(linear, '_cg', cg))
                elif mode == 'fused_updates':
                    stack.enter_context(patch.object(linear, '_cg', fused_cg))
                elif mode == 'anisotropic':
                    if 'C' not in cache:
                        cp.cuda.Stream.null.synchronize(); started = time.perf_counter()
                        cache['C'] = mean_stiffness(kwargs['A'])
                        cache['reference_mean_s'] = time.perf_counter()-started
                    stack.enter_context(patch.object(linear, 'DisplacementCoordinates',
                        lambda P: AnisotropicCoordinates(P, cache['C'], cache)))
                result, info = solve(**parameters)
                records.append({k: info[k] for k in ('coordinate_iterations',
                    'correction_iterations', 'true_norm_res_rel', 'true_residual_converged')})
                return result, info
        return run
    with ExitStack() as stack:
        stack.enter_context(patch.object(linear, 'make_linear_solver', factory))
        if mode in ('polarization_macro', 'polarization_local'):
            from ffthompy import applications
            original_load = applications.solve_load_elasticity
            def load(iL, *args, **kwargs):
                cache['_load_index'] = int(iL)
                return original_load(iL, *args, **kwargs)
            stack.enter_context(patch.object(applications, 'solve_load_elasticity', load))
        yield records, cache
