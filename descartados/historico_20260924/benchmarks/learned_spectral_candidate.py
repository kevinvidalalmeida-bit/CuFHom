"""Experimental fixed SPD Fourier-block correction; not a production import.

Fit B r ~= e from exact operator pairs r=H e. H is the existing matrix-root
coordinate operator. Symmetric least squares + eigenvalue clipping ensures a
fixed SPD preconditioner. No material law or discretization is learned.
"""
from contextlib import contextmanager, ExitStack
from unittest.mock import patch
import numpy as np
from pipeline import linear_coordinates as linear
from pipeline.linear_krylov import SymmetricCoordinates
from pipeline.nonlinear_coordinates import _CODE
from pipeline.nonlinear_solver import cp

NBINS = 8*4*8
_KERNELS = {}
_GROUP = r'''
extern "C" __global__ void groups(short* out,long long n0,long long n1,long long n2,
 long long nz,double iy0,double iy1,double iy2) {
 long long s=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=n0*n1*nz;
 if(s>=count)return;
 long long i2=s%nz,i1=(s/nz)%n1,i0=s/(nz*n1);
 if(s==0||(n0%2==0&&i0==n0/2)||(n1%2==0&&i1==n1/2)||(n2%2==0&&i2==n2/2)) {
   out[s]=-1;return;
 }
 double x=(i0<=n0/2?i0:i0-n0)*iy0,y=(i1<=n1/2?i1:i1-n1)*iy1,z=i2*iy2;
 if(z==0&&(y<0||(y==0&&x<0))){x=-x;y=-y;}
 double r=sqrt(x*x+y*y+z*z);
 double a=(n0/2)*iy0,b=(n1/2)*iy1,c=(n2/2)*iy2;
 double maximum=sqrt(a*a+b*b+c*c);
 int rad=min(7,(int)(8*log1p(r)/log1p(maximum)));
 int elev=min(3,(int)(4*z/r));
 int az=min(7,(int)(8*(atan2(y,x)+3.14159265358979323846)/6.28318530717958647692));
 out[s]=(rad*4+elev)*8+az;
}
'''


def make_groups(shape, lengths):
    if 'groups' not in _KERNELS:
        _KERNELS['groups'] = cp.RawKernel(_GROUP, 'groups')
    nz = shape[-1]//2+1
    count = int(shape[0]*shape[1]*nz)
    out = cp.empty(count, cp.int16)
    _KERNELS['groups'](((count+255)//256,), (256,),
        (out, *(np.int64(n) for n in shape), np.int64(nz), *(np.float64(1/L) for L in lengths)))
    return out


def fit_spd(covariance, cross, ridge=1e-4, lower=.01, upper=10.):
    """Symmetric least squares: B S + S B = T + T.T, regularized to I."""
    if not 0 < lower <= upper or ridge <= 0:
        raise ValueError('Positive clipping bounds and ridge are required')
    if not np.isfinite(covariance).all() or not np.isfinite(cross).all():
        raise ValueError('Nonfinite fitting data')
    blocks = np.tile(np.eye(3), (len(covariance), 1, 1))
    for j, (S, T) in enumerate(zip(covariance, cross)):
        scale = np.trace(S)/3
        if scale <= np.finfo(float).tiny:
            continue
        S = (S+S.T)*.5/scale+ridge*np.eye(3)
        T = T/scale+ridge*np.eye(3)
        w, v = np.linalg.eigh(S)
        if w.min() <= 0:
            raise ValueError('Invalid covariance')
        B = v @ ((v.T@(T+T.T)@v)/(w[:, None]+w[None, :])) @ v.T
        w, v = np.linalg.eigh((B+B.T)*.5)
        blocks[j] = (v*np.clip(w, lower, upper))@v.T
    return blocks


class OperatorSamples:
    def __init__(self, samples_per_bin=48, stride=4, last_action=36):
        self.samples_per_bin, self.stride, self.last_action = samples_per_bin, stride, last_action
        self.covariance = np.zeros((NBINS, 3, 3))
        self.cross = np.zeros_like(self.covariance)
        self.pairs = 0
        self.shape = None

    def prepare(self, shape, lengths):
        if self.shape is not None:
            if tuple(shape) != self.shape or tuple(lengths) != self.lengths:
                raise ValueError('Training geometry changed')
            return
        self.shape, self.lengths = tuple(shape), tuple(lengths)
        groups = cp.asnumpy(make_groups(shape, lengths))
        self.count = len(groups)
        order = np.argsort(groups, kind='stable')
        sorted_groups = groups[order]
        rng = np.random.default_rng(510)
        selections = []
        for j in range(NBINS):
            start, end = np.searchsorted(sorted_groups, [j, j+1])
            if end > start:
                selections.append(order[rng.choice(np.arange(start, end),
                    size=min(end-start, self.samples_per_bin), replace=False)])
        selected = np.concatenate(selections) if selections else np.empty(0, np.int64)
        self.labels = groups[selected].astype(np.int64)
        self.selected = cp.asarray(selected, dtype=cp.int64)

    def add(self, residual, error):
        r = cp.asnumpy(residual.reshape(3, self.count)[:, self.selected]).astype(np.complex128)
        e = cp.asnumpy(error.reshape(3, self.count)[:, self.selected]).astype(np.complex128)
        scale = np.sum(np.abs(r)**2)
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            return
        for i in range(3):
            for j in range(3):
                self.covariance[:, i, j] += np.bincount(self.labels,
                    weights=(r[i]*r[j].conj()).real/scale, minlength=NBINS)
                self.cross[:, i, j] += np.bincount(self.labels,
                    weights=(e[i]*r[j].conj()).real/scale, minlength=NBINS)
        self.pairs += 1

    def model(self):
        return dict(shape=list(self.shape), lengths=list(self.lengths),
            blocks=fit_spd(self.covariance, self.cross).tolist(), pairs=self.pairs,
            sampled_frequencies=int(self.selected.size),
            scope='Fixed SPD spectral inverse correction fitted to H e -> e pairs; not a surrogate solution.')


@contextmanager
def collect_samples(samples):
    original_factory, original_cg = linear.make_linear_solver, linear.fused_cg
    def factory(**kwargs):
        samples.prepare(tuple(kwargs['A'].N), tuple(kwargs['pb'].Y))
        return original_factory(**kwargs)
    def cg(action, *args, **kwargs):
        calls = [0]
        def observed(x):
            y = action(x)
            if calls[0] < samples.last_action and calls[0] % samples.stride == 0:
                samples.add(y, x)
            calls[0] += 1
            return y
        return original_cg(observed, *args, **kwargs)
    with patch.object(linear, 'make_linear_solver', factory), patch.object(linear, 'fused_cg', cg):
        yield


class LearnedCoordinates(SymmetricCoordinates):
    def __init__(self, projector, bulk, shear, model, cache, blend=1.):
        super().__init__(projector, bulk, shear)
        if not 0 <= blend <= 1 or tuple(model['shape']) != self.shape or tuple(model['lengths']) != tuple(projector.lengths):
            raise ValueError('Invalid blend or incompatible spectral model geometry')
        if self.xp is not cp:
            raise ValueError('Learned spectral prototype is GPU-only')
        if 'groups' not in cache:
            B = np.asarray(model['blocks'], dtype=np.float64)
            if B.shape != (NBINS, 3, 3) or not np.isfinite(B).all() or not np.allclose(B, B.transpose(0, 2, 1), atol=1e-12):
                raise ValueError('Model blocks must be finite symmetric 3x3 matrices')
            w, v = np.linalg.eigh(B)
            if w.min() <= 0:
                raise ValueError('Model blocks must be SPD')
            w = 1-blend+blend*w
            roots = (v*np.sqrt(w)[:, None, :])@v.transpose(0, 2, 1)
            cache.update(groups=make_groups(self.shape, projector.lengths),
                         roots=cp.asarray(roots, dtype=cp.float32), blend=blend)
        self.groups, self.roots = cache['groups'], cache['roots']

    def _map(self, x, adjoint):
        if x.dtype != cp.complex64:
            raise ValueError('Learned spectral prototype requires complex64 coordinates')
        if 'learned' not in _KERNELS:
            code = _CODE.replace('int adjoint)',
                'int adjoint, const short* groups, const REAL* roots, REAL transverse, REAL longitudinal)')
            old = 'v[c]=r2*v[c]+(1-r2)*k[c]*dot;'
            if code.count(old) != 1:
                raise RuntimeError('Review changed coordinate kernel')
            code = code.replace(old, 'v[c]=r2*transverse*v[c]+(longitudinal-r2*transverse)*k[c]*dot;')
            code = code.replace('   REAL dot=k[0]', r'''
   const REAL* S=roots+9*groups[s];
   if(!adjoint) {
     REAL t[3];for(int a=0;a<3;a++)t[a]=S[3*a]*v[0]+S[3*a+1]*v[1]+S[3*a+2]*v[2];
     for(int a=0;a<3;a++)v[a]=t[a];
   }
   REAL dot=k[0]''')
            code = code.replace('   if(adjoint) {for(int c=0;c<3;c++)p[c]=v[c];}', r'''
   if(adjoint) {for(int c=0;c<3;c++)p[c]=S[3*c]*v[0]+S[3*c+1]*v[1]+S[3*c+2]*v[2];}''')
            code = code.replace('COMPLEX', 'float2').replace('REAL', 'float').replace('MAKE', 'make_float2')
            _KERNELS['learned'] = cp.RawKernel(code, 'coordinates')
        out = cp.empty(self.size if adjoint else (6,)+self.hshape, dtype=x.dtype)
        _KERNELS['learned'](((self.count+255)//256,), (256,),
            (x, out, *(np.int64(n) for n in self.shape), np.int64(self.hshape[-1]),
             *(np.float32(1/L) for L in self.P.lengths), np.float32(self.norm), np.int32(0),
             np.int32(adjoint), self.groups, self.roots, np.float32(self.transverse), np.float32(self.longitudinal)))
        return out


@contextmanager
def learned_candidate(model, blend=1.):
    cache, records = {}, []
    original_factory = linear.make_linear_solver
    def factory(**kwargs):
        solve = original_factory(**kwargs)
        def run(**parameters):
            result, info = solve(**parameters)
            records.append({k:info[k] for k in ('coordinate_iterations', 'correction_iterations',
                'true_norm_res_rel', 'true_residual_converged')})
            return result, info
        return run
    with ExitStack() as stack:
        stack.enter_context(patch.object(linear, 'SymmetricCoordinates',
            lambda P, K, mu: LearnedCoordinates(P, K, mu, model, cache, blend)))
        stack.enter_context(patch.object(linear, 'make_linear_solver', factory))
        yield records, cache
