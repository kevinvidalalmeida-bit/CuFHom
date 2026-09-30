"""GPU fiber-only polarization experiment; never imported by production.

Woodbury form of the unchanged discrete compatible linear problem:
  u0 = Gamma_m b; (I + D S* Gamma_m S D)p = D S* u0;
  u = u0 - Gamma_m S D p,  D=(C_f-C_m)^(1/2).
Unlike the macroscopic-load formula this also handles arbitrary compatible RHS.
"""
import time
import numpy as np
from pipeline.nonlinear_solver import MixedProjector, cp
from pipeline.nonlinear_reference import IsotropicReference
from ffthompy.general.cupy_warp_reduction import CupyWarpCGReduction
from ffthompy.general.solver import _finalize_cg_info

_KERNELS = {}
_CODE = r'''
extern "C" __global__ void macro_rhs(const int* ids,const int* selected,
 const float* table,float* y,long long nf,int load) {
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=nf)return;
 const float* D=table+36*(long long)ids[selected[j]];
 for(int a=0;a<6;a++)y[a*nf+j]=D[6*load+a];
}
extern "C" __global__ void scatter_root(const int* ids,const int* selected,
 const float* table,const float* x,float* y,long long n,long long nf) {
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=nf)return;
 long long s=selected[j]; const float* D=table+36*(long long)ids[s];
 for(int a=0;a<6;a++) {
   float sum=0;for(int b=0;b<6;b++)sum+=D[6*a+b]*x[b*nf+j];
   y[a*n+s]=sum;
 }
}
extern "C" __global__ void restrict_root(const int* ids,const int* selected,
 const float* table,const float* base,const float* x,const float* identity,float* y,
 long long n,long long nf,int add_identity,int transformed) {
 long long j=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(j>=nf)return;
 long long s=selected[j]; const float* D=table+36*(long long)ids[s];
 for(int a=0;a<6;a++) {
   float sum=(add_identity&&!transformed)?identity[a*nf+j]:0;
   if(add_identity&&transformed)for(int b=0;b<6;b++)
     sum+=base[36*(long long)ids[s]+6*a+b]*identity[b*nf+j];
   for(int b=0;b<6;b++)sum+=D[6*b+a]*x[b*n+s];
   y[a*nf+j]=sum;
 }
}
'''


class FiberPolarization:
    def __init__(self, A, lengths, bulk, shear, local=False):
        if A.val.dtype != cp.int32 or A.material_table.dtype != cp.float32:
            raise ValueError('Polarization experiment requires indexed float32 material')
        self.ids, self.shape = A.val, tuple(A.N)
        self.n = int(np.prod(self.shape))
        if self.n > np.iinfo(np.int32).max:
            raise ValueError('Experimental fiber offsets require int32 range')
        self.selected = cp.flatnonzero(self.ids.ravel() != 0).astype(cp.int32)
        self.nf = int(self.selected.size)
        table = cp.asnumpy(A.material_table).astype(np.float64)
        rows, cols = np.triu_indices(6)
        C = np.zeros((len(table), 6, 6)); C[:, rows, cols] = table; C[:, cols, rows] = table
        expected = 2*shear*np.eye(6); expected[:3, :3] += bulk-2*shear/3
        if not np.allclose(C[0], expected, rtol=2e-6, atol=1e-7):
            raise ValueError('Material zero must equal the isotropic matrix reference')
        roots = np.zeros_like(C)
        if len(C) > 1:
            w, v = np.linalg.eigh(C[1:]-C[0])
            if not np.isfinite(w).all() or w.min() <= 0:
                raise ValueError('Fiber-only SPD polarization requires positive material contrast')
            roots[1:] = (v*np.sqrt(w)[:, None, :])@v.transpose(0, 2, 1)
        self.local = bool(local)
        base = np.broadcast_to(np.eye(6), roots.shape).copy()
        if local:
            # Angular average of the isotropic Green symbol, not a new
            # discretization and not claimed to be the exact lattice diagonal.
            a = 1/(2*shear); b = 1/(bulk+4*shear/3)
            gdev = (6*a+2*b)/15; gvol = b/3
            G = gdev*np.eye(6); G[:3, :3] += (gvol-gdev)/3
            G *= (np.prod([n-(n%2==0) for n in self.shape])-1)/self.n
            diagonal = np.eye(6)+roots@G@roots
            w, v = np.linalg.eigh(diagonal)
            if not np.isfinite(w).all() or w.min() <= 0:
                raise ValueError('Local polarization preconditioner must be SPD')
            L = (v/np.sqrt(w)[:, None, :])@v.transpose(0, 2, 1)
            base = L@L
            roots = roots@L  # K=D L, so restriction MUST apply K transpose.
        self.roots = cp.asarray(roots, dtype=cp.float32)
        self.base = cp.asarray(base, dtype=cp.float32)
        self.buffer = cp.zeros((6,)+self.shape, cp.float32)
        self.P = MixedProjector(self.shape, lengths, np.ones(6, bool), cp)
        self.Gamma = IsotropicReference(self.P, bulk, shear)
        self.Gamma.set_shift(0., np.dtype('float32'))
        for name in ('scatter_root', 'restrict_root', 'macro_rhs'):
            if name not in _KERNELS:
                _KERNELS[name] = cp.RawKernel(_CODE, name)

    def scatter(self, x):
        if self.nf:
            _KERNELS['scatter_root'](((self.nf+255)//256,), (256,),
                (self.ids, self.selected, self.roots, x, self.buffer,
                 np.int64(self.n), np.int64(self.nf)))
        return self.buffer

    def restrict(self, x, identity=None):
        out = cp.empty(6*self.nf, cp.float32)
        if self.nf:
            _KERNELS['restrict_root'](((self.nf+255)//256,), (256,),
                (self.ids, self.selected, self.roots, self.base, x,
                 identity if identity is not None else out, out,
                 np.int64(self.n), np.int64(self.nf), np.int32(identity is not None), np.int32(self.local)))
        return out

    def action(self, p):
        return self.restrict(self.Gamma(self.scatter(p)), identity=p)

    def macro_rhs(self, load):
        if load not in range(6):
            raise ValueError('Macro load index must be in range(6)')
        out = cp.empty(6*self.nf, cp.float32)
        if self.nf:
            _KERNELS['macro_rhs'](((self.nf+255)//256,), (256,),
                (self.ids, self.selected, self.roots, out, np.int64(self.nf), np.int32(load)))
        return out


def make_polarization_solver(*, A, GN, pb, fallback, bulk, shear, cache, cg, macro=False, local=False):
    load_index = cache.get('_load_index') if macro else None
    if macro and load_index not in range(6):
        raise ValueError('Direct polarization requires an explicitly intercepted macro load')
    def solve(solver, Afun, B, x0, par, callback):
        if x0 is not None or callback is not None:
            raise ValueError('Polarization experiment requires zero start and no callback')
        if 'engine' not in cache:
            cp.cuda.Stream.null.synchronize(); started = time.perf_counter()
            engine = FiberPolarization(A, tuple(pb.Y), bulk, shear, local=local)
            cp.cuda.Stream.null.synchronize()
            cache.update(engine=engine, polarization_setup_s=time.perf_counter()-started,
                fiber_voxels=engine.nf, krylov_vector_bytes=6*engine.nf*4,
                persistent_buffer_bytes=engine.buffer.nbytes+engine.roots.nbytes+engine.base.nbytes+engine.selected.nbytes)
        engine = cache['engine']
        if engine.ids is not A.val:
            raise ValueError('Do not reuse polarization for another material field')
        reduction = CupyWarpCGReduction(B, max_blocks=16384)
        def dot(a, b):
            reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                (a, b, reduction.partial, np.int64(a.size)))
            return float(cp.sum(reduction.partial, dtype=cp.float64))
        volume = engine.n
        rhs_norm = np.sqrt(dot(B.val, B.val)/volume)
        threshold = max(par.get('atol', 0.), par['rtol']*rhs_norm)
        relative = threshold/rhs_norm if rhs_norm else par['rtol']
        if rhs_norm == 0:
            result, iterations = B.copy(val=cp.zeros_like(B.val)), 0
        else:
            if macro:
                rhs = engine.macro_rhs(load_index)
            else:
                u0 = engine.Gamma(B.val/np.float32(volume))
                rhs = engine.restrict(u0)
            p, iterations = cg(engine.action, rhs, dot, relative, par['maxiter'], cp)
            fluctuation = engine.Gamma(engine.scatter(p))
            result = B.copy(val=-fluctuation if macro else u0-fluctuation)
        residual = B.val-Afun(result).val
        norm = np.sqrt(dot(residual, residual)/volume)
        corrections = 0
        if not np.isfinite(norm) or norm > threshold:
            remaining = par['maxiter']-iterations
            if remaining <= 0:
                raise RuntimeError('Polarization exhausted iteration budget before physical convergence')
            result, info = fallback(solver=solver, Afun=Afun, B=B, x0=result,
                par=dict(par, rtol=par['rtol']*.5, atol=par.get('atol', 0.)*.5,
                         maxiter=remaining, keep_solution_on_device=True), callback=None)
            corrections = info['kit']
            residual = B.val-Afun(result).val
            norm = np.sqrt(dot(residual, residual)/volume)
        if not np.isfinite(norm) or norm > threshold:
            raise RuntimeError('Polarization failed physical residual audit')
        info = dict(kit=iterations+corrections, norm_res=norm, norm_res_log=[norm],
                    coordinate_iterations=iterations, correction_iterations=corrections)
        _finalize_cg_info(info, rhs_norm, threshold, par['maxiter'])
        info.update(true_norm_res_rel=info['norm_res_rel'], true_residual_converged=True)
        if not par.get('keep_solution_on_device', False):
            result = result.copy(val=cp.asnumpy(result.val))
        return result, info
    return solve
