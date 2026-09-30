"""Bounded, single-operator linear recycling experiments; never a production import.

Real inner products on complex RFFT coordinates preserve the admissible real
subspace. Small Gramians and all dot products accumulate in FP64. The original
linear adapter retains its independent physical-equation audit and fallback.
Use the patch context only in a dedicated, single-threaded benchmark process.
"""
from contextlib import contextmanager, ExitStack
import time
from unittest.mock import patch
import numpy as np
from pipeline import linear_coordinates as linear
from pipeline.linear_krylov import fused_cg
from pipeline.nonlinear_solver import cp, _cg

_KERNELS = {}
_CODE = r'''
extern "C" __global__ void project(const float* U, const float* x,
 double* partial, long long n) {
 int row=blockIdx.y;
 double sum=0;
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     i<n;i+=(long long)gridDim.x*blockDim.x)
   sum+=(double)U[(long long)row*n+i]*(double)x[i];
 for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
 __shared__ double sums[8];
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 if(!lane)sums[warp]=sum;
 __syncthreads();
 if(!warp) {
   sum=lane<8?sums[lane]:0;
   for(int o=16;o;o>>=1)sum+=__shfl_down_sync(0xffffffff,sum,o);
   if(!lane)partial[(long long)row*gridDim.x+blockIdx.x]=sum;
 }
}
extern "C" __global__ void combine(const float* U, const double* c,
 const float* base, float* out, long long n, int rank, int add) {
 for(long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
     i<n;i+=(long long)gridDim.x*blockDim.x) {
   double v=add?(double)base[i]:0.;
   for(int j=0;j<rank;j++)v+=(double)U[(long long)j*n+i]*c[j];
   out[i]=(float)v;
 }
}
'''


class BasisOps:
    """Contiguous rows, each one a full coordinate vector; no large FP64 copies."""
    def __init__(self, xp):
        self.xp = xp
        self.partial = None

    def project(self, rows, x):
        if self.xp is np:
            return np.array([np.vdot(v, x).real for v in rows], dtype=np.float64)
        if rows.dtype not in (cp.float32, cp.complex64) or not rows.flags.c_contiguous:
            raise ValueError('GPU basis requires contiguous float32/complex64 rows')
        if 'project' not in _KERNELS:
            _KERNELS['project'] = cp.RawKernel(_CODE, 'project')
        rank, n = rows.shape[0], x.nbytes//4
        blocks = min(2048, (n+255)//256)
        if self.partial is None or self.partial.shape != (rank, blocks):
            self.partial = cp.empty((rank, blocks), cp.float64)
        _KERNELS['project']((blocks, rank), (256,),
            (rows, x, self.partial, np.int64(n)))
        return cp.asnumpy(cp.sum(self.partial, axis=1, dtype=cp.float64))

    def combine(self, rows, coefficients, base=None):
        if self.xp is np:
            out = np.asarray(coefficients) @ rows
            return out if base is None else out+base
        if 'combine' not in _KERNELS:
            _KERNELS['combine'] = cp.RawKernel(_CODE, 'combine')
        out = cp.empty_like(rows[0]); n = out.nbytes//4
        _KERNELS['combine']((min(16384, (n+255)//256),), (256,),
            (rows, cp.asarray(coefficients, dtype=cp.float64), out if base is None else base,
             out, np.int64(n), np.int32(len(rows)), np.int32(base is not None)))
        return out

    def gram(self, left, right):
        return np.column_stack([self.project(left, row) for row in right])


def spd_inverse(a):
    a = (a+a.T)/2
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite small Gram matrix')
    np.linalg.cholesky(a)
    return np.linalg.solve(a, np.eye(len(a)))


class CoarseSpace:
    """Fixed SPD spectral/balanced corrections; A itself is never approximated."""
    def __init__(self, rows, arows, ops):
        self.rows, self.arows, self.ops = rows, arows, ops
        self.G = ops.gram(rows, rows)
        self.H = ops.gram(rows, arows)
        self.Ginv, self.Hinv = spd_inverse(self.G), spd_inverse(self.H)
        self.D = self.Hinv-self.Ginv

    def seed(self, rhs):
        return self.ops.combine(self.rows, self.Hinv@self.ops.project(self.rows, rhs))

    def spectral(self, rhs):
        # I-U(U*U)^-1 U* + U(U*AU)^-1 U* is SPD for full-rank U.
        return self.ops.combine(self.rows, self.D@self.ops.project(self.rows, rhs), rhs)

    def balanced(self, rhs):
        # B=U H^-1 U* + (I-U H^-1 (AU)*)(I-AU H^-1 U*).
        c = self.Hinv@self.ops.project(self.rows, rhs)
        t = self.ops.combine(self.arows, -c, rhs)
        return self.ops.combine(self.rows, c-self.Hinv@self.ops.project(self.arows, t), t)


class Builder:
    """Capture at most m early CG search directions and their small products.

    No complete Krylov history and no copies of every Ap are retained. True AU
    is evaluated once for each selected Ritz vector, inside the measured solve.
    """
    def __init__(self, window, rank, which='high', xp=np, rolling=False):
        if not 1 <= rank <= window or which not in ('low', 'high'):
            raise ValueError('Require 1 <= rank <= window and low/high selection')
        self.window, self.rank, self.which = window, rank, which
        self.xp, self.ops = xp, BasisOps(xp)
        self.rolling = rolling
        if rolling and rank >= window:
            raise ValueError('Rolling basis requires rank < window')
        self.restarts = 0
        self.rows = None
        self.G, self.H = np.zeros((window, window)), np.zeros((window, window))
        self.count = 0
        self.capture_s = 0.
        self.info = {}

    def capture(self, p, ap, dot):
        if self.count >= self.window and not self.rolling:
            return
        started = time.perf_counter()
        if self.count >= self.window:
            self.compress()
        norm = np.sqrt(dot(p, p))
        if not np.isfinite(norm) or norm == 0:
            return
        if self.rows is None:
            self.rows = self.xp.empty((self.window, p.size), dtype=p.dtype)
        j = self.count
        self.rows[j] = p/norm
        rows = self.rows[:j+1]
        self.G[:j+1, j] = self.ops.project(rows, rows[j])
        self.H[:j+1, j] = self.ops.project(rows, ap)/norm
        self.G[j, :j+1] = self.G[:j+1, j]
        self.H[j, :j+1] = self.H[:j+1, j]
        self.count += 1
        self.capture_s += time.perf_counter()-started

    def coefficients(self, m, rank):
        G, H = self.G[:m, :m], self.H[:m, :m]
        values, vectors = np.linalg.eigh((G+G.T)/2)
        cutoff = (1e-7 if self.xp is cp else 1e-12)*max(values[-1], 1.)
        keep = values > cutoff
        W = vectors[:, keep]/np.sqrt(values[keep])
        if W.shape[1] == 0:
            raise ValueError('No independent captured directions')
        small = W.T@H@W
        eigenvalues, eigenvectors = np.linalg.eigh((small+small.T)/2)
        k = min(rank, len(eigenvalues))
        select = np.arange(k) if self.which == 'low' else np.arange(len(eigenvalues)-k, len(eigenvalues))
        return W@eigenvectors[:, select], eigenvalues[select]

    def compress(self):
        # Thick restart: retain a few Ritz directions, then add fresh CG
        # operator arguments. This also admits a final true-residual argument;
        # Rayleigh-Ritz does not require the vectors to be CG directions.
        coefficients, _ = self.coefficients(self.count, min(self.rank+2, self.window-1))
        k = coefficients.shape[1]
        selected = self.xp.empty((k, self.rows.shape[1]), dtype=self.rows.dtype)
        H = coefficients.T@self.H[:self.count, :self.count]@coefficients
        for j in range(k):
            selected[j] = self.ops.combine(self.rows[:self.count], coefficients[:, j])
        self.rows[:k] = selected
        self.G[:k, :k] = self.ops.gram(selected, selected)
        self.H[:k, :k] = H
        self.count = k
        self.restarts += 1

    def finish(self, action, cg_iterations):
        started = time.perf_counter()
        # If convergence preceded the full window, the last recorded action was
        # the true-residual audit, not a search direction: do not use it.
        m = self.count if self.rolling else min(self.count, cg_iterations)
        if not m:
            self.rows = None
            self.info = dict(rank=0, captured=0, capture_s=self.capture_s, setup_s=0.)
            return None
        coefficients, eigenvalues = self.coefficients(m, self.rank)
        k = len(eigenvalues)
        U = self.xp.empty((k, self.rows.shape[1]), dtype=self.rows.dtype)
        for j in range(k):
            U[j] = self.ops.combine(self.rows[:m], coefficients[:, j])
        captured_bytes = self.rows.nbytes
        self.rows = None
        AU = self.xp.empty_like(U)
        for j in range(k):
            AU[j] = action(U[j])
        coarse = CoarseSpace(U, AU, self.ops)
        aa = self.ops.gram(AU, AU)
        defect = aa-coarse.H.T@coarse.Ginv@coarse.H
        relative_defect = np.sqrt(np.maximum(np.diag(defect), 0.)/np.diag(aa))
        self.info = dict(rank=k, captured=m, requested_window=self.window,
            rolling=self.rolling, restarts=self.restarts,
            ritz_values=eigenvalues.tolist(), gram_condition=float(np.linalg.cond(coarse.G)),
            relative_invariant_subspace_defect=relative_defect.tolist(),
            H_min_eigenvalue=float(np.linalg.eigvalsh((coarse.H+coarse.H.T)/2).min()),
            gram_symmetry_defect=float(np.linalg.norm(coarse.H-coarse.H.T)),
            captured_basis_bytes=captured_bytes, retained_basis_bytes=U.nbytes+AU.nbytes,
            setup_operator_calls=k, capture_s=self.capture_s, setup_s=time.perf_counter()-started)
        return coarse


@contextmanager
def candidate(mode='spectral', *, window=12, rank=4, which='high', rolling=False):
    """Exactly one six-load material solve per context; patches always restored."""
    if mode not in ('baseline', 'spectral', 'balanced', 'seed'):
        raise ValueError(mode)
    records, state = [], dict(loads=[], mode=mode)
    builder = Builder(window, rank, which, cp, rolling)
    coarse = None
    load_count = 0
    operator_id = None
    original_factory = linear.make_linear_solver

    def inner(action, rhs, dot, tolerance, maxiter, xp, **kwargs):
        nonlocal coarse, load_count
        if load_count >= 6 or xp is not cp or kwargs.get('precondition') is not None:
            raise ValueError('Experiment requires one symmetric six-load GPU solve')
        calls, pcalls = 0, 0
        def counted(v):
            nonlocal calls
            calls += 1
            return action(v)
        started = time.perf_counter()
        if load_count == 0 and mode != 'baseline':
            def capture(v):
                av = counted(v)
                builder.capture(v, av, dot)
                return av
            z, iterations = fused_cg(capture, rhs, dot, tolerance, maxiter, xp, **kwargs)
            coarse = builder.finish(counted, iterations)
            state['basis'] = builder.info
        elif coarse is None or mode == 'baseline':
            z, iterations = fused_cg(counted, rhs, dot, tolerance, maxiter, xp, **kwargs)
        elif mode == 'seed':
            guess = coarse.seed(rhs)
            r = rhs-counted(guess)
            bn, rn = np.sqrt(dot(rhs, rhs)), np.sqrt(dot(r, r))
            if rn > bn:
                z, iterations = fused_cg(counted, rhs, dot, tolerance, maxiter, xp, **kwargs)
            elif rn <= tolerance*bn:
                z, iterations = guess, 0
            else:
                delta, iterations = fused_cg(counted, r, dot, tolerance*bn/rn, maxiter, xp, **kwargs)
                z = guess+delta
                r = rhs-counted(z)
                rn = np.sqrt(dot(r, r))
                if rn > tolerance*bn:
                    remaining = maxiter-iterations
                    if remaining <= 0:
                        raise RuntimeError('Seed correction failed true residual')
                    delta, extra = fused_cg(counted, r, dot, tolerance*bn/rn, remaining, xp, **kwargs)
                    z += delta
                    iterations += extra
        else:
            def precondition(v):
                nonlocal pcalls
                pcalls += 1
                return getattr(coarse, mode)(v)
            z, iterations = fused_cg(counted, rhs, dot, tolerance, maxiter, xp,
                                    **dict(kwargs, precondition=precondition))
        state['loads'].append(dict(load=load_count, operator_calls=calls,
            precondition_calls=pcalls, coordinate_iterations=iterations,
            inner_including_setup_s=time.perf_counter()-started))
        load_count += 1
        return z, iterations

    def factory(**kwargs):
        nonlocal operator_id
        if operator_id is None:
            operator_id = id(kwargs['A'])
        elif operator_id != id(kwargs['A']):
            raise ValueError('Cannot reuse a basis across different operators')
        solve = original_factory(**kwargs)
        def run(**parameters):
            result, info = solve(**parameters)
            records.append({k: info[k] for k in ('coordinate_iterations', 'correction_iterations',
                'true_norm_res_rel', 'true_residual_converged')})
            return result, info
        return run

    with ExitStack() as stack:
        stack.enter_context(patch.object(linear, 'make_linear_solver', factory))
        stack.enter_context(patch.object(linear, 'fused_cg', inner))
        yield records, state


def energy_error_certificate(residuals, alpha, volume):
    """Discrete iterative-error bound in exact arithmetic, not a mesh certificate.

    residuals contains six REAL-coordinate residual columns (or a complex
    representation with its real inner product). alpha must be a proved lower
    spectral bound on the admissible subspace, never just a Ritz estimate.
    """
    if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(volume) or volume <= 0:
        raise ValueError('Positive proven alpha and volume required')
    if residuals.ndim != 2 or not np.isfinite(residuals).all():
        raise ValueError('Finite residual columns required')
    gram = (residuals.conj().T@residuals).real/volume
    upper = gram/alpha
    return upper, float(np.trace(upper))
