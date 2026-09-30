"""Scoped nonlinear experiments; no production imports or default changes."""
from contextlib import contextmanager
import inspect
import time
from unittest.mock import patch
import numpy as np
from profile_g09_fft import cp
from benchmark_fused_adjoint_dot import FusedAdjointDot
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from pipeline.nonlinear_reference import IsotropicReference
from ffthompy.tensors import fft


def action_context(action):
    fields = inspect.getclosurevars(action).nonlocals
    required = {'coordinates', 'tangent', 'response', 'shift'}
    if not required.issubset(fields):
        raise ValueError('Experiment needs the audited nonlinear reduced_action closure')
    gpu = inspect.getclosurevars(fields['tangent']).nonlocals.get('gpu')
    if gpu is None:
        raise ValueError('Experiment requires GPU material')
    return fields, gpu


def tangent_reference(Q, gpu, response, baseline, shift, mode, cache):
    """Positive isotropic projections; fixed throughout one complete CG solve.

    Matrix J2 deviatoric tangent has eigenvalues a (x4) and a+b (x1).
    Voigt uses their arithmetic mean; Reuss uses inverse mean compliance.
    Full Voigt also includes the invariant bulk/shear trace of elastic fibers.
    These approximate only the preconditioner, never the material operator.
    """
    if 'counts' not in cache:
        cache['counts'] = cp.asnumpy(cp.bincount(gpu.ids.ravel(), minlength=len(gpu.table)))
        cache['table'] = cp.asnumpy(gpu.table).astype(np.float64)
    counts, table = cache['counts'], cache['table']
    matrix_count = int(counts[0])
    if mode != 'full_voigt' and matrix_count == 0:
        return baseline, dict(mode=mode, fallback='no matrix voxels')
    if 'voigt' not in cache:
        cache['voigt'] = cp.ReductionKernel('float32 a, float32 b, int32 ids', 'float64 out',
            'ids==0 ? 5.0*(double)a+(double)b : 0.0', 'a+b', 'out=a', '0',
            'matrix_tangent_trace', reduce_type='double')
        cache['reuss'] = cp.ReductionKernel('float32 a, float32 b, int32 ids', 'float64 out',
            'ids==0 ? 4.0/(double)a+1.0/((double)a+(double)b) : 0.0', 'a+b',
            'out=a', '0', 'matrix_tangent_compliance', reduce_type='double')
    if mode == 'matrix_reuss':
        inverse_trace = float(cache['reuss'](response.a, response.b, gpu.ids))
        K, mu = baseline.K, 5*matrix_count/(2*inverse_trace)
    else:
        trace = float(cache['voigt'](response.a, response.b, gpu.ids))
        if mode == 'matrix_voigt':
            K, mu = baseline.K, trace/(10*matrix_count)
        elif mode == 'full_voigt':
            bulk = np.sum(table[:, :3, :3], axis=(1,2))/9
            shear = (np.trace(table, axis1=1, axis2=2)-3*bulk)/10
            K = (matrix_count*baseline.K+counts[1:]@bulk[1:])/gpu.n
            mu = (trace/10+counts[1:]@shear[1:])/gpu.n
        else:
            raise ValueError('Unknown tangent reference')
    selected = IsotropicReference(Q.P, K, mu)  # validates SPD/finite
    selected.set_shift(shift, np.dtype('float32'))
    return selected, dict(mode=mode, K=float(K), mu=float(mu), shift=float(shift))


@contextmanager
def installed(*, fused_dot=False, reference='matrix'):
    """Single-threaded benchmark only; restores the real solver on every exit."""
    if reference not in ('matrix', 'matrix_voigt', 'matrix_reuss', 'full_voigt'):
        raise ValueError('Unknown reference')
    original = FusedCoordinatePCG.solve
    stats = dict(solves=[], fused_products=0, shift_fallbacks=0)
    caches = {}
    def solve(engine, action, rhs, dot, tolerance, maxiter, baseline):
        fields, gpu = action_context(action)
        Q, response, shift = fields['coordinates'], fields['response'], fields['shift']
        if Q is not engine.Q or rhs.dtype != cp.complex64:
            raise ValueError('Unsupported coordinate engine/precision')
        cache = caches.setdefault(engine, {})
        selected = baseline
        row = dict(mode=reference)
        started = time.perf_counter()
        if reference != 'matrix':
            selected, row = tangent_reference(Q, gpu, response, baseline, shift, reference, cache)
        row['setup_wall_s'] = time.perf_counter()-started
        stats['solves'].append(row)
        # Exact fallback for PTC: no accidental omission of shift*p.p, no
        # hidden change of the shifted operator or its true residual audit.
        if not fused_dot or shift:
            stats['shift_fallbacks'] += int(fused_dot and bool(shift))
            return original(engine, action, rhs, dot, tolerance, maxiter, selected)
        if 'fused' not in cache:
            cache['fused'] = FusedAdjointDot(Q, rhs.dtype)
        fused = cache['fused']
        latest = [None, None, None]
        def fused_action(z):
            strain = Q.strain(z)
            stress = fields['tangent'](strain, response)
            h = fft.rfftn_cached(stress, Q.shape, (1,2,3))
            out, pap = fused(h, z)
            latest[:] = z, out, pap
            return out
        def fused_product(a, b):
            if a is latest[0] and b is latest[1]:
                stats['fused_products'] += 1
                return float(latest[2])
            return dot(a, b)
        # The retained PCG consumes Ap before the next action. It never keeps
        # two operator outputs. The independent rhs-action(x) audit is kept.
        return original(engine, fused_action, rhs, fused_product, tolerance, maxiter, selected)
    with patch.object(FusedCoordinatePCG, 'solve', solve):
        yield stats
