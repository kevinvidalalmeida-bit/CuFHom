"""Conservative SAM kernels, activated by the production RVE entry point.

Module switching is serialized within a process; separate worker processes can
generate in parallel. Direct SAMLite instances outside this API must not run
concurrently with a switched context. No fastmath or tolerance relaxation.
"""
from contextlib import contextmanager
from functools import lru_cache
from threading import RLock
import numpy as np
from numba import njit, prange
from . import sam_generator as sam

_candidate_reference = sam._check_candidate_serial_numba
_force_reference = sam._apply_forces_per_fiber_numba
_pair_reference = sam._apply_pair_forces_numba
SAM_CONTEXT_LOCK = RLock()


@lru_cache(maxsize=1)
def _probes():
    rng = np.random.default_rng(104729)
    probes = rng.standard_normal((8192, 3))
    probes /= np.maximum(np.linalg.norm(probes, axis=1)[:, None], 1e-12)
    return probes


@njit(cache=True)
def _angular_moments(probes, shape):
    # Fused quadrature: same 8192 directions, no fastmath or weaker tolerance.
    scale = np.sqrt(shape)
    sums = np.zeros(3)
    for i in range(len(probes)):
        x, y, z = probes[i, 0]*scale[0], probes[i, 1]*scale[1], probes[i, 2]*scale[2]
        norm = max(np.sqrt(x*x+y*y+z*z), 1e-12)
        x /= norm; y /= norm; z /= norm
        sums[0] += x*x; sums[1] += y*y; sums[2] += z*z
    return sums/len(probes)


@lru_cache(maxsize=512)
def angular_transform(a2_key):
    a2 = np.asarray(a2_key, dtype=float).reshape(3, 3)
    vals, vecs = np.linalg.eigh(.5*(a2+a2.T))
    vals = np.clip(vals, 1e-8, None); vals /= np.sum(vals)
    log_shape = np.log(vals); log_shape -= np.mean(log_shape)
    for _ in range(40):
        moment = _angular_moments(_probes(), np.exp(np.clip(log_shape, -30., 30.)))
        if np.max(np.abs(moment-vals)) <= 2e-7:
            break
        log_shape += 1.5*(np.log(vals)-np.log(np.maximum(moment, 1e-12)))
        log_shape -= np.mean(log_shape)
    principal_scale = np.sqrt(np.exp(np.clip(log_shape, -30., 30.)))
    return np.ascontiguousarray(vecs @ np.diag(principal_scale), dtype=float)


@njit(cache=True)
def _calibrate_principal_scale(vals, probes):
    log_shape=np.log(vals); log_shape-=np.mean(log_shape)
    for _ in range(40):
        moment=_angular_moments(probes,np.exp(np.clip(log_shape,-30.,30.)))
        if np.max(np.abs(moment-vals))<=2e-7:
            break
        log_shape+=1.5*(np.log(vals)-np.log(np.maximum(moment,1e-12)))
        log_shape-=np.mean(log_shape)
    return np.sqrt(np.exp(np.clip(log_shape,-30.,30.)))


@lru_cache(maxsize=512)
def fused_angular_transform(a2_key):
    a2=np.asarray(a2_key,dtype=float).reshape(3,3)
    vals,vecs=np.linalg.eigh(.5*(a2+a2.T))
    vals=np.clip(vals,1e-8,None);vals/=np.sum(vals)
    scale=_calibrate_principal_scale(vals,_probes())
    return np.ascontiguousarray(vecs@np.diag(scale),dtype=float)


@njit(inline='always')
def separated(cx, cy, cz, ux, uy, uz, vx, vy, vz, length, diameter):
    # Any distance between the two axis-aligned segment boxes is a lower bound
    # on the segment distance. A roundoff margin makes rejection conservative.
    margin = 1e-12 * max(1., length, diameter, abs(cx), abs(cy), abs(cz))
    gx = max(0., abs(cx) - .5*length*(abs(ux)+abs(vx)))
    gy = max(0., abs(cy) - .5*length*(abs(uy)+abs(vy)))
    gz = max(0., abs(cz) - .5*length*(abs(uz)+abs(vz)))
    return gx*gx+gy*gy+gz*gz > (diameter+margin)**2


@contextmanager
def accelerated_sam(candidates=True, forces=True, angular=True, periodic=False, fused_angular=False):
    """Serialize activation and restore kernels even when generation fails."""
    with SAM_CONTEXT_LOCK:
        with _activated_sam(candidates, forces, angular, periodic, fused_angular):
            yield


@contextmanager
def _activated_sam(candidates, forces, angular, periodic, fused_angular):
    if periodic:
        from . import sam_periodic_kernels as kernels
    else:
        from . import sam_acceleration_kernels as kernels
    changes = {}
    if periodic:
        changes['_check_candidate_serial_numba'] = kernels._check_candidate_serial_numba
        candidates = forces = True
    if candidates:
        changes['_check_candidates_against_existing_numba'] = kernels._check_candidates_against_existing_numba
    if forces:
        changes['_apply_forces_per_fiber_numba'] = kernels._apply_forces_per_fiber_numba
        changes['_apply_pair_forces_numba'] = kernels._apply_pair_forces_numba
    if angular:
        changes['_angular_gaussian_transform'] = fused_angular_transform if fused_angular else angular_transform
    originals = {name: getattr(sam, name) for name in changes}
    try:
        for name, fn in changes.items():
            setattr(sam, name, fn)
        yield
    finally:
        for name, fn in originals.items():
            setattr(sam, name, fn)
