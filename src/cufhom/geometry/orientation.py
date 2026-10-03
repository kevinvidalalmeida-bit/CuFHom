"""Explicit orientation targets for SAM: A2 -> angular central Gaussian A4.

A2 does not uniquely determine A4. This is a *chosen distributional closure*,
not recovery of a measured A4 and not the time-evolution FEC algorithm.
Moments of u=sqrt(C)z/|sqrt(C)z|, z~N(0,I), are evaluated by deterministic
one-dimensional quadrature. Singular planar/aligned limits are analytic.
"""
from functools import lru_cache
from itertools import permutations
import numpy as np
from numba import njit
from scipy.integrate import quad_vec
from scipy.optimize import least_squares


def validate_a2(value):
    a = np.array(value, dtype=float, copy=True)
    if (a.shape != (3, 3) or not np.isfinite(a).all()
            or not np.allclose(a, a.T, rtol=0, atol=1e-10)
            or abs(np.trace(a)-1) > 1e-10
            or np.linalg.eigvalsh(a).min() < -1e-12):
        raise ValueError('A2 must be finite, symmetric PSD and trace one')
    return a


def validate_a4(value, a2=None):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (3, 3, 3, 3) or not np.isfinite(a).all():
        raise ValueError('A4 must be a finite full 3x3x3x3 tensor (not Voigt)')
    for axes in permutations(range(4)):
        if not np.allclose(a, a.transpose(axes), rtol=0, atol=1e-10):
            raise ValueError('A4 must be fully symmetric')
    contracted = validate_a2(np.einsum('ijkk->ij', a))
    if a2 is not None and not np.allclose(contracted, validate_a2(a2), rtol=0, atol=1e-9):
        raise ValueError('A4 contraction is inconsistent with the requested A2')
    if np.linalg.eigvalsh(a.reshape(9, 9)).min() < -1e-10:
        raise ValueError('A4 must be positive semidefinite on second-order tensors')
    # E[(u*u - A2) tensor (u*u - A2)] must also be PSD.
    covariance = a.reshape(9, 9)-np.outer(contracted.ravel(), contracted.ravel())
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('A4 has an invalid second-moment covariance')
    return a


def _principal_moments(log_c):
    """Laplace integral for inverse squared Gaussian norm; log variable avoids
    near-alignment scale separation. Tail bounds decay at least exp(-T/2).
    """
    log_c = np.asarray(log_c)
    def integrand(t):
        denominator = np.logaddexp(0., t+log_c)
        determinant = -.5*denominator.sum()
        second = .5*np.exp(log_c+t-denominator+determinant)
        fourth = .25*np.exp(log_c[:, None]+log_c[None, :]+2*t
                            -denominator[:, None]-denominator[None, :]+determinant)
        fourth[np.diag_indices(3)] *= 3
        return np.r_[second, fourth.ravel()]
    result, error = quad_vec(integrand, -float(max(log_c))-60,
                             -float(min(log_c))+60, epsabs=2e-12, epsrel=2e-11)
    if not np.isfinite(result).all() or error > 1e-8:
        raise ValueError('ACG moment quadrature did not converge')
    return result[:3], result[3:].reshape(3, 3)


@lru_cache(maxsize=256)
def _acg_cached(key):
    a2 = validate_a2(np.asarray(key).reshape(3, 3))
    values, q = np.linalg.eigh(a2)
    values = np.maximum(values, 0.)
    rank = np.count_nonzero(values > 1e-14)
    products = np.zeros((3, 3))
    if rank == 1:
        products[2, 2] = 1.
    elif rank == 2:
        # Exact angular Gaussian on the supporting plane. c_i proportional a_i².
        cross = values[1]*values[2]/2
        products[1, 2] = products[2, 1] = cross
        products[1, 1], products[2, 2] = values[1]-cross, values[2]-cross
    elif np.max(np.abs(values-1/3)) < 1e-14:
        products[:] = 1/15
        np.fill_diagonal(products, 1/5)
    else:
        # Fix c_3=1 to remove the distribution's arbitrary scalar scale.
        def residual(x):
            moment, _ = _principal_moments(np.r_[x, 0.])
            return np.log(moment[:2]/values[:2])
        fit = least_squares(residual, 2*np.log(values[:2]/values[2]),
                            xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=100)
        moment, products = _principal_moments(np.r_[fit.x, 0.])
        if not fit.success or np.max(np.abs(moment-values)) > 2e-9:
            raise ValueError('ACG closure failed to match A2; no silent fallback')
    principal = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for indices in set(permutations((i, i, j, j))):
                principal[indices] = products[i, j]
    a4 = np.einsum('ia,jb,kc,ld,abcd->ijkl', q, q, q, q, principal, optimize=True)
    return validate_a4(a4, a2)


def acg_closure(a2):
    """Return an independent array; cached targets cannot be mutated by callers."""
    a2 = validate_a2(a2)
    return _acg_cached(tuple(a2.ravel())).copy()


def resolve_orientation(a2=None, a4=None):
    if a4 is not None:
        fourth = validate_a4(a4, a2)
        second = np.einsum('ijkk->ij', fourth)
        source = 'explicit_A4'
    else:
        second = validate_a2(np.eye(3)/3 if a2 is None else a2)
        fourth = acg_closure(second)
        source = 'A2_to_A4_angular_central_Gaussian'
    return second.copy(), fourth.copy(), dict(
        source=source, A2=second.tolist(), A4=fourth.tolist(),
        contraction_error=float(np.linalg.norm(np.einsum('ijkk->ij', fourth)-second)),
        norm='full_tensor_frobenius',
        warning='A2 does not uniquely determine A4; closure is a distributional assumption')


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
def angular_transform(a2_key):
    a2=np.asarray(a2_key,dtype=float).reshape(3,3)
    vals,vecs=np.linalg.eigh(.5*(a2+a2.T))
    vals=np.clip(vals,1e-8,None);vals/=np.sum(vals)
    scale=_calibrate_principal_scale(vals,_probes())
    return np.ascontiguousarray(vecs@np.diag(scale),dtype=float)
