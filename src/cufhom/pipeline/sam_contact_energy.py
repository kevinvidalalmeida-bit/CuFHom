"""Experimental, continuous capsule relaxation; never enabled implicitly.

Exact periodic pair/self-image energy, analytic position/orientation gradients,
and L-BFGS line search. No voxels, fiber removal, diameter changes or FFT solves.
The independent geometric audit, not optimizer.success, decides feasibility.
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize
from .sam_periodic_audit import audit_periodic_fibers
from .sam_periodic_kernels import _contact_normal


@njit(cache=True)
def _closest(delta, u, v):
    w = -.5*u + .5*v - delta
    a, b, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d, e = np.dot(u, w), np.dot(v, w)
    det = a*c-b*b
    ss = np.empty(5); tt = np.empty(5)
    ss[0], tt[0] = 0., min(1., max(0., e/c))
    ss[1], tt[1] = 1., min(1., max(0., (e+b)/c))
    ss[2], tt[2] = min(1., max(0., -d/a)), 0.
    ss[3], tt[3] = min(1., max(0., (b-d)/a)), 1.
    ss[4], tt[4] = -1., -1.
    if det > 1e-12*a*c:
        ss[4], tt[4] = (b*e-c*d)/det, (a*e-b*d)/det
    best, s, t = np.inf, 0., 0.
    for k in range(5):
        if 0 <= ss[k] <= 1 and 0 <= tt[k] <= 1:
            r = w+ss[k]*u-tt[k]*v
            val = np.dot(r, r)
            if val < best:
                best, s, t = val, ss[k], tt[k]
    return math.sqrt(max(0., best)), s, t, w+s*u-t*v


@njit(cache=True)
def contact_energy(centers, directions, length, diameter, box, first_new=0, segments_per_fiber=1,
                   conservative_cull=True, neighbor_offsets=None, neighbor_indices=None):
    """Unnormalized energy and gradients wrt centers and unit directions.

Each unordered inter-fiber image contact occurs once. Self images use one
lexicographic half of the lattice, so +/- shifts are not counted twice.
    At intersecting axes the distance is nonsmooth; use a separating subgradient.
"""
    n = len(centers)
    gc = np.zeros_like(centers); gu = np.zeros_like(directions)
    energy, maximum = 0., 0.
    for i in range(n):
        start = max(i, first_new)
        entries = n-start if neighbor_offsets is None else neighbor_offsets[i+1]-neighbor_offsets[i]
        for entry in range(entries):
            j = start+entry if neighbor_offsets is None else neighbor_indices[neighbor_offsets[i]+entry]
            if j < start:
                continue
            if conservative_cull:
                # Reject impossible contacts BEFORE creating temporary arrays.
                # Nearest-center displacement is a componentwise lower bound
                # over all periodic images; narrow phase still enumerates them.
                distance_lower_sq = 0.
                for axis in range(3):
                    dx = centers[j, axis]-centers[i, axis]
                    dx -= np.rint(dx/box[axis])*box[axis]
                    gap = max(0., abs(dx)-.5*length*(abs(directions[i, axis])+abs(directions[j, axis])))
                    distance_lower_sq += gap*gap
                if distance_lower_sq > (diameter+1e-12*max(1., length, diameter))**2:
                    continue
            delta = centers[j]-centers[i]
            delta -= np.rint(delta/box)*box
            u, v = length*directions[i], length*directions[j]
            ext = .5*(np.abs(u)+np.abs(v))+diameter+1e-12
            lo = np.ceil((-ext-delta)/box).astype(np.int64)
            hi = np.floor((ext-delta)/box).astype(np.int64)
            for sx in range(lo[0], hi[0]+1):
                for sy in range(lo[1], hi[1]+1):
                    for sz in range(lo[2], hi[2]+1):
                        if i == j and not (sx > 0 or (sx == 0 and sy > 0)
                                           or (sx == 0 and sy == 0 and sz > 0)):
                            continue
                        other = delta + box*np.array([sx, sy, sz])
                        # Connected neighbors naturally share their end cap in
                        # a polygonal fiber. Exempt ONLY the original image;
                        # periodic copies and nonadjacent self contacts remain.
                        if (segments_per_fiber > 1 and i//segments_per_fiber == j//segments_per_fiber
                                and j-i <= 1 and np.max(np.abs(other-(centers[j]-centers[i]))) < 1e-9):
                            continue
                        gap = np.maximum(0., np.abs(other)-.5*(np.abs(u)+np.abs(v)))
                        if np.dot(gap, gap) > diameter**2:
                            continue
                        dist, s, t, diff = _closest(other, u, v)
                        penetration = diameter-dist
                        if penetration <= 0:
                            continue
                        maximum = max(maximum, penetration/diameter)
                        energy += .5*(penetration/diameter)**2
                        if dist > 1e-10:
                            g = -penetration/diameter**2 * diff/dist
                        else:
                            normal = _contact_normal(diff[0], diff[1], diff[2], dist, directions, i, j)
                            g = penetration/diameter**2 * np.array(normal)
                        gc[i] += g
                        gc[j] -= g
                        gu[i] += (s-.5)*length*g
                        gu[j] -= (t-.5)*length*g
    return energy, gc, gu, maximum


def objective(x, length, box, target, weight):
    """Diameter-scaled coordinates, orientation metric set by half-length."""
    values = np.asarray(x).reshape(-1, 6)
    raw = values[:, 3:]
    norms = np.maximum(np.linalg.norm(raw, axis=1), 1e-12)
    directions = np.ascontiguousarray(raw/norms[:, None])
    energy, gc, gu, maximum = contact_energy(
        np.ascontiguousarray(values[:, :3]), directions, length, 1., box)
    n = len(values)
    error = directions.T @ directions/n-target
    denom = float(np.sum(target*target))
    a2_error = float(np.linalg.norm(error)/np.sqrt(denom))
    energy = energy/n + .5*weight*a2_error**2
    gu = gu/n + (2*weight/(n*denom))*(directions @ error)
    gr = (gu-directions*np.sum(gu*directions, axis=1)[:, None])/norms[:, None]
    return energy, np.column_stack((gc/n, gr)).ravel(), maximum, a2_error


def relax_contact_energy(fibers, box, target, *, maxiter=300,
                         orientation_weight=1., overlap_tolerance=1/120,
                         a2_tolerance=.035, progress=None):
    """Return a NEW table and diagnostics; inputs and historical data untouched."""
    f = np.array(fibers, dtype=float, copy=True)
    # Validation, including unit orientations, also checks the initial state.
    before = audit_periodic_fibers(f, box, overlap_tolerance)
    if not (np.allclose(f[:, 6], f[0, 6]) and np.allclose(f[:, 7], f[0, 7])):
        raise ValueError('This prototype requires equal lengths and diameters')
    target = np.asarray(target, dtype=float)
    if (target.shape != (3, 3) or not np.isfinite(target).all()
            or not np.allclose(target, target.T) or not np.isclose(np.trace(target), 1.)
            or np.linalg.eigvalsh(target).min() < -1e-12):
        raise ValueError('Expected symmetric PSD, trace-one A2')
    if maxiter < 1 or orientation_weight < 0 or overlap_tolerance <= 0 or a2_tolerance <= 0:
        raise ValueError('Invalid optimization settings')
    diameter = f[0, 7]; length = f[0, 6]/diameter
    domain = np.broadcast_to(np.asarray(box, dtype=float)/diameter, (3,)).copy()
    x0 = np.column_stack((f[:, :3]/diameter, .5*length*f[:, 3:6])).ravel()
    history = []
    best = [np.inf, x0.copy()]

    def evaluate(x):
        return objective(x, length, domain, target, orientation_weight)

    def record(x):
        energy, _, overlap, a2 = evaluate(x)
        score = max(overlap/overlap_tolerance, a2/a2_tolerance)
        if score < best[0]:
            best[:] = [score, x.copy()]
        history.append(dict(iteration=len(history), energy=float(energy),
                            overlap_relative=float(overlap), A2_error=float(a2)))
        if progress and (len(history) == 1 or (len(history)-1) % 25 == 0):
            progress(history[-1])

    record(x0)
    result = minimize(lambda x: evaluate(x)[:2], x0, jac=True, method='L-BFGS-B',
                      callback=record, options=dict(maxiter=maxiter, maxls=30,
                                                   ftol=1e-14, gtol=1e-9))
    record(result.x)
    chosen = best[1].reshape(-1, 6)
    f[:, :3] = np.mod(chosen[:, :3]*diameter, np.asarray(box))
    f[:, 3:6] = chosen[:, 3:]/np.linalg.norm(chosen[:, 3:], axis=1)[:, None]
    after = audit_periodic_fibers(f, box, overlap_tolerance)
    a2 = float(np.linalg.norm(f[:, 3:6].T@f[:, 3:6]/len(f)-target)/np.linalg.norm(target))
    return f, dict(before=before, after=after, A2_error=a2,
                   contact_and_A2_accepted=bool(after['accepted'] and a2 <= a2_tolerance),
                   vf_not_certified=True, optimizer_success=bool(result.success),
                   optimizer_message=str(result.message), iterations=int(result.nit),
                   evaluations=int(result.nfev), history=history)
