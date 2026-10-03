"""Allocation-based contact reference for independent regression checks."""
import math
import numpy as np
from numba import njit
from cufhom.pipeline.sam_contact_energy import _contact_normal

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
