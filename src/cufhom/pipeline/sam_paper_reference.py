"""Opt-in straight SAM experiment, NOT a verified reproduction of AR33/Vf60.

References: Schneider (2017), doi:10.1007/s00466-016-1350-7, eq. 9;
Mehta & Schneider (2022), doi:10.1007/s00466-022-02201-x, eqs. 3.12--3.15.
Equal-length capsules, general A4, inertia metric, normalized tangent step
(2017) or optional sphere exponential map (2022). Armijo is our safeguard.
The physical volume/raster convention remains capsules, not a 2017 replica.
"""
import numpy as np
from numba import njit
from .sam_contact_energy import contact_energy, _closest
from .sam_neighbor_cache import build_aabb_pairs, endpoint_motion_bound
from .sam_periodic_audit import audit_periodic_fibers
from .sam_orientation import resolve_orientation


def rotational_inertia(length, diameter):
    """Length-squared scale, eq. 3.14; denominator must NOT be multiplied."""
    if not np.isfinite([length, diameter]).all() or min(length, diameter) <= 0:
        raise ValueError('Positive finite length and diameter required')
    a = length / diameter
    return diameter**2 * (a**3/12 + a*a/6 + 3*a/16 + 1/15) / (a + 2/3)


def isotropic_a4():
    eye = np.eye(3)
    return (np.einsum('ij,kl->ijkl', eye, eye)
            + np.einsum('ik,jl->ijkl', eye, eye)
            + np.einsum('il,jk->ijkl', eye, eye))/15


def fourth_moment(u):
    outer = np.einsum('ni,nj->nij', u, u)
    return np.einsum('nij,nkl->ijkl', outer, outer)/len(u)


def a4_error(u, target=None):
    target = isotropic_a4() if target is None else target
    return float(np.linalg.norm(fourth_moment(u)-target)/np.linalg.norm(target))


def sphere_exp(u, tangent):
    """Great-circle update; normalization only removes round-off."""
    tangent = tangent-u*np.sum(u*tangent, axis=1)[:, None]
    theta = np.linalg.norm(tangent, axis=1)
    out = np.cos(theta)[:, None]*u + np.sinc(theta/np.pi)[:, None]*tangent
    return out/np.linalg.norm(out, axis=1)[:, None]


def sphere_retract(u, tangent):
    """Normalized tangent Euler update used in the 2017 paper."""
    tangent = tangent-u*np.sum(u*tangent, axis=1)[:, None]
    out = u+tangent
    return out/np.linalg.norm(out, axis=1)[:, None]


@njit(cache=True)
def _near_segment_pairs(pairs, centers, u, length, box, reach):
    keep = np.zeros(len(pairs), dtype=np.bool_)
    for k in range(len(pairs)):
        i, j = pairs[k]
        delta = centers[j]-centers[i]
        delta -= np.rint(delta/box)*box
        a, b = length*u[i], length*u[j]
        ext = .5*(np.abs(a)+np.abs(b))+reach+1e-12
        lo = np.ceil((-ext-delta)/box).astype(np.int64)
        hi = np.floor((ext-delta)/box).astype(np.int64)
        for sx in range(lo[0], hi[0]+1):
            for sy in range(lo[1], hi[1]+1):
                for sz in range(lo[2], hi[2]+1):
                    shift = box*np.array([sx, sy, sz])
                    if _closest(delta+shift, a, b)[0] <= reach+1e-12:
                        keep[k] = True
    return pairs[keep]


class ContactNeighbors:
    """Single-level AABB/segment Verlet list, checked on EVERY objective evaluation.

    Hausdorff endpoint-motion bound includes rotations and rejected line-search
    trials. No assumption about a fixed rebuild interval. Self images are always
    kept. Reference positions are copied, never views into mutable optimizer data.
    """
    def __init__(self, length, domain, skin=.5, contact_diameter=1.):
        if not np.isfinite(skin) or skin <= 0:
            raise ValueError('Positive finite skin required')
        self.length, self.domain, self.skin = length, domain.copy(), skin
        self.contact_diameter = contact_diameter
        self.centers = None
        self.rebuilds = 0

    def get(self, centers, u):
        rebuild = self.centers is None or len(centers) != len(self.centers)
        if not rebuild:
            movement = endpoint_motion_bound(centers, u, self.centers, self.u,
                                              self.domain, self.length)
            rebuild = 2*movement >= self.skin*(1-1e-12)
        if rebuild:
            pairs = build_aabb_pairs(centers, u, self.domain, self.length, self.contact_diameter, self.skin)
            pairs = _near_segment_pairs(pairs, centers, u, self.length, self.domain,
                                        self.contact_diameter+self.skin)
            n = len(centers)
            counts = np.bincount(pairs[:, 0], minlength=n)+1
            offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
            indices = np.empty(offsets[-1], dtype=np.int64)
            indices[offsets[:-1]] = np.arange(n)  # retain periodic self contacts
            if len(pairs):
                slots = np.arange(len(pairs))+pairs[:, 0]+1
                indices[slots] = pairs[:, 1]
            self.offsets, self.indices = offsets, indices
            self.centers, self.u = centers.copy(), u.copy()
            self.rebuilds += 1
        return self.offsets, self.indices


class MultilevelContactNeighbors:
    """Nested Verlet lists with separate endpoint-motion bounds per level.

    Skins halve towards the contact radius. A parent is rebuilt before its
    motion can consume the *difference* of parent/child cutoffs, ensuring it
    contains every pair needed to rebuild that child. A simple contact-only
    validity bound on the parent would be incorrect. Self images remain in CSR.
    """
    def __init__(self, length, domain, skin=.5, contact_diameter=1.):
        if not np.isfinite(skin) or skin <= 0:
            raise ValueError('Positive finite skin required')
        self.length, self.domain = length, domain.copy()
        self.contact_diameter = contact_diameter
        skins = [skin]
        while skins[-1] < max(skin, length/2):
            skins.append(2*skins[-1])
        self.skins = skins[::-1]
        self.refs = [None]*len(skins)
        self.pairs = [None]*len(skins)
        self.level_rebuilds = [0]*len(skins)
        self.rebuilds = 0

    def get(self, centers, u):
        parent_rebuilt = False
        for level, skin in enumerate(self.skins):
            reference = self.refs[level]
            rebuild = parent_rebuilt or reference is None or len(reference[0]) != len(centers)
            if not rebuild:
                allowance = skin-(self.skins[level+1] if level+1 < len(self.skins) else 0.)
                movement = endpoint_motion_bound(centers, u, *reference, self.domain, self.length)
                rebuild = 2*movement >= allowance*(1-1e-12)
            if rebuild:
                candidates = (build_aabb_pairs(centers, u, self.domain, self.length,
                                               self.contact_diameter, skin)
                              if level == 0 else self.pairs[level-1])
                self.pairs[level] = _near_segment_pairs(candidates, centers, u, self.length,
                                                        self.domain, self.contact_diameter+skin)
                self.refs[level] = (centers.copy(), u.copy())
                self.level_rebuilds[level] += 1
            parent_rebuilt = rebuild
        if parent_rebuilt:
            pairs = self.pairs[-1]
            n = len(centers)
            offsets = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=n)+1)].astype(np.int64)
            indices = np.empty(offsets[-1], dtype=np.int64)
            indices[offsets[:-1]] = np.arange(n)
            if len(pairs):
                indices[np.arange(len(pairs))+pairs[:, 0]+1] = pairs[:, 1]
            self.offsets, self.indices = offsets, indices
            self.rebuilds += 1
        return self.offsets, self.indices


def objective(centers, u, length, domain, weight=1., neighbors=None, target_a4=None,
              contact_diameter=1.):
    """Diameter=1 coordinates. Returns energy and Euclidean derivatives.

    E = E_contact + N*epsilon*weight/8 * ||A4-A4_target||_F**2.
    Dividing the orientation gradient by epsilon recovers the torque/A4
    balance in the published equation for equal lengths.
    """
    kwargs = {}
    if neighbors is not None:
        kwargs['neighbor_offsets'], kwargs['neighbor_indices'] = neighbors.get(centers, u)
    energy, gc, gu, maximum = contact_energy(centers, u, length, contact_diameter, domain, **kwargs)
    # contact_energy normalizes by collision diameter; restore physical-diameter
    # units so adding a gap does not also weaken the contact penalty.
    energy *= contact_diameter**2
    gc *= contact_diameter**2
    gu *= contact_diameter**2
    maximum *= contact_diameter  # violation relative to physical diameter=1
    eps = rotational_inertia(length, 1.)
    target_a4 = isotropic_a4() if target_a4 is None else target_a4
    residual = fourth_moment(u)-target_a4
    energy += len(u)*eps*weight/8*np.sum(residual*residual)
    gu += eps*weight*np.einsum('ijkl,nj,nk,nl->ni', residual, u, u, u)
    return float(energy), gc, gu, float(maximum), float(np.linalg.norm(residual)/np.linalg.norm(target_a4))


def relax_paper_reference(fibers, box, target, *, maxiter=300,
                          orientation_weight=1., overlap_tolerance=1/120,
                          a2_tolerance=.035, a4_tolerance=1e-4,
                          progress=None, use_neighbors=True, target_a4=None,
                          rotation_update='retraction', min_gap=0., neighbor_mode='single'):
    """Keep the best feasibility score, audit independently, never mutate input.

    A2 input chooses the explicit ACG closure unless A4 is supplied. Full tensor
    Frobenius norm is used, not unweighted engineering-Voigt norm. Both A2 and
    A4 are checked independently after optimization, including on rollback.
    """
    f = np.array(fibers, dtype=float, copy=True)
    before = audit_periodic_fibers(f, box, overlap_tolerance)
    target, target_a4, orientation = resolve_orientation(target, target_a4)
    if rotation_update not in ('retraction', 'geodesic'):
        raise ValueError('Unknown rotation update')
    update = sphere_retract if rotation_update == 'retraction' else sphere_exp
    if neighbor_mode not in ('single', 'multilevel'):
        raise ValueError('Unknown neighbor mode')
    if not (np.allclose(f[:, 6:], f[0, 6:])):
        raise ValueError('Equal lengths and diameters required')
    if (not np.isfinite([orientation_weight, overlap_tolerance, a2_tolerance, a4_tolerance]).all()
            or min(orientation_weight, overlap_tolerance, a2_tolerance, a4_tolerance) <= 0
            or maxiter < 1):
        raise ValueError('Positive finite solver settings required')
    diameter = f[0, 7]
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError('Minimum gap must be finite and nonnegative')
    contact_diameter = 1+min_gap/diameter
    length = f[0, 6]/diameter
    domain = np.broadcast_to(np.asarray(box, dtype=float)/diameter, (3,)).copy()
    c = np.ascontiguousarray(f[:, :3]/diameter)
    u = np.ascontiguousarray(f[:, 3:6])
    eps = rotational_inertia(length, 1.)
    neighbor_type = ContactNeighbors if neighbor_mode == 'single' else MultilevelContactNeighbors
    neighbors = neighbor_type(length, domain, contact_diameter=contact_diameter) if use_neighbors else None
    evaluations = 0

    def evaluate(c, u):
        nonlocal evaluations
        evaluations += 1
        return objective(c, u, length, domain, orientation_weight, neighbors,
                         target_a4, contact_diameter)

    current = evaluate(c, u)
    history = []
    best_score, best_c, best_u = np.inf, c.copy(), u.copy()
    step, accepted_steps = .3, 0
    status = 'iteration_budget_exhausted'
    for iteration in range(maxiter+1):
        energy, gc, gu, overlap, a4 = current
        a2 = float(np.linalg.norm(u.T@u/len(u)-target)/np.linalg.norm(target))
        score = max(overlap/overlap_tolerance, a2/a2_tolerance, a4/a4_tolerance)
        if score < best_score:
            best_score, best_c, best_u = score, c.copy(), u.copy()
        row = dict(iteration=iteration, energy=energy, overlap_relative=overlap,
                   A2_error=a2, A4_error=a4, step=step, evaluations=evaluations)
        # Bounded history: checkpoints/progress contain every 25th iteration.
        if iteration % 25 == 0 or score <= 1 or iteration == maxiter:
            history.append(row)
            if progress:
                progress(row.copy())
        if score <= 1:
            status = 'constraints_satisfied'
            break
        if iteration == maxiter:
            break
        dc = -gc
        du = -(gu-u*np.sum(gu*u, axis=1)[:, None])/eps
        slope = float(np.sum(gc*dc)+np.sum(gu*du))
        if not np.isfinite(slope) or slope >= -1e-30:
            status = 'stationary_infeasible'
            break
        trial_step = min(.3, step*1.5)
        for _ in range(24):
            trial_c = np.ascontiguousarray(np.mod(c+trial_step*dc, domain))
            trial_u = np.ascontiguousarray(update(u, trial_step*du))
            trial = evaluate(trial_c, trial_u)
            if trial[0] <= energy+1e-4*trial_step*slope:
                c, u, current = trial_c, trial_u, trial
                step = trial_step
                accepted_steps += 1
                break
            trial_step *= .5
        else:
            status = 'line_search_stalled'
            break
    f[:, :3] = np.mod(best_c*diameter, np.asarray(box))
    f[:, 3:6] = best_u
    after = audit_periodic_fibers(f, box, overlap_tolerance)
    inflated = f.copy()
    inflated[:, 7] += min_gap
    gap_audit = audit_periodic_fibers(inflated, box, overlap_tolerance/contact_diameter) if min_gap else after
    a2 = float(np.linalg.norm(best_u.T@best_u/len(f)-target)/np.linalg.norm(target))
    a4 = a4_error(best_u, target_a4)
    accepted = bool(after['accepted'] and gap_audit['accepted'] and a2 <= a2_tolerance and a4 <= a4_tolerance)
    return f, dict(before=before, after=after, A2_error=a2, A4_error=a4,
                   contact_and_A2_accepted=bool(after['accepted'] and a2 <= a2_tolerance),
                   contact_A2_A4_accepted=accepted, vf_not_certified=True,
                   optimizer_success=accepted, optimizer_message=status,
                   iterations=accepted_steps, evaluations=evaluations, history=history,
                   neighbor_rebuilds=neighbors.rebuilds if neighbors is not None else 0,
                   neighbor_mode=neighbor_mode if use_neighbors else 'dense',
                   neighbor_level_rebuilds=getattr(neighbors, 'level_rebuilds', []),
                   orientation_norm='full_tensor_frobenius',
                   orientation_target=orientation, rotation_update=rotation_update,
                   gap_audit=gap_audit, min_gap=min_gap,
                   solver='inertia_A4_Armijo')
