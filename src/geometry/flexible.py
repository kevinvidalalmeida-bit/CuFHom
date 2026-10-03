"""Inextensible polygonal fibers represented by connected capsules.

Connectivity and contour length hold by construction. Bending is a geometric
regularizer; this model does not supply a mechanical bending constitutive law.
"""
import itertools
from dataclasses import dataclass
import numpy as np
from numba import njit
from scipy.optimize import minimize
from cufhom.geometry.contacts import contact_energy
from cufhom.geometry.audit import segment_distance
from cufhom.geometry.straight import Settings, draw_directions


@dataclass(frozen=True)
class FlexibleSettings(Settings):
    segments: int = 4
    bending_weight: float = .01
    max_turn_degrees: float = 30.
    conservative_cull: bool = True


def decode(q, length):
    """q[n,1+segments,3] = anchor and normalized tangent coordinates."""
    q = np.asarray(q, dtype=float)
    raw = q[:, 1:]
    norm = np.linalg.norm(raw, axis=2)
    if np.any(norm < 1e-12):
        raise ValueError('Degenerate segment tangent')
    u = raw/norm[:, :, None]
    ell = length/u.shape[1]
    offsets = np.concatenate((np.zeros((len(q), 1, 3)), np.cumsum(ell*u, axis=1)), axis=1)
    vertices = q[:, :1]+offsets
    centers = .5*(vertices[:, :-1]+vertices[:, 1:])
    return vertices, centers, u, norm


@njit(inline='always')
def _audit_pair_possible(c, u, i, j, ell, diameter, box):
    for axis in range(3):
        delta = c[j, axis]-c[i, axis]
        delta -= np.rint(delta/box)*box
        extent = .5*(abs(ell*u[i, axis])+abs(ell*u[j, axis]))+diameter
        if abs(delta) > extent+1e-12:
            return False
    return True


@njit(cache=True)
def _audit_candidate_indices(c, u, ell, diameter, box):
    """Exact-size broad-phase indices, no dense O(M²) vector temporaries.

    Retains the lexicographic reference pair ordering. The independent NumPy
    narrow phase below is untouched; this only filters impossible pairs.
    """
    count = 0
    for i in range(len(c)):
        for j in range(i, len(c)):
            if _audit_pair_possible(c, u, i, j, ell, diameter, box): count += 1
    ii = np.empty(count, dtype=np.int64); jj = np.empty(count, dtype=np.int64)
    at = 0
    for i in range(len(c)):
        for j in range(i, len(c)):
            if _audit_pair_possible(c, u, i, j, ell, diameter, box):
                ii[at], jj[at] = i, j
                at += 1
    return ii, jj


def chain_audit(q, length, diameter, box, target, *, overlap_tolerance, a2_tolerance,
                max_turn_degrees, conservative_cull=True):
    """Independent NumPy contact audit; no optimization kernel reused."""
    if (np.ndim(q) != 3 or q.shape[0] < 1 or q.shape[1] < 2 or q.shape[2] != 3
            or not np.isfinite(q).all() or min(length, diameter, box) <= 0):
        raise ValueError('Invalid chain table or physical dimensions')
    vertices, centers, u, _ = decode(q, length)
    ns = u.shape[1]; ell = length/ns
    c = centers.reshape(-1, 3); tangents = u.reshape(-1, 3)
    if conservative_cull:
        i, j = _audit_candidate_indices(c, tangents, ell, diameter, box)
    else:
        i, j = np.triu_indices(len(c), 0)
    original = c[j]-c[i]
    delta = original-np.rint(original/box)*box
    axes_i, axes_j = ell*tangents[i], ell*tangents[j]
    extent = .5*(np.abs(axes_i)+np.abs(axes_j))+diameter
    limit = int(np.ceil((ell+diameter)/box))
    shifts = [range(-limit, limit+1)]*3
    if conservative_cull:
        # Bound image ranges over surviving pairs. For short segments this
        # often reduces 27 lattice shifts to one without a nearest-image assumption.
        if len(delta):
            lower = np.ceil((-extent-delta-1e-12)/box).min(axis=0).astype(int)
            upper = np.floor((extent-delta+1e-12)/box).max(axis=0).astype(int)
            shifts = [range(max(-limit, a), min(limit, b)+1) for a, b in zip(lower, upper)]
        else:
            shifts = [range(0)]*3
    violations = 0; maximum = 0.; self_violations = 0
    for shift in itertools.product(*shifts):
        other = delta+box*np.array(shift)
        original_image = np.max(np.abs(other-original), axis=1) < 1e-9
        adjacent = (i//ns == j//ns) & (j-i <= 1) & original_image
        keep = np.all(np.abs(other) <= extent+1e-12, axis=1) & ~adjacent
        if not keep.any():
            continue
        distance = segment_distance(other[keep], axes_i[keep], axes_j[keep])
        penetration = (diameter-distance)/diameter
        bad = penetration > overlap_tolerance+1e-10
        violations += int(bad.sum())
        self_violations += int(np.sum(bad & (i[keep]//ns == j[keep]//ns)))
        maximum = max(maximum, float(penetration.max()))
    actual_lengths = np.linalg.norm(np.diff(vertices, axis=1), axis=2)
    length_error = float(np.max(np.abs(actual_lengths.sum(axis=1)-length))/length)
    dot = np.sum(u[:, 1:]*u[:, :-1], axis=2)
    maximum_turn = float(np.rad2deg(np.arccos(np.clip(dot, -1, 1))).max()) if ns > 1 else 0.
    a2 = tangents.T@tangents/len(tangents)
    a2_error = float(np.linalg.norm(a2-target)/np.linalg.norm(target))
    return dict(accepted=bool(violations == 0 and a2_error <= a2_tolerance
                             and maximum_turn <= max_turn_degrees+1e-8 and length_error < 1e-10),
                violating_image_contacts=violations, self_violating_image_contacts=self_violations,
                maximum_relative_overlap=maximum, A2_error=a2_error,
                maximum_turn_degrees=maximum_turn, contour_length_relative_error=length_error,
                local_junction_exemption='same fiber adjacent segments, original image only',
                model='union of connected capsules; nonadjacent self images audited')


def chain_objective(x, n, settings, target):
    s = settings
    q = x.reshape(n, s.segments+1, 3)
    _, centers, u, norms = decode(q, s.length/s.diameter)
    ell = s.length/s.diameter/s.segments
    energy, gc, gu, _ = contact_energy(
        np.ascontiguousarray(centers.reshape(-1, 3)), np.ascontiguousarray(u.reshape(-1, 3)),
        ell, 1., np.full(3, s.box/s.diameter), segments_per_fiber=s.segments,
        conservative_cull=s.conservative_cull)
    gc = gc.reshape(n, s.segments, 3)/n
    gu = gu.reshape(n, s.segments, 3)/n
    # Each tangent moves its own midpoint and every downstream midpoint.
    downstream = np.cumsum(gc[:, ::-1], axis=1)[:, ::-1]-gc
    gu += ell*(.5*gc+downstream)
    error = np.einsum('nsi,nsj->ij', u, u)/(n*s.segments)-target
    denom = np.sum(target*target)
    energy = energy/n+.5*s.orientation_weight*np.sum(error*error)/denom
    gu += 2*s.orientation_weight/(n*s.segments*denom)*(u@error)
    if s.segments > 1:
        differences = u[:, 1:]-u[:, :-1]
        weight = s.bending_weight/(n*(s.segments-1))
        energy += .5*weight*np.sum(differences**2)
        gu[:, 1:] += weight*differences; gu[:, :-1] -= weight*differences
        # Soft turn penalty aids optimization; hard audit still controls output.
        dot = np.sum(u[:, 1:]*u[:, :-1], axis=2)
        excess = np.maximum(0., np.cos(np.deg2rad(s.max_turn_degrees))-dot)
        energy += 5.*np.sum(excess**2)/n
        gu[:, 1:] -= 10./n*excess[:, :, None]*u[:, :-1]
        gu[:, :-1] -= 10./n*excess[:, :, None]*u[:, 1:]
    grad = np.empty_like(q)
    grad[:, 0] = gc.sum(axis=1)
    grad[:, 1:] = (gu-u*np.sum(gu*u, axis=2)[:, :, None])/norms[:, :, None]
    return float(energy), grad.ravel()


def relax_chains(q, s, target, progress=None):
    scaled = q.copy(); scaled[:, 0] /= s.diameter
    # Raw tangent norm preconditions angle vs positional displacements.
    scaled[:, 1:] *= .5*s.length/s.diameter
    best_score, best_table, best_audit = np.inf, q.copy(), None
    iteration = 0

    def record(x):
        nonlocal best_score, best_table, best_audit, iteration
        table = x.reshape(q.shape).copy()
        table[:, 0] = (table[:, 0]*s.diameter) % s.box
        table[:, 1:] /= np.linalg.norm(table[:, 1:], axis=2)[:, :, None]
        checked = chain_audit(table, s.length, s.diameter, s.box, target,
                              overlap_tolerance=s.overlap_tolerance, a2_tolerance=s.a2_tolerance,
                              max_turn_degrees=s.max_turn_degrees, conservative_cull=s.conservative_cull)
        score = max(checked['maximum_relative_overlap']/s.overlap_tolerance,
                    checked['A2_error']/s.a2_tolerance,
                    checked['maximum_turn_degrees']/s.max_turn_degrees)
        if score < best_score:
            best_score, best_table, best_audit = score, table, checked
        if progress and iteration % 25 == 0:
            progress(dict(event='relax_progress', iteration=iteration, **checked))
        iteration += 1

    record(scaled.ravel())
    result = minimize(lambda x: chain_objective(x, len(q), s, target), scaled.ravel(), jac=True,
                      method='L-BFGS-B', callback=record,
                      options=dict(maxiter=s.relax_iterations, maxls=30, ftol=1e-13, gtol=1e-9))
    record(result.x)
    return best_table, best_audit, dict(iterations=int(result.nit), evaluations=int(result.nfev))


def generate_flexible(s, target, *, progress=None, checkpoint=None):
    # Base validation excluding the straight rank-one precheck: a curved fiber
    # needs its own full audit. Exact rank-one A2 would also force straightness.
    from cufhom.geometry.straight import validate_settings
    target = validate_settings(s, target)
    if s.solver != 'a2' or s.min_gap:
        raise ValueError('Flexible generator currently supports a2 without explicit minimum gap only')
    if (s.segments < 2 or int(s.segments) != s.segments or not np.isfinite(s.bending_weight)
            or s.bending_weight < 0 or not 0 < s.max_turn_degrees < 90):
        raise ValueError('Flexible mode requires >=2 segments, nonnegative bending and turn in (0,90)')
    if s.length/s.segments < s.diameter:
        raise ValueError('Flexible chains require segment length >= diameter to audit nonadjacent segments')
    rng = np.random.default_rng(s.seed)
    accepted = np.empty((0, s.segments+1, 3)); events = []; batch = s.batch_size
    status = 'stage_budget_exhausted'
    def emit(event):
        events.append(event)
        if progress: progress(event.copy())
    for stage in range(s.max_stages):
        if len(accepted) >= s.target_count: break
        bootstrap = not len(accepted)
        addition = min(s.initial_count if bootstrap else batch, s.target_count-len(accepted))
        committed = False
        for attempt in range(s.attempts):
            emit(dict(event='trial_start', stage=stage, attempt=attempt,
                      accepted_count=len(accepted), addition=addition))
            trial = accepted.copy()
            for _ in range(addition):
                n = len(trial)
                u = trial[:, 1:]
                moment = np.einsum('nsi,nsj->ij', u, u)/s.segments if n else np.zeros((3, 3))
                dirs = draw_directions((n+1)*target-moment, rng, s.candidates)
                anchors = rng.uniform(0, s.box, (s.candidates, 3))
                best = [np.inf, None]
                for anchor, direction in zip(anchors, dirs):
                    new = np.vstack((anchor, np.tile(direction, (s.segments, 1))))
                    candidate = np.concatenate((trial, new[None]), axis=0)
                    _, c, u, _ = decode(candidate, s.length)
                    energy, _, _, _ = contact_energy(
                        np.ascontiguousarray(c.reshape(-1, 3)), np.ascontiguousarray(u.reshape(-1, 3)),
                        s.length/s.segments, s.diameter, np.full(3, s.box),
                        first_new=n*s.segments, segments_per_fiber=s.segments,
                        conservative_cull=s.conservative_cull)
                    error = np.linalg.norm((moment+np.outer(direction, direction))/(n+1)-target)/np.linalg.norm(target)
                    score = energy/(n+1)+.5*s.orientation_weight*error**2
                    if score < best[0]: best[:] = [score, candidate]
                trial = best[1]
            candidate, checked, opt = relax_chains(trial, s, target, progress)
            if candidate.shape != trial.shape:
                raise RuntimeError('Optimizer changed fiber or segment count')
            # Transaction boundary: re-audit the returned state, never trust
            # a stale optimizer flag/certificate from a different iterate.
            checked = chain_audit(candidate, s.length, s.diameter, s.box, target,
                                   overlap_tolerance=s.overlap_tolerance, a2_tolerance=s.a2_tolerance,
                                   max_turn_degrees=s.max_turn_degrees, conservative_cull=s.conservative_cull)
            emit(dict(event='commit' if checked['accepted'] else 'rollback', stage=stage, attempt=attempt,
                      previous_count=len(accepted), proposed_count=len(candidate), certificate=checked, **opt))
            if checked['accepted']:
                accepted = candidate.copy(); committed = True
                if checkpoint: checkpoint(accepted.copy(), events[-1].copy())
                break
        if not committed:
            if not bootstrap and addition > 1:
                batch = max(1, addition//2)
                emit(dict(event='reduce_batch', stage=stage, batch=batch, accepted_count=len(accepted)))
                continue
            status = 'bootstrap_failed' if bootstrap else 'stalled'
            break
    if len(accepted) >= s.target_count: status = 'target_count_reached'
    final = (chain_audit(accepted, s.length, s.diameter, s.box, target,
                        overlap_tolerance=s.overlap_tolerance, a2_tolerance=s.a2_tolerance,
                        max_turn_degrees=s.max_turn_degrees, conservative_cull=s.conservative_cull) if len(accepted) else None)
    return accepted, dict(status=status, accepted_count=len(accepted), target_count=s.target_count,
                          geometric_certificate=final, vf_not_certified=True, events=events)
