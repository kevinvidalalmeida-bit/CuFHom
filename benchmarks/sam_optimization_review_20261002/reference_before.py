"""Frozen functions before the additional optimization review, 2026-10-02."""
import numpy as np
from scipy.optimize import minimize
from cufhom.pipeline.sam_contact_energy import contact_energy, objective
from cufhom.pipeline.sam_periodic_audit import audit_periodic_fibers
from cufhom.pipeline.sam_staged_insertion import draw_directions

def propose(accepted, count, target, settings, rng, target_a4=None):
    """Draw random soft insertions or rank them by contacts and orientation.

    Candidate ranking is an initialization heuristic, not an acceptance test.
    With A4 enabled, rank by its residual too. This is still only a proposal;
    independent certificates, not this heuristic, decide acceptance.
    """
    s = settings
    if s.insertion_policy == 'random':
        # The 2017 control draws from the prescribed distribution, independent
        # of the current residual and contacts. Isotropy is sampled exactly,
        # without calibrating an ACG transform on a finite quadrature.
        if np.allclose(target, np.eye(3)/3, rtol=0, atol=1e-12):
            raw = rng.normal(size=(count, 3))
            directions = raw/np.linalg.norm(raw, axis=1)[:, None]
        else:
            directions = draw_directions(target, rng, count)
        centers = rng.uniform(0., s.box, (count, 3))
        additions = np.column_stack((centers, directions, np.full(count, s.length),
                                     np.full(count, s.diameter)))
        return np.vstack((accepted, additions))
    result = accepted.copy()
    box = np.full(3, s.box)
    for _ in range(count):
        n = len(result)
        moment = result[:, 3:6].T@result[:, 3:6] if n else np.zeros((3, 3))
        moment4 = None
        if target_a4 is not None:
            from .sam_paper_reference import fourth_moment
            moment4 = n*fourth_moment(result[:, 3:6]) if n else np.zeros((3, 3, 3, 3))
        residual = (n+1)*target-moment
        directions = draw_directions(residual, rng, s.candidates)
        centers = rng.uniform(0., s.box, (s.candidates, 3))
        best_score, best = np.inf, None
        for center, direction in zip(centers, directions):
            candidate = np.r_[center, direction, s.length, s.diameter]
            trial = np.vstack((result, candidate))
            energy, _, _, _ = contact_energy(
                np.ascontiguousarray(trial[:, :3]), np.ascontiguousarray(trial[:, 3:6]),
                s.length, s.diameter+s.min_gap, box, first_new=n)
            if target_a4 is None:
                error = np.linalg.norm((moment+np.outer(direction, direction))/(n+1)-target)/np.linalg.norm(target)
            else:
                outer = np.outer(direction, direction)
                updated = (moment4+np.einsum('ij,kl->ijkl', outer, outer))/(n+1)
                error = np.linalg.norm(updated-target_a4)/np.linalg.norm(target_a4)
            score = energy/(n+1)+.5*s.orientation_weight*error**2
            if score < best_score:
                best_score, best = score, candidate
        result = np.vstack((result, best))
    return result


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
