"""Experimental rigid-fiber SAM: soft insertion, collective solve, atomic commit.

Only independently audited configurations become checkpoints. A failed trial
never changes the accepted geometry. No dimension changes or FFT corrections.
This is not the flexible-fiber fSAM algorithm and is not a production default.
"""
from dataclasses import dataclass
import copy
import time
import numpy as np
from .sam_acceleration import fused_angular_transform
from .sam_contact_energy import candidate_contact_energies, relax_contact_energy
from .sam_periodic_audit import audit_periodic_fibers
from .sam_orientation import validate_a2, resolve_orientation


@dataclass(frozen=True)
class Settings:
    box: float = 16.
    length: float = 16.
    diameter: float = 1.
    target_count: int = 125
    initial_count: int = 12
    batch_size: int = 4
    candidates: int = 32
    attempts: int = 3
    max_stages: int = 40
    relax_iterations: int = 200
    orientation_weight: float = 10.
    overlap_tolerance: float = 1/120
    a2_tolerance: float = .035
    seed: int = 2026092901
    solver: str = 'lbfgs-a2'
    a4_tolerance: float = 1e-4
    rotation_update: str = 'retraction'
    min_gap: float = 0.
    neighbor_mode: str = 'single'
    bootstrap_max_count: int = 48
    optimizer: str = 'gradient'
    neighbor_skin: float = .5
    adaptive_batches: bool = False
    max_relax_iterations: int = 0  # zero uses the initial per-trial budget
    retry_shake: float = 0.  # translational standard deviation in physical diameters
    insertion_policy: str = 'ranked'
    batch_reduction: str = 'halve'


def validate_settings(settings, target):
    s = settings
    numbers = [s.box, s.length, s.diameter, s.orientation_weight,
               s.overlap_tolerance, s.a2_tolerance]
    if not np.isfinite(numbers).all() or min(numbers) <= 0:
        raise ValueError('Positive finite dimensions, weights and tolerances required')
    for name in ('target_count', 'initial_count', 'batch_size', 'candidates',
                 'attempts', 'max_stages', 'relax_iterations', 'bootstrap_max_count'):
        value = getattr(s, name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    t = validate_a2(target)
    if s.solver not in ('lbfgs-a2', 'paper-a4'):
        raise ValueError('Unknown straight solver')
    if not np.isfinite(s.a4_tolerance) or s.a4_tolerance <= 0:
        raise ValueError('Positive finite A4 tolerance required')
    if not np.isfinite(s.min_gap) or s.min_gap < 0:
        raise ValueError('Minimum gap must be finite and nonnegative')
    if s.rotation_update not in ('retraction', 'geodesic'):
        raise ValueError('Unknown rotation update')
    if s.neighbor_mode not in ('single', 'multilevel'):
        raise ValueError('Unknown neighbor mode')
    if s.optimizer not in ('gradient', 'lbfgs'):
        raise ValueError('Unknown A4 optimizer')
    if s.insertion_policy not in ('ranked', 'random'):
        raise ValueError('Unknown insertion policy')
    if s.batch_reduction not in ('halve', 'one'):
        raise ValueError('Unknown batch reduction')
    if s.optimizer != 'gradient' and s.solver != 'paper-a4':
        raise ValueError('The optimizer option only selects the paper-a4 optimizer')
    if s.optimizer == 'lbfgs' and s.rotation_update != 'retraction':
        raise ValueError('L-BFGS uses normalized coordinates; geodesic update requires gradient optimizer')
    if not np.isfinite(s.retry_shake) or s.retry_shake < 0:
        raise ValueError('Retry shake must be finite and nonnegative')
    if s.retry_shake and s.solver != 'paper-a4':
        raise ValueError('Collective retry shake currently requires paper-a4')
    if not np.isfinite(s.neighbor_skin) or s.neighbor_skin <= 0:
        raise ValueError('Positive finite neighbor skin required')
    if (isinstance(s.max_relax_iterations, bool)
            or not isinstance(s.max_relax_iterations, (int, np.integer))
            or s.max_relax_iterations < 0):
        raise ValueError('max_relax_iterations must be a nonnegative integer')
    if s.max_relax_iterations and s.max_relax_iterations < s.relax_iterations:
        raise ValueError('Maximum relaxation budget is smaller than the initial budget')
    if s.min_gap and s.solver != 'paper-a4':
        raise ValueError('Explicit minimum gap currently requires paper-a4')
    if s.solver == 'paper-a4':
        # All relevant periodic images are enumerated by the contact kernel and
        # independent audit; small boxes are not claimed to replicate the paper.
        if min(s.target_count, s.initial_count) < 6:
            raise ValueError('paper-a4 requires at least six initial/target fibers')
    # An exact rank-one target forces every axis to the same direction. Its
    # self-image feasibility is then independent of placement or fiber count.
    values, vectors = np.linalg.eigh(t)
    if values[-1] >= 1.-1e-12:
        fiber = np.r_[np.full(3, s.box/2), vectors[:, -1], s.length, s.diameter+s.min_gap][None, :]
        if not audit_periodic_fibers(fiber, s.box,
                                    s.overlap_tolerance*s.diameter/(s.diameter+s.min_gap))['accepted']:
            raise ValueError('Exact rank-one target has self-image overlaps in this box; '
                             'cannot solve by translating fibers or refining voxels')
    return t


def certificate(fibers, box, target, overlap_tolerance, a2_tolerance, a4_tolerance=None,
                target_a4=None, min_gap=0.):
    audit = audit_periodic_fibers(fibers, box, overlap_tolerance)
    u = fibers[:, 3:6]
    error = float(np.linalg.norm(u.T@u/len(u)-target)/np.linalg.norm(target))
    result = dict(accepted=bool(audit['accepted'] and error <= a2_tolerance),
                  A2_error=error, audit=audit)
    if a4_tolerance is not None:
        from .sam_paper_reference import a4_error
        _, fourth, _ = resolve_orientation(target, target_a4)
        result['A4_error'] = a4_error(u, fourth)
        result['A4_tolerance'] = a4_tolerance
        result['accepted'] = bool(result['accepted'] and result['A4_error'] <= a4_tolerance)
    if min_gap:
        inflated = fibers.copy()
        inflated[:, 7] += min_gap
        gap_audit = audit_periodic_fibers(inflated, box,
                                         overlap_tolerance*fibers[0, 7]/inflated[0, 7])
        result.update(gap_audit=gap_audit, min_gap=min_gap)
        result['accepted'] = bool(result['accepted'] and gap_audit['accepted'])
    return result


def draw_directions(target, rng, count):
    # Residual moment may not be PSD for a small addition. Project only this
    # proposal distribution; the actual target used for acceptance never changes.
    values, vectors = np.linalg.eigh(.5*(target+target.T))
    values = np.maximum(values, 0.)
    if values.sum() < 1e-12:
        values[:] = 1.
    projected = (vectors*(values/values.sum()))@vectors.T
    transform = fused_angular_transform(tuple(projected.ravel()))
    raw = rng.normal(size=(count, 3))@transform.T
    norm = np.linalg.norm(raw, axis=1)
    if np.any(norm < 1e-14):
        raw[norm < 1e-14] = vectors[:, -1]
        norm = np.linalg.norm(raw, axis=1)
    return raw/norm[:, None]


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
    result = np.empty((len(accepted)+count, 8), dtype=float)
    result[:len(accepted)] = accepted
    box = np.full(3, s.box)
    for n in range(len(accepted), len(result)):
        current = result[:n]
        moment = current[:, 3:6].T@current[:, 3:6] if n else np.zeros((3, 3))
        moment4 = None
        if target_a4 is not None:
            from .sam_paper_reference import fourth_moment
            moment4 = n*fourth_moment(current[:, 3:6]) if n else np.zeros((3, 3, 3, 3))
        residual = (n+1)*target-moment
        directions = draw_directions(residual, rng, s.candidates)
        centers = rng.uniform(0., s.box, (s.candidates, 3))
        energies = candidate_contact_energies(current[:, :3], current[:, 3:6],
                                              centers, directions, s.length,
                                              s.diameter+s.min_gap, box)
        best_score, best = np.inf, 0
        for at, (direction, energy) in enumerate(zip(directions, energies)):
            if target_a4 is None:
                error = np.linalg.norm((moment+np.outer(direction, direction))/(n+1)-target)/np.linalg.norm(target)
            else:
                outer = np.outer(direction, direction)
                updated = (moment4+np.einsum('ij,kl->ijkl', outer, outer))/(n+1)
                error = np.linalg.norm(updated-target_a4)/np.linalg.norm(target_a4)
            score = energy/(n+1)+.5*s.orientation_weight*error**2
            if score < best_score:
                best_score, best = score, at
        result[n, :3] = centers[best]
        result[n, 3:6] = directions[best]
        result[n, 6:] = s.length, s.diameter
    return result


def perturb_trial(trial, settings, rng):
    """Perturb an uncommitted trial; endpoint rotation scale matches translation.

    Centers move by a Gaussian standard deviation of retry_shake diameters.
    Each tangent component has std 2*retry_shake/AR radians. The sphere update
    preserves unit axes; independent certification still decides acceptance.
    """
    from .sam_paper_reference import sphere_exp
    candidate = trial.copy()
    scale = settings.retry_shake*settings.diameter
    candidate[:, :3] = np.mod(candidate[:, :3]+rng.normal(size=(len(trial),3))*scale,
                              settings.box)
    tangent = rng.normal(size=(len(trial),3))*(2*scale/settings.length)
    candidate[:, 3:6] = sphere_exp(candidate[:, 3:6],tangent)
    return candidate


def generate_staged(settings, target, *, initial=None, progress=None, checkpoint=None,
                    target_a4=None, resume_state=None, wall_seconds=None):
    """Return last feasible table, possibly EMPTY/underfilled, with explicit status.

    checkpoint receives a copy only after the independent contact/A2 audit passes.
    Vf and voxel A2 must additionally be checked by the caller at the final grid.
    Failed trials and their statistics are retained in the returned event history.
    """
    s = settings
    target = validate_settings(s, target)
    if target_a4 is not None and s.solver != 'paper-a4':
        raise ValueError('An A4 target requires the paper-a4 solver; it cannot be ignored')
    orientation = None
    if s.solver == 'paper-a4':
        target, target_a4, orientation = resolve_orientation(target, target_a4)
    a4_tol = s.a4_tolerance if s.solver == 'paper-a4' else None
    def certify(table):
        return certificate(table, s.box, target, s.overlap_tolerance, s.a2_tolerance,
                           a4_tol, target_a4, s.min_gap)
    relaxer = relax_contact_energy
    extra = {}
    if s.solver == 'paper-a4':
        from .sam_paper_reference import relax_paper_reference
        relaxer = relax_paper_reference
        extra['a4_tolerance'] = s.a4_tolerance
        extra.update(target_a4=target_a4, rotation_update=s.rotation_update, min_gap=s.min_gap,
                     neighbor_mode=s.neighbor_mode, optimizer=s.optimizer,
                     neighbor_skin=s.neighbor_skin)
    rng = np.random.default_rng(s.seed)
    accepted = np.empty((0, 8)) if initial is None else np.array(initial, dtype=float, copy=True)
    if len(accepted):
        checked = certify(accepted)
        if not checked['accepted']:
            raise ValueError('Initial checkpoint is not contact/A2 feasible')
        if not np.allclose(accepted[:, 6:], [s.length, s.diameter], rtol=0, atol=1e-12):
            raise ValueError('Initial fiber dimensions differ from settings')
        if len(accepted) > s.target_count:
            raise ValueError('Initial checkpoint exceeds target count; no silent trimming')
    events = []
    batch = s.batch_size
    bootstrap_count = s.initial_count
    success_streak = 0
    next_attempt = 0
    if resume_state is not None:
        if not len(accepted):
            raise ValueError('Resume state requires an accepted geometry')
        rng.bit_generator.state = copy.deepcopy(resume_state['rng_state'])
        batch = int(resume_state['batch_size'])
        bootstrap_count = int(resume_state['bootstrap_count'])
        success_streak = int(resume_state.get('success_streak', 0))
        next_attempt = int(resume_state.get('next_attempt', 0))
        if (not 1 <= batch <= s.batch_size or bootstrap_count < 1 or success_streak < 0
                or not 0 <= next_attempt < s.attempts):
            raise ValueError('Invalid resume schedule')
    if wall_seconds is not None and (not np.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError('Positive finite wall budget required')
    started = time.perf_counter()

    def scheduler_state():
        return dict(rng_state=copy.deepcopy(rng.bit_generator.state), batch_size=batch,
                    bootstrap_count=bootstrap_count, success_streak=success_streak,
                    next_attempt=next_attempt)

    # Persist the RNG before the next trial. If interrupted mid-trial, restart
    # that whole trial from the last committed state rather than skipping draws.
    committed_resume = scheduler_state()

    class WallBudget(Exception):
        pass

    def relax_progress(row, stage, attempt, offset):
        if progress:
            progress(dict(event='relax_progress', stage=stage, attempt=attempt,
                          **dict(row, iteration=offset+row['iteration'])))
        if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
            raise WallBudget

    def emit(event):
        events.append(event)
        if progress:
            progress(event.copy())

    for stage in range(s.max_stages):
        if len(accepted) >= s.target_count:
            break
        bootstrap = len(accepted) == 0
        addition = min(bootstrap_count if bootstrap else batch, s.target_count-len(accepted))
        committed = False
        interrupted = False
        for attempt in range(next_attempt, s.attempts):
            if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                interrupted = True
                break
            next_attempt = attempt
            committed_resume = scheduler_state()
            emit(dict(event='trial_start', stage=stage, attempt=attempt,
                      accepted_count=len(accepted), addition=addition,
                      insertion_policy=s.insertion_policy))
            trial = propose(accepted, addition, target, s, rng, target_a4)
            if attempt > 0 and s.retry_shake:
                trial = perturb_trial(trial,s,rng)
                emit(dict(event='shake_trial',stage=stage,attempt=attempt,
                          translation_std_d=s.retry_shake,
                          rotation_component_std_rad=2*s.retry_shake*s.diameter/s.length))
            total_iterations = total_evaluations = 0
            limit = s.max_relax_iterations or s.relax_iterations
            candidate = trial
            while True:
                budget = min(max(s.relax_iterations,2*total_iterations),limit-total_iterations)
                try:
                    candidate, optimization = relaxer(
                        candidate, s.box, target, maxiter=budget,
                        orientation_weight=s.orientation_weight,
                        overlap_tolerance=s.overlap_tolerance, a2_tolerance=s.a2_tolerance,
                        progress=lambda row: relax_progress(row, stage, attempt, total_iterations),
                        **extra)
                except WallBudget:
                    break
                total_iterations += optimization['iterations']
                total_evaluations += optimization['evaluations']
                history = optimization.get('history', [])
                improving = False
                if len(history) >= 2:
                    def score(row):
                        return max(row['overlap_relative']/s.overlap_tolerance,
                                   row['A2_error']/s.a2_tolerance,
                                   row.get('A4_error', 0.)/s.a4_tolerance)
                    # Dense rearrangements can improve slowly over thousands
                    # of steps. Judge the entire chunk, not its last 100 steps.
                    earlier = history[0]
                    improving = (score(history[-1]) < .9*score(earlier)
                                 or ('energy' in history[-1] and 'energy' in earlier
                                     and history[-1]['energy'] < .98*earlier['energy']))
                recoverable = bool(optimization.get('recoverable_stop',False)
                                   and optimization['iterations'] > 0)
                if ((not improving and not recoverable)
                        or (optimization['iterations'] < budget and not recoverable)
                        or total_iterations >= limit
                        or optimization.get('optimizer_success', False)):
                    break
                emit(dict(event='extend_relaxation', stage=stage, attempt=attempt,
                          iterations=total_iterations, limit=limit,
                          reason='numerical_stop_with_gradient' if recoverable else 'improving_trial'))
            if wall_seconds is not None and time.perf_counter()-started >= wall_seconds:
                interrupted = True
                break
            checked = certify(candidate)
            # Guard dimensions/count even if a future optimizer changes API.
            shape_ok = (candidate.shape == trial.shape and np.array_equal(candidate[:, 6:], trial[:, 6:]))
            commit = bool(checked['accepted'] and shape_ok)
            if commit:
                accepted = candidate.copy()
                committed = True
                next_attempt = 0
                success_streak += 1
                if s.adaptive_batches and success_streak >= 2 and total_iterations <= s.relax_iterations//2:
                    batch = min(s.batch_size, max(batch+1, int(np.ceil(batch*1.5))))
                    success_streak = 0
                committed_resume = scheduler_state()
            else:
                success_streak = 0
                next_attempt = (attempt+1) % s.attempts
                committed_resume = scheduler_state()
            emit(dict(event='commit' if commit else 'rollback', stage=stage, attempt=attempt,
                      previous_count=len(trial)-addition, proposed_count=len(trial),
                      certificate=checked, optimizer_iterations=total_iterations,
                      optimizer_evaluations=total_evaluations,
                      optimizer_message=optimization.get('optimizer_message'),
                      returned_state=optimization.get('returned_state'),
                      final_gradient_inf_norm=optimization.get('final_gradient_inf_norm'),
                      fallback_step=optimization.get('fallback_step'),
                      **(dict(resume_state=committed_resume) if commit else {})))
            if commit:
                if checkpoint:
                    checkpoint(accepted.copy(), events[-1].copy())
                break
            if s.adaptive_batches and addition > 1:
                break  # reduce a failed batch before spending another full solve
        if interrupted:
            status = 'wall_budget_exhausted'
            break
        if not committed:
            next_attempt = 0
            # A few equal-weight directions can fail to fit A4 even without
            # contacts. Increase only an uncommitted bootstrap, within a named
            # budget, rather than weakening tolerances or discarding fibers.
            if (bootstrap and s.solver == 'paper-a4' and checked['audit']['accepted']
                    and checked.get('gap_audit', checked['audit'])['accepted']
                    and checked['A4_error'] > s.a4_tolerance
                    and addition < min(s.target_count, max(s.initial_count, s.bootstrap_max_count))):
                bootstrap_count = min(2*addition, s.target_count, max(s.initial_count, s.bootstrap_max_count))
                committed_resume = scheduler_state()
                emit(dict(event='grow_bootstrap', stage=stage, previous_count=addition,
                          next_count=bootstrap_count, reason='A4_fit_failed_with_contacts_satisfied'))
                continue
            if not bootstrap and addition > 1:
                batch = addition-1 if s.batch_reduction == 'one' else max(1, addition//2)
                committed_resume = scheduler_state()
                emit(dict(event='reduce_batch', stage=stage, batch=batch,
                          accepted_count=len(accepted), policy=s.batch_reduction))
                continue
            status = 'bootstrap_failed' if bootstrap else 'stalled'
            committed_resume = scheduler_state()
            break
    else:
        status = 'stage_budget_exhausted'
    if len(accepted) >= s.target_count:
        status = 'target_count_reached'
    elif 'status' not in locals():
        status = 'stage_budget_exhausted'
    final = (certify(accepted)
             if len(accepted) else None)
    return accepted, dict(status=status, accepted_count=len(accepted),
                          target_count=s.target_count, geometric_certificate=final,
                          orientation_target=orientation, vf_not_certified=True, events=events,
                          resume_state=committed_resume)
