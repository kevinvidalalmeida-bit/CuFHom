"""Inertia-scaled L-BFGS for rigid translations and normalized fiber axes.

The orientation coordinates have initial radius sqrt(epsilon), so the first
quasi-Newton step has the same rotational metric as the SAM gradient step.
Physical length/diameter are fixed. Optimizer stopping is never a certificate.
"""
import numpy as np
from scipy.optimize import minimize


def normalized_coordinates(x):
    values = np.asarray(x).reshape(-1, 6)
    raw = values[:, 3:]
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError('Degenerate orientation coordinate')
    return np.ascontiguousarray(values[:, :3]), np.ascontiguousarray(raw/norms[:, None]), norms


def coordinate_gradient(gc, gu, u, norms):
    tangent = gu-u*np.sum(gu*u, axis=1)[:, None]
    return np.column_stack((gc, tangent/norms[:, None])).ravel()


def minimize_a4(c, u, evaluate, epsilon, target, *, maxiter, overlap_tolerance,
                a2_tolerance, a4_tolerance, progress=None):
    x0 = np.column_stack((c, np.sqrt(epsilon)*u)).ravel()
    last_x = None
    last_value = None
    evaluations = 0
    history = []
    best_score, best_c, best_u = np.inf, c.copy(), u.copy()
    lowest_energy, energy_c, energy_u = np.inf, c.copy(), u.copy()
    energy_x = x0.copy()
    iterations = 0

    def fun(x):
        nonlocal last_x, last_value, evaluations
        if last_x is not None and np.array_equal(x, last_x):
            return last_value[0], last_value[1]
        try:
            cc, uu, norms = normalized_coordinates(x)
        except ValueError:
            return np.inf, np.zeros_like(x)
        energy, gc, gu, overlap, a4 = evaluate(cc, uu)
        evaluations += 1
        g = coordinate_gradient(gc, gu, uu, norms)
        last_x, last_value = x.copy(), (energy, g, overlap, a4)
        return energy, g

    def record(x, final=False):
        nonlocal best_score, best_c, best_u, lowest_energy, energy_c, energy_u, energy_x
        fun(x)
        cc, uu, norms = normalized_coordinates(x)
        energy, _, overlap, a4 = last_value
        a2 = float(np.linalg.norm(uu.T@uu/len(uu)-target)/np.linalg.norm(target))
        score = max(overlap/overlap_tolerance, a2/a2_tolerance, a4/a4_tolerance)
        if score < best_score:
            best_score, best_c, best_u = score, cc.copy(), uu.copy()
        if energy < lowest_energy:
            lowest_energy, energy_c, energy_u = energy, cc.copy(), uu.copy()
            energy_x = x.copy()
        if iterations % 25 == 0 or score <= 1 or final:
            row = dict(iteration=iterations, energy=energy, overlap_relative=overlap,
                       A2_error=a2, A4_error=a4, evaluations=evaluations,
                       axis_coordinate_radius_min=float(norms.min()),
                       axis_coordinate_radius_max=float(norms.max()))
            if not history or row != history[-1]:
                history.append(row)
                if progress:
                    progress(row.copy())
        return score

    def callback(x):
        nonlocal iterations
        iterations += 1
        if record(x) <= 1:
            raise StopIteration

    score = record(x0)
    message = 'constraints_satisfied'
    gradient_inf_norm = float(np.max(np.abs(last_value[1])))
    recoverable = False
    fallback_step = None
    if score > 1:
        result = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                          options=dict(maxiter=maxiter, maxfun=32*(maxiter+1),
                                       maxls=30, maxcor=12, ftol=0., gtol=1e-12))
        record(result.x, final=True)
        message = 'constraints_satisfied' if best_score <= 1 else str(result.message)
        gradient_inf_norm = float(np.max(np.abs(last_value[1])))
        # Floating-point relative-reduction and line-search stops are not
        # stationary-point certificates. Clear the quasi-Newton history on a
        # subsequent solve when a significant gradient and budget remain.
        numerical_stop = ('RELATIVE REDUCTION' in message or 'ABNORMAL' in message)
        if best_score > 1 and numerical_stop and gradient_inf_norm > 1e-10 and iterations < maxiter:
            # Verify actual descent before requesting another chunk. For almost
            # parallel rods a full angular step may cross a sharp contact ridge;
            # a center-only step leaves A2/A4 unchanged and can still remove
            # contacts. Both alternatives must decrease the same objective.
            base = energy_x.copy()
            base_energy, base_gradient = fun(base)
            minimum_change = 8*np.finfo(float).eps*max(1.,abs(base_energy))
            for kind in ('full_gradient','centers'):
                direction = -base_gradient.copy()
                if kind == 'centers':
                    direction.reshape(-1,6)[:,3:] = 0.
                slope = float(base_gradient@direction)
                if slope >= -1e-30:
                    continue
                step = .3
                for _ in range(32):
                    trial = base+step*direction
                    value,_ = fun(trial)
                    if (value < lowest_energy-minimum_change
                            and value <= base_energy+1e-4*step*slope):
                        iterations += 1
                        record(trial,final=True)
                        recoverable = best_score > 1
                        fallback_step = kind
                        break
                    step *= .5
                if fallback_step:
                    break
            message = 'constraints_satisfied' if best_score <= 1 else message
            gradient_inf_norm = float(np.max(np.abs(fun(energy_x)[1])))
    returned_state = 'best_feasible' if best_score <= 1 else 'lowest_energy_infeasible'
    if best_score > 1:
        # A migration path can reduce total energy while its largest overlap
        # temporarily grows. Returning the smallest-max-overlap state here
        # would undo that progress before the caller's continuation chunk.
        best_c, best_u = energy_c, energy_u
    return best_c, best_u, dict(history=history, iterations=iterations,
                                optimizer_message=message,
                                returned_state=returned_state, lowest_energy=float(lowest_energy),
                                final_gradient_inf_norm=gradient_inf_norm,
                                recoverable_stop=bool(recoverable),fallback_step=fallback_step)
