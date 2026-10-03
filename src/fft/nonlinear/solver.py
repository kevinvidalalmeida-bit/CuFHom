"""Periodic small-strain FFT homogenization with history-dependent J2 plasticity.

Mandel order: 11,22,33,23,13,12. Stress/moduli use GPa, strain is dimensionless.
This API is independent of the six-load linear homogenization entrypoint.
"""
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
import csv
import copy
import hashlib
import json
import os
import platform
import tempfile
import time
import numpy as np

from cufhom_core.tensors.fft import cp
from cufhom.fft.linear.solver import compile_cfield_geometry
from cufhom.materials.elasticity import rotate_C_mandel, TI_stiffness_voigt, voigt_to_mandel
from cufhom.fft.operators import Timings, MixedProjector, IsotropicReference, DisplacementCoordinates
from cufhom.fft.krylov import NonlinearConvergenceError, _cg
from cufhom.materials.plasticity import J2Material, PlasticState, initial_state, update_cpu, tangent_cpu, commit_state, CuPyMaterial
from cufhom.fft.nonlinear.curves import loading_component, export_constitutive_curve, plot_constitutive_curve
from cufhom_core.tensors import fft as core_fft
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction


def _table(phase, ori, material, fiber):
    # Inputs are copied only if needed to establish immutable cache ownership.
    phase, ori = np.asarray(phase), np.asarray(ori)
    layout = compile_cfield_geometry(phase, ori)
    C = np.zeros((6, 6))
    C[:3, :3] = material.K - 2*material.mu/3
    C += np.eye(6)*2*material.mu
    matrices = [C]
    if len(layout['rotations']):
        Cf = voigt_to_mandel(TI_stiffness_voigt(**fiber))
        if np.min(np.linalg.eigvalsh(Cf)) <= 0:
            raise ValueError('Fiber stiffness must be positive definite')
        matrices.extend(rotate_C_mandel(Cf, R) for R in layout['rotations'])
    ids = np.zeros(phase.shape, dtype=np.int32)
    ids[phase == 1] = layout['inverse_groups'] + 1
    return ids, np.asarray(matrices)


def _host(x):
    return cp.asnumpy(x) if cp is not None and isinstance(x, cp.ndarray) else np.asarray(x)


def _fingerprint(phase, ori, configuration):
    h = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for a in (phase, ori):
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    return h.hexdigest()


def save_checkpoint(path, strain, state, fingerprint, history):
    """Atomic replace of a confirmed state only; never serializes Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
            temp = f.name
            np.savez(f, strain=_host(strain), plastic_strain=_host(state.plastic_strain),
                     equivalent=_host(state.equivalent),
                     metadata=json.dumps(dict(version=1, fingerprint=fingerprint, history=history)))
        os.replace(temp, path)
    finally:
        if temp is not None and os.path.exists(temp):
            os.unlink(temp)


def _ew_tolerance(error, previous_error, previous_eta, floor, target=None):
    """Safeguarded EW2-style forcing; controls inner CG, never acceptance.

    Constants are conservative project choices, not fitted material parameters.
    Reset the history at every load attempt and use legacy forcing for PTC.
    """
    if previous_error is None:
        eta = .1
    else:
        eta = .9*(error/previous_error)**1.5
        safeguard = .9*previous_eta**1.5
        if safeguard > .1:
            eta = max(eta, safeguard)
    if target is not None:
        # Avoid resolving a Newton equation far below the OUTER tolerance.
        # Both norms are physical/isometric, with the same fixed stress scale.
        # The nonlinear residual/line search is still evaluated independently.
        # Analogous to the final-target safeguard of PETSc's EW version 3.
        eta = max(eta, .5*target/error)
    return min(.3, max(floor, eta))


def annotate_pseudotime(history, strain_scale=.01):
    """Monotone path clock, including unloading; no physical rate dependence.

    tau = integral |d prescribed_Mandel_strain| / strain_scale. Rejected increments are
    absent from confirmed history and consequently cannot advance this clock.
    """
    tau, last = 0., None
    for row in history:
        value = row['target_strain']
        if last is not None:
            tau += abs(value-last)/strain_scale
        row['pseudo_time'] = tau
        row['load_factor'] = value/strain_scale
        last = value


def export_curve(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = result['history']
    axis = result['configuration']['axis']
    # Older Newton-only results may be re-exported without rerunning physics.
    for row in rows:
        row.setdefault('target_strain', row['strain'][axis])
        row.setdefault('ptc_iterations', 0)
        row.setdefault('ptc_retries', 0)
        row.setdefault('ptc_dt_final', 0.)
    annotate_pseudotime(rows, result.get('pseudo_time_strain_scale', .01))
    labels = ['11', '22', '33', '23', '13', '12']
    scalar = ['pseudo_time', 'load_factor', 'equivalent_plastic_mean', 'plastic_fraction', 'residual', 'macro_residual',
              'yield_violation', 'hill_mandel_error', 'plastic_dissipation',
              'newton_iterations', 'cg_iterations', 'ptc_iterations', 'ptc_retries', 'ptc_dt_final', 'seconds']
    with (directory/'curve.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step']+['strain_'+s for s in labels]+['stress_GPa_'+s for s in labels]+scalar)
        for i, r in enumerate(rows):
            writer.writerow([i]+r['strain']+r['stress']+[r[k] for k in scalar])
    export_constitutive_curve(result, directory)
    serializable = {k: v for k, v in result.items() if k not in ('fields', 'state')}
    (directory/'metadata.json').write_text(json.dumps(serializable, indent=2, allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plot_constitutive_curve(ax, result)
    ax.set_title('FFT J2 — parámetros demostrativos, sin daño ni rotura')
    fig.tight_layout()
    fig.savefig(directory/'curve.png', dpi=150)
    plt.close(fig)


def solve_nonlinear_homogenization(parameters):
    """Prescribe one mean Mandel strain with all other mean stresses zero.

    axis=0,1,2,3,4,5 selects 11,22,33,23,13,12, respectively. For shear,
    path/increments/pseudo_time_strain_scale use gamma/sqrt(2); history and
    fields remain Mandel. constitutive.csv and curve.png use gamma and tau.

    Required: phase_array (0=matrix,1=fiber), ori_array (...,3). Optional:
    matrix (J2Material fields), fiber (TI E_L,E_T,nu_LT,nu_TT,G_LT), backend
    ('scipy'/'cupy'), dtype, path=(0,.01), axis=0, initial_increment=.00025,
    min_increment=1e-7, rtol, max_newton=25, max_cg=1000, output_dir,
    checkpoint_path, restart_from, checkpoint_every=0, return_fields=False,
    profile=False, on_step (called only after commit). Optional pseudotime
    controls: pseudo_time_strain_scale=.01, initial_pseudo_step, min_pseudo_step.
    A pseudo step multiplies the strain scale; it replaces, not supplements,
    the corresponding strain increment. Defaults use formulation='displacement'
    and preconditioner='matrix', the verified fast G09 path; pass
    formulation='strain', preconditioner='none' for the direct reference.
    coordinate_pcg='auto' uses fused reference PCG for CuPy float32 displacement
    with matrix preconditioning; other configurations keep classical CG.
    reuse_trial_response=True avoids reevaluating accepted Newton candidates.
    Default inner_forcing='ew_target' on that fused route avoids oversolving
    near the final tolerance; other routes retain 'ew'. Final acceptance is
    unchanged. For the historical path select coordinate_pcg='classic',
    reuse_trial_response=False, inner_forcing='ew'.
    Inputs are never mutated.
    """
    old = core_fft.get_fft_backend_status()
    try:
        return _solve(parameters)
    finally:
        core_fft.set_fft_backend(old['requested'])
        core_fft.set_cupy_plan_mode(old['cupy_plan_mode'], preserve_cache=True)
        core_fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode'])


def _solve(p):
    if any(k in p for k in ('fiber_fraction_array', 'interface_normal_array', 'Cfield_array')):
        raise ValueError('Nonlinear v1 accepts binary voxels only; no composite constitutive mixing')
    backend = p.get('backend', 'scipy')
    if backend not in ('scipy', 'cupy') or (backend == 'cupy' and cp is None):
        raise ValueError('Use available backend scipy or cupy')
    xp = np if backend == 'scipy' else cp
    dtype = np.dtype(p.get('dtype', 'float64' if xp is np else 'float32'))
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64')
    phase = np.asarray(p['phase_array']).view()
    ori = np.asarray(p['ori_array']).view()
    if phase.ndim != 3 or min(phase.shape) < 2 or ori.shape != phase.shape+(3,):
        raise ValueError('Expected 3D phase and (...,3) orientations')
    if not np.isin(phase, [0, 1]).all() or not np.isfinite(ori).all():
        raise ValueError('Finite orientations and binary phases required')
    if np.any(np.linalg.norm(ori[phase == 1], axis=-1) < 1e-12):
        raise ValueError('Fiber orientations must be nonzero')
    phase.setflags(write=False)
    ori.setflags(write=False)
    material = J2Material(**p.get('matrix', {}))
    fiber = p.get('fiber', dict(E_L=70., E_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.))
    axis = int(p.get('axis', 0))
    path = np.asarray(p.get('path', [0., .01]), dtype=float)
    lengths = tuple(p.get('lengths', [1., 1., 1.]))
    pseudo_scale = float(p.get('pseudo_time_strain_scale', .01))
    if not np.isfinite(pseudo_scale) or pseudo_scale <= 0:
        raise ValueError('pseudo_time_strain_scale must be finite and positive')
    for a, b in [('initial_increment', 'initial_pseudo_step'), ('min_increment', 'min_pseudo_step')]:
        if a in p and b in p:
            raise ValueError('Specify only one of '+a+' and '+b)
    inc = float(p.get('initial_increment', float(p.get('initial_pseudo_step', .025))*pseudo_scale))
    minimum = float(p.get('min_increment', float(p.get('min_pseudo_step', 1e-5))*pseudo_scale))
    rtol = float(p.get('rtol', 1e-8 if dtype == np.float64 else 1e-5))
    max_newton, max_cg = int(p.get('max_newton', 25)), int(p.get('max_cg', 1000))
    cg_projection = p.get('cg_projection', 'galerkin')
    fused_cg = bool(p.get('fused_cg', True))
    coordinate_pcg = p.get('coordinate_pcg', 'auto')
    reuse_trial_response = bool(p.get('reuse_trial_response', True))
    if coordinate_pcg not in ('auto', 'classic', 'fused'):
        raise ValueError('coordinate_pcg must be auto, classic or fused')
    # Production default: the verified G09 fast path.  The strain coordinates
    # remain available as a transparent reference formulation.
    preconditioner = p.get('preconditioner', 'matrix')
    # The historical double-projection diagnostic is defined in strain
    # coordinates.  Keep it usable when a caller selects it explicitly,
    # while the normal Galerkin route defaults to the measured fast path.
    formulation = p.get('formulation', 'strain' if cg_projection == 'double' else 'displacement')
    if formulation not in ('strain', 'displacement'):
        raise ValueError('formulation must be strain or displacement')
    if formulation == 'displacement' and cg_projection != 'galerkin':
        raise ValueError('displacement formulation requires Galerkin CG')
    if preconditioner not in ('none', 'matrix'):
        raise ValueError("preconditioner must be 'none' or 'matrix'")
    fused_coordinate_compatible = (xp is cp and dtype == np.float32 and fused_cg
                                  and formulation == 'displacement' and preconditioner == 'matrix')
    if coordinate_pcg == 'fused' and not fused_coordinate_compatible:
        raise ValueError('Fused coordinate PCG requires CuPy float32, displacement, matrix and fused_cg')
    coordinate_pcg_used = ('fused' if coordinate_pcg != 'classic' and fused_coordinate_compatible
                           else 'classic')
    inner_forcing = p.get('inner_forcing', 'ew_target' if coordinate_pcg_used == 'fused' else 'ew')
    if inner_forcing not in ('legacy', 'ew', 'ew_target'):
        raise ValueError('inner_forcing must be legacy, ew or ew_target')
    if cg_projection not in ('double', 'galerkin'):
        raise ValueError('cg_projection must be double or galerkin')
    globalization = p.get('globalization', 'auto')
    ptc_initial = float(p.get('ptc_initial_dt', 10.))
    ptc_min, ptc_max = float(p.get('ptc_min_dt', 1e-8)), float(p.get('ptc_max_dt', 1e12))
    max_ptc = int(p.get('max_ptc', 60))
    explicit_steps = int(p.get('ptc_explicit_steps', 0))
    if not 0 <= explicit_steps <= 100:
        raise ValueError('ptc_explicit_steps must be between 0 and 100')
    if globalization not in ('auto', 'newton', 'ptc'):
        raise ValueError('globalization must be auto, newton or ptc')
    if not np.isfinite([ptc_initial, ptc_min, ptc_max]).all() or not 0 < ptc_min <= ptc_initial <= ptc_max or max_ptc < 1:
        raise ValueError('Invalid PTC pseudostep bounds')
    if int(p.get('checkpoint_every', 0)) < 0:
        raise ValueError('checkpoint_every must be nonnegative')
    if axis not in range(6) or path.ndim != 1 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('Invalid strain path or Mandel axis (expected 0..5)')
    if len(lengths) != 3 or not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise ValueError('Domain lengths must be finite positive values')
    if not np.isfinite([inc, minimum, rtol]).all() or not 0 < minimum <= inc or not 0 < rtol < 1:
        raise ValueError('Invalid increment bounds or tolerance')
    if min(max_newton, max_cg) < 1:
        raise ValueError('Iteration limits must be positive')
    timers = Timings(xp, p.get('profile', False))
    timers.sync()
    started = time.perf_counter()
    package_root = Path(__file__).resolve().parents[2]
    source_paths = [package_root/name for name in (
        'fft/nonlinear/solver.py', 'fft/nonlinear/krylov.py',
        'fft/nonlinear/curves.py', 'fft/operators.py', 'fft/krylov.py',
        'fft/linear/solver.py', 'materials/plasticity.py', 'materials/elasticity.py')]
    source_paths += [Path(core_fft.__file__), Path(core_fft.__file__).with_name('objects.py')]
    source_hashes = {str(f.relative_to(package_root.parent)):
                     hashlib.sha256(f.read_bytes()).hexdigest() for f in source_paths}
    core_fft.set_fft_backend(backend)
    core_fft.set_cupy_plan_mode('manual', preserve_cache=True)
    core_fft.set_cupy_rfft_batch_mode(p.get('cupy_rfft_batch_mode', 'auto'))
    config = dict(matrix=asdict(material), fiber=fiber, axis=axis, lengths=lengths,
                  shape=list(phase.shape), conventions='Mandel 11,22,33,23,13,12; GPa',
                  model='small_strain_J2_linear_isotropic_hardening', calibrated=False)
    fingerprint = _fingerprint(phase, ori, config)
    with timers.measure('geometry_and_transfer'):
        ids_cpu, table_cpu = _table(phase, ori, material, fiber)
        # J2 positive-hardening tangents are bounded above by their elastic
        # stiffness. P is orthogonal, so this also bounds P C_alg P.
        elastic_bound = float(np.max(np.linalg.eigvalsh(table_cpu)))
        ids, table = xp.asarray(ids_cpu), xp.asarray(table_cpu, dtype=dtype)
        state = initial_state(phase.shape, xp, dtype)
        strain = xp.zeros((6,)+phase.shape, dtype=dtype)
    del ids_cpu, table_cpu
    control = np.zeros(6, dtype=bool)
    control[axis] = True
    projector = MixedProjector(phase.shape, lengths, control, xp, timers,
                               spectral_fusion=p.get('spectral_fusion', True))
    reference = None if preconditioner == 'none' else IsotropicReference(
        projector, material.K, material.mu)
    coordinates = DisplacementCoordinates(projector) if formulation == 'displacement' else None
    coordinate_engine = None
    if coordinate_pcg_used == 'fused':
        from cufhom.fft.nonlinear.krylov import FusedCoordinatePCG
        coordinate_engine = FusedCoordinatePCG(coordinates)
    gpu = CuPyMaterial(cp, material, ids, table, dtype) if xp is not np else None
    history, rejected = [], []
    current = 0.
    previous_delta, previous_step = None, 0.
    if p.get('restart_from'):
        with np.load(p['restart_from'], allow_pickle=False) as ck:
            meta = json.loads(str(ck['metadata']))
            if meta['version'] != 1 or meta['fingerprint'] != fingerprint:
                raise ValueError('Checkpoint geometry/material/loading mismatch')
            for key, shape in [('strain', strain.shape), ('plastic_strain', strain.shape),
                               ('equivalent', phase.shape)]:
                if ck[key].shape != shape or not np.isfinite(ck[key]).all():
                    raise ValueError('Invalid checkpoint field '+key)
            strain = xp.asarray(ck['strain'], dtype=dtype)
            state = PlasticState(xp.asarray(ck['plastic_strain'], dtype=dtype),
                                 xp.asarray(ck['equivalent'], dtype=dtype))
            if np.min(ck['equivalent']) < 0:
                raise ValueError('Checkpoint equivalent plastic strain is negative')
            history = meta['history']
            current = history[-1]['strain'][axis]
    if abs(path[0]-current) > 1e-9:
        raise ValueError('Path must begin at zero, or at the checkpoint prescribed Mandel strain')
    current = float(path[0])
    for row in history:
        row.setdefault('target_strain', row['strain'][axis])
    annotate_pseudotime(history, pseudo_scale)

    def dot(a, b):
        with timers.measure('reductions'):
            # float64 accumulation; multiplication is promoted too, without
            # retaining double-sized fields (einsum CPU, ReductionKernel GPU).
            if xp is np:
                return float(np.einsum('i,i->', a.ravel(), b.ravel(), dtype=np.float64))
            reduction.dot_kernel((reduction.blocks,), (reduction.threads,),
                                 (cp.ascontiguousarray(a), cp.ascontiguousarray(b),
                                  reduction.partial, np.int64(a.size)))
            scalar = cp.sum(reduction.partial, dtype=cp.float64)
        # Keep the host transfer outside the device reduction timing.
        return float(scalar)

    def coordinate_dot(a, b):
        return dot(a.view(dtype), b.view(dtype))

    reduction = None if xp is np else CupyWarpCGReduction(
        SimpleNamespace(N=phase.shape, val=strain), max_blocks=16384)
    def norm(a):
        return np.sqrt(dot(a, a)/phase.size)

    def update(e):
        with timers.measure('constitutive'):
            return gpu.update(e, state) if gpu else update_cpu(e, state, material, ids, table)

    def tangent(v, response):
        with timers.measure('tangent'):
            return gpu.tangent(v, response) if gpu else tangent_cpu(v, response, material, ids, table)

    def equilibrium(e):
        response = update(e)
        residual = projector(response.stress)
        # Fixed material stress scale avoids artificial convergence near zero.
        return response, residual, norm(residual)/material.yield_stress

    def attempt(target, use_ptc=False):
        delta = target-current
        if p.get('predictor', True) and previous_delta is not None and delta*previous_step > 0:
            e = strain + (delta/previous_step)*previous_delta
        else:
            e = strain.copy()
            e[axis] += delta
        # Fixed macroscopic strain and admissibility must not drift over a path.
        e = projector(e)
        e[axis] += target
        cg_total = 0
        previous_error, previous_eta = None, None
        linear_solves = []
        dt, retries = ptc_initial, 0
        explicit_accepted, explicit_reductions = 0, 0
        if use_ptc and explicit_steps:
            # Optional forward-Euler relaxation for the fictitious evolution
            # E de/dtau = -P sigma. Conservative spectral cap, then residual
            # backtracking; never update the physical plastic state here.
            explicit_dt = min(dt, .8*material.E/elastic_bound)
            for _ in range(explicit_steps):
                response, residual, error = equilibrium(e)
                if error <= rtol:
                    break
                for backtrack in range(12):
                    trial_dt = explicit_dt * .5**backtrack
                    candidate = e - (trial_dt/material.E)*residual
                    candidate = projector(candidate)
                    candidate[axis] += target
                    trial_response, trial_residual, trial_error = equilibrium(candidate)
                    accepted = np.isfinite(trial_error) and (trial_error < error or trial_error <= rtol)
                    del trial_response, trial_residual
                    if accepted:
                        e = candidate
                        explicit_accepted += 1
                        explicit_reductions += backtrack
                        explicit_dt = min(trial_dt*1.5, .8*material.E/elastic_bound)
                        break
                else:
                    break  # Switch to implicit PTC; no false convergence.
        limit = max_ptc if use_ptc else max_newton
        # Only reuse an accepted evaluation within this attempt, against the
        # same confirmed plastic state. Never carry it across commits/rejects.
        accepted_equilibrium = None
        for iteration in range(limit+1):
            if accepted_equilibrium is None:
                response, residual, error = equilibrium(e)
            else:
                response, residual, error = accepted_equilibrium
                accepted_equilibrium = None
            if not np.isfinite(error):
                raise NonlinearConvergenceError('Nonfinite Newton residual')
            if error <= rtol:
                diagnostics = dict(globalization='ptc' if use_ptc else 'newton',
                                   ptc_iterations=iteration if use_ptc else 0,
                                   ptc_retries=retries, ptc_dt_final=dt if use_ptc else 0.,
                                   explicit_steps=explicit_accepted,
                                   linear_solves=linear_solves,
                                   explicit_backtracks=explicit_reductions)
                return e, response, error, iteration, cg_total, diagnostics
            if iteration == limit:
                break
            inner = min(.01, max(1e-5 if dtype == np.float32 else 1e-10, .1*np.sqrt(error)))
            if inner_forcing in ('ew', 'ew_target') and not use_ptc:
                inner = _ew_tolerance(error, previous_error, previous_eta,
                                      1e-5 if dtype == np.float32 else 1e-10,
                                      rtol if inner_forcing == 'ew_target' else None)
            for retry in range(9 if use_ptc else 1):
                # PTC: (P C_alg P + E/dt I_admissible) de = -P sigma.
                # This mass term regularizes the iteration, NOT the constitutive
                # law or the equilibrium acceptance test above.
                shift = material.E/dt if use_ptc else 0.
                def action(v):
                    out = tangent(v, response)
                    if cg_projection == 'double':
                        out = projector(out)
                    if shift:
                        out += shift*v
                    return out
                try:
                    with timers.measure('cg_total_inclusive'):
                        if reference is not None:
                            reference.set_shift(shift, dtype)
                        if coordinates is not None:
                            def reduced_action(z):
                                v = coordinates.strain(z)
                                out = coordinates.adjoint(tangent(v, response))
                                if shift:
                                    out += shift*z
                                return out
                            reduced_reference = (None if reference is None else
                                lambda z: coordinates.precondition(z, reference))
                            rhs = coordinates.adjoint(-residual)
                            if coordinate_engine is None:
                                z, count = _cg(reduced_action, rhs, coordinate_dot, inner, max_cg, xp,
                                    fused=fused_cg, precondition=reduced_reference)
                            else:
                                z, count = coordinate_engine.solve(reduced_action, rhs,
                                    coordinate_dot, inner, max_cg, reference)
                            correction = coordinates.strain(z)
                        else:
                            correction, count = _cg(action, -residual, dot, inner, max_cg, xp,
                                project=projector, fused=fused_cg, precondition=reference)
                    cg_total += count
                    linear_solves.append(dict(residual_before=error, relative_tolerance=inner,
                                              cg_iterations=count, shift=shift))
                    correction = projector(correction)
                    accepted = False
                    for backtrack in range(12):
                        candidate = e + (0.5**backtrack)*correction
                        trial_response, trial_residual, trial_error = equilibrium(candidate)
                        if trial_error <= (1-1e-4*0.5**backtrack)*error or trial_error <= rtol:
                            e, accepted = candidate, True
                            if reuse_trial_response:
                                accepted_equilibrium = trial_response, trial_residual, trial_error
                            del trial_response, trial_residual
                            break
                        del trial_response, trial_residual
                    if not accepted:
                        raise NonlinearConvergenceError('Equilibrium line search failed')
                except NonlinearConvergenceError:
                    if not use_ptc or retry == 8 or dt*.25 < ptc_min:
                        raise
                    dt *= .25
                    retries += 1
                    continue
                if use_ptc:
                    # Switched evolution relaxation; bounded growth avoids
                    # removing stabilization abruptly after a lucky iteration.
                    dt = min(ptc_max, dt*min(10., max(1., error/max(trial_error, rtol))))
                break
            previous_error, previous_eta = error, inner
        raise NonlinearConvergenceError('PTC iteration limit' if use_ptc else 'Newton iteration limit')

    def record(e, response, error, it, cg, elapsed):
        with timers.measure('diagnostics_and_transfer'):
            mean_e = _host(xp.mean(e, axis=(1, 2, 3), dtype=xp.float64))
            mean_s = _host(xp.mean(response.stress, axis=(1, 2, 3), dtype=xp.float64))
            macro_error = np.linalg.norm(mean_s[~control])/material.yield_stress
            dev = response.stress.copy()
            dev[:3] -= xp.sum(dev[:3], axis=0)/3
            q = xp.sqrt(1.5*xp.sum(dev*dev, axis=0))
            violation = xp.where(ids == 0, xp.maximum(q-material.yield_stress-
                                material.hardening*(state.equivalent+response.dp), 0), 0)
            yield_error = float(xp.max(violation))/material.yield_stress
            hm = abs(dot(response.stress, e)/phase.size - float(mean_s@mean_e))
            hm /= material.yield_stress*max(norm(e), 1e-12)
            dissipation = material.yield_stress*float(xp.mean(response.dp, dtype=xp.float64))
            if macro_error > rtol or yield_error > max(rtol, 5e-6 if dtype == np.float32 else 1e-12):
                raise NonlinearConvergenceError('Macro stress or local yield audit failed')
            if dissipation < 0 or not np.isfinite(hm) or hm > 5*rtol:
                raise NonlinearConvergenceError('Invalid energy diagnostic')
            return dict(strain=mean_e.tolist(), stress=mean_s.tolist(), residual=error,
                        macro_residual=macro_error, yield_violation=yield_error,
                        hill_mandel_error=hm, plastic_dissipation=dissipation,
                        equivalent_plastic_mean=float(xp.mean(state.equivalent+response.dp, dtype=xp.float64)),
                        plastic_fraction=float(xp.mean((state.equivalent+response.dp)>1e-10)),
                        newton_iterations=it, cg_iterations=cg, seconds=elapsed,
                        globalization='initial', ptc_iterations=0, ptc_retries=0, ptc_dt_final=0.)

    if not history:
        response, _, error = equilibrium(strain)
        history.append(record(strain, response, error, 0, 0, 0.))
        history[-1]['target_strain'] = current
        annotate_pseudotime(history, pseudo_scale)
        del response
    checkpoint = p.get('checkpoint_path')
    for endpoint in path[1:]:
        step = inc
        while abs(endpoint-current) > 1e-12:
            target = current + np.sign(endpoint-current)*min(step, abs(endpoint-current))
            timers.sync()
            step_started = time.perf_counter()
            try:
                fallback_reason = None
                try:
                    e, response, error, it, cg, diagnostics = attempt(target, globalization == 'ptc')
                except NonlinearConvergenceError as exc:
                    if globalization != 'auto':
                        raise
                    fallback_reason = str(exc)
                # Leave the except block before PTC: its traceback can retain
                # several full GPU fields belonging to the failed Newton solve.
                if fallback_reason is not None:
                    e, response, error, it, cg, diagnostics = attempt(target, True)
                    diagnostics['ptc_fallback_reason'] = fallback_reason
                timers.sync()
                row = record(e, response, error, it, cg, time.perf_counter()-step_started)
                row.update(diagnostics)
                row['target_strain'] = float(target)
            except NonlinearConvergenceError as exc:
                rejected.append(dict(from_strain=current, target=float(target), reason=str(exc)))
                if p.get('on_reject'):
                    p['on_reject'](dict(rejected[-1]))
                step = abs(target-current)/2
                if step < minimum:
                    if checkpoint:
                        save_checkpoint(checkpoint, strain, state, fingerprint, history)
                    raise NonlinearConvergenceError(
                        f'{exc}; minimum increment reached at {current}; {len(history)-1} confirmed steps') from exc
                continue
            previous_delta, previous_step = e-strain, float(target-current)
            state = commit_state(state, response)
            strain, current = e, float(target)
            history.append(row)
            annotate_pseudotime(history, pseudo_scale)
            del response
            if p.get('on_step'):
                p['on_step'](copy.deepcopy(row))
            every = int(p.get('checkpoint_every', 0))
            if checkpoint and every > 0 and (len(history)-1) % every == 0:
                save_checkpoint(checkpoint, strain, state, fingerprint, history)
            # Grow only back to the requested resolution, never skip the path.
            if it <= 6:
                step = min(inc, step*1.5)
    if checkpoint:
        with timers.measure('checkpoint_transfer_io'):
            save_checkpoint(checkpoint, strain, state, fingerprint, history)
    timers.sync()
    result = dict(history=history, rejected_steps=rejected, configuration=config,
                  loading=loading_component(axis),
                  fingerprint=fingerprint, backend=backend, dtype=dtype.name, rtol=rtol,
                  source_sha256=source_hashes,
                  cupy_rfft_batch_mode=core_fft.CUPY_RFFT_BATCH_MODE,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   cupy=cp.__version__ if cp is not None else None),
                  predictor=bool(p.get('predictor', True)),
                  cg_projection=cg_projection, fused_cg=fused_cg,
                  coordinate_pcg=coordinate_pcg_used, coordinate_pcg_requested=coordinate_pcg,
                  reuse_trial_response=reuse_trial_response,
                  preconditioner=preconditioner,
                  formulation=formulation,
                  spectral_fusion=projector.spectral_fusion,
                  inner_forcing=inner_forcing,
                  initial_increment=inc, min_increment=minimum, path=path.tolist(),
                  pseudo_time_strain_scale=pseudo_scale,
                  globalization=globalization,
                  ptc=dict(initial_dt=ptc_initial, min_dt=ptc_min, max_dt=ptc_max,
                           max_iterations=max_ptc, explicit_steps=explicit_steps),
                  wall_seconds=time.perf_counter()-started,
                  timings=timers.seconds, timing_calls=timers.calls,
                  timings_synchronized=bool(timers.enabled),
                  sampled_device_peak_bytes=timers.sampled_device_peak,
                  gpu_pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes()) if xp is not np else 0,
                  stress_scale_GPa=material.yield_stress)
    if p.get('return_fields', False):
        response = update(strain)
        result['fields'] = dict(strain=_host(strain), stress=_host(response.stress))
        result['state'] = PlasticState(_host(state.plastic_strain), _host(state.equivalent))
    if p.get('output_dir'):
        export_curve(result, p['output_dir'])
    return result
