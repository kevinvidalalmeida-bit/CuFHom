"""Linear elasticity homogenization (GaNi, primal) on NumPy or CuPy fields.

One CG solve per macroscopic load (or a Krylov adapter supplied by the
caller), then postprocessing of the effective stiffness. Optional outputs:
affine sensitivities, streamed strain/stress fields and partial columns.
"""
import time

import numpy as np
import cufhom_core.projections as proj
from cufhom_core.materials import Material
from cufhom_core.postprocess import postprocess
from cufhom_core.general.solver import linear_solver
from cufhom_core.general.cupy_capacity import host_iterate_required,library_capacity_required,current_memory_pool
from cufhom_core.general.base import Timer
from cufhom_core.tensors import Tensor, DFT, Operator
from cufhom_core.tensors.fft import (
    CUPY_AVAILABLE,
    cp,
    cupy_synchronize,
    get_array_module,
    get_fft_backend,
    to_backend_array,
    to_host_array,
)


PROFILE_COMPONENT_LABELS = ("FN", "hG", "FiN", "A")


def _new_component_stats():
    return {label: {"seconds": 0.0, "calls": 0} for label in PROFILE_COMPONENT_LABELS}


class TimedCallable:
    def __init__(self, op, label, stats):
        self.op = op
        self.label = label
        self.stats = stats
        self.name = getattr(op, "name", label)

    def __call__(self, x):
        cupy_synchronize()
        t0 = time.perf_counter()
        out = self.op(x)
        cupy_synchronize(out)
        dt = time.perf_counter() - t0
        self.stats[self.label]["seconds"] += dt
        self.stats[self.label]["calls"] += 1
        return out

    def __getattr__(self, item):
        return getattr(self.op, item)


def _profile_enabled(pb):
    return bool(pb.solve.get("profile_timing", False))


def _stage_tic():
    cupy_synchronize()
    return time.perf_counter()


def _stage_toc(t0):
    cupy_synchronize()
    return time.perf_counter() - t0


def _resolve_affine_sensitivity_batch_size(pb, field_shape, nloads):
    """Choose a bounded GPU RHS batch without risking a large-grid OOM."""
    requested = int(pb.solve.get("affine_sensitivity_batch_size", 0))
    if requested < 0:
        raise ValueError("affine_sensitivity_batch_size debe ser no negativo.")
    if requested > 0:
        return max(1, min(requested, int(nloads)))
    if (
        not CUPY_AVAILABLE
        or cp is None
        or str(pb.solve.get("fft_backend", "")).lower() != "cupy"
    ):
        return 1
    nvox = int(np.prod(field_shape))
    # Small grids are faster with the already measured scalar path.
    if nvox < 1_000_000:
        return 1
    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        rhs_bytes = 6 * nvox * np.dtype(_solver_real_dtype(pb)).itemsize
        # B, x, R, P, AP plus FFT/operator temporaries. Keep this budget
        # deliberately below half of the currently free VRAM.
        estimated_per_rhs = 5.0 * float(rhs_bytes)
        memory_limited = int((0.40 * float(free_bytes)) // estimated_per_rhs)
        return max(1, min(2, int(nloads), memory_limited))
    except Exception:
        return 1


def _release_affine_sensitivity_batch_memory(pb):
    """Return dead batched CG workspaces to CuPy between sensitivity batches."""
    if (
        not CUPY_AVAILABLE
        or cp is None
        or str(pb.solve.get("fft_backend", "")).lower() != "cupy"
    ):
        return
    try:
        cp.cuda.Stream.null.synchronize()
        current_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        # Pool trimming is an optimization; the solve result remains valid if
        # a backend/version does not expose one of these pool APIs.
        return


def _solver_real_dtype(pb):
    dtype = pb.solver.get("real_dtype", None)
    if dtype is None:
        dtype = pb.solve.get("real_dtype", None)
    if dtype is None:
        dtype = "float32" if pb.solve.get("fft_backend", "") == "cupy" else "float64"
    return np.dtype(dtype)


def _instrument_afun_for_profile(Afun, GN, A, stats):
    if len(getattr(GN, "mat_rev", [])) != 1 or len(GN.mat_rev[0]) != 3:
        return Afun
    if str(get_fft_backend()) == 'cupy':
        # Wrapping individual tensors disables Operator's native GPU fast path.
        # Instrument the same raw operations inside that fast path instead.
        instrumented = Operator(name=Afun.name, mat_rev=Afun.mat_rev)
        instrumented._component_profile = stats
        return instrumented

    timed_FN = TimedCallable(GN.mat_rev[0][0], "FN", stats)
    timed_hG = TimedCallable(GN.mat_rev[0][1], "hG", stats)
    timed_FiN = TimedCallable(GN.mat_rev[0][2], "FiN", stats)
    timed_GN = Operator(name=GN.name, mat_rev=[[timed_FN, timed_hG, timed_FiN]])
    timed_A = TimedCallable(A, "A", stats)
    return Operator(name=Afun.name, mat_rev=[[timed_A, timed_GN]])


def _extract_perf_counter_time(info):
    try:
        return float(info["time"][-1][1])
    except Exception:
        return None


def _build_load_timing_payload(iL, load_wall_s, rhs_build_s, info, stats):
    return {
        "load_id": int(iL),
        "load_wall_s": float(load_wall_s),
        "rhs_build_s": float(rhs_build_s),
        "linear_solver_perf_counter_s": _extract_perf_counter_time(info),
        "cg_iterations": int(info.get("kit", 0)),
        "final_norm_res": float(info.get("norm_res", np.nan)),
        "final_norm_res_rel": float(info.get("norm_res_rel", np.nan)),
        "rhs_norm": float(info.get("rhs_norm", np.nan)),
        "converged": bool(info.get("converged", False)),
        "true_norm_res_rel": info.get('true_norm_res_rel'),
        "true_residual_converged": info.get('true_residual_converged'),
        "components": {
            label: {
                "seconds": float(stats[label]["seconds"]),
                "calls": int(stats[label]["calls"]),
            }
            for label in PROFILE_COMPONENT_LABELS
        },
    }


def _solver_info_payload(info, load_ids):
    norm_per_rhs = info.get("norm_res_per_rhs", None)
    if norm_per_rhs is not None:
        norm_per_rhs = [float(value) for value in np.asarray(norm_per_rhs).ravel()]
    rel_per_rhs = info.get("norm_res_rel_per_rhs", None)
    if rel_per_rhs is not None:
        rel_per_rhs = [float(value) for value in np.asarray(rel_per_rhs).ravel()]
    rhs_norm_per_rhs = info.get("rhs_norm_per_rhs", None)
    if rhs_norm_per_rhs is not None:
        rhs_norm_per_rhs = [float(value) for value in np.asarray(rhs_norm_per_rhs).ravel()]
    converged_per_rhs = info.get("converged_per_rhs", None)
    if converged_per_rhs is not None:
        converged_per_rhs = [bool(value) for value in np.asarray(converged_per_rhs).ravel()]

    payload = {
        "load_ids": [int(value) for value in load_ids],
        "cg_iterations": int(info.get("kit", 0)),
        "final_norm_res": float(info.get("norm_res", np.nan)),
        "final_norm_res_rel": float(info.get("norm_res_rel", np.nan)),
        "rhs_norm": float(info.get("rhs_norm", np.nan)),
        "threshold": float(info.get("threshold", np.nan)),
        "converged": bool(info.get("converged", False)),
        "hit_maxiter": bool(info.get("hit_maxiter", False)),
    }
    if norm_per_rhs is not None:
        payload["final_norm_res_per_rhs"] = norm_per_rhs
    if 'true_norm_res_rel' in info:
        payload['true_norm_res_rel'] = info['true_norm_res_rel']
        payload['true_residual_converged'] = info['true_residual_converged']
    for key in ('coordinate_iterations', 'correction_iterations', 'warm_start_requested', 'warm_start_accepted',
                'initial_norm_res_rel', 'linear_final_strain_reuse', 'linear_final_audit_reuse',
                'linear_final_spectrum_reuse', 'linear_fused_fft', 'linear_fused_fft_config',
                'linear_coordinate_rhs', 'linear_coordinate_layout', 'physical_rhs_host', 'retained_solution_host',
                'physical_audit_tightenings', 'physical_audit_ratios', 'physical_audit_margin',
                'projection_continuations','host_cg_iterate','resident_cg_iterate',
                'active_state_strategy','active_state_transfer_bytes',
                'linear_working_dtype','fp64_evaluations','reliable_drop','energy_G0',
                'criterion','energy_rtol','energy','energy_threshold','phase_spectra_rhs','evaluation',
                'replacements','restarts','candidate_checks','recursive_norm_res',
                'linear_cg_control','control_queries','device_control_iterations'):
        if key in info:
            payload[key] = info[key]
    if rel_per_rhs is not None:
        payload["final_norm_res_rel_per_rhs"] = rel_per_rhs
    if rhs_norm_per_rhs is not None:
        payload["rhs_norm_per_rhs"] = rhs_norm_per_rhs
    if converged_per_rhs is not None:
        payload["converged_per_rhs"] = converged_per_rhs
    return payload


def _summarize_solver_infos(load_solver_infos, rtol, atol, maxiter):
    norms = []
    relative_norms = []
    iterations = []
    convergence_flags = []
    load_count = 0
    for item in load_solver_infos:
        load_count += len(item.get("load_ids", []))
        iterations.append(int(item.get("cg_iterations", 0)))
        if "final_norm_res_per_rhs" in item:
            norms.extend(float(value) for value in item["final_norm_res_per_rhs"])
        else:
            norms.append(float(item.get("final_norm_res", np.nan)))
        if "final_norm_res_rel_per_rhs" in item:
            relative_norms.extend(float(value) for value in item["final_norm_res_rel_per_rhs"])
        else:
            relative_norms.append(float(item.get("final_norm_res_rel", np.nan)))
        if "converged_per_rhs" in item:
            convergence_flags.extend(bool(value) for value in item["converged_per_rhs"])
        else:
            convergence_flags.append(bool(item.get("converged", False)))

    finite_norms = [value for value in norms if np.isfinite(value)]
    finite_relative_norms = [value for value in relative_norms if np.isfinite(value)]
    return {
        "batch_count": int(len(load_solver_infos)),
        "load_count": int(load_count),
        "cg_iterations_sum_batches": int(sum(iterations)),
        "cg_iterations_max": int(max(iterations, default=0)),
        "final_norm_res_max": float(max(finite_norms, default=np.nan)),
        "final_norm_res_mean": float(np.mean(finite_norms)) if finite_norms else np.nan,
        "final_norm_res_rel_max": float(max(finite_relative_norms, default=np.nan)),
        "final_norm_res_rel_mean": float(np.mean(finite_relative_norms)) if finite_relative_norms else np.nan,
        "rtol": float(rtol),
        "atol": float(atol),
        "tol": float(rtol),
        "maxiter": int(maxiter),
        "all_converged": bool(convergence_flags and all(convergence_flags)),
        "converged_load_count": int(sum(convergence_flags)),
        "hit_maxiter": bool(any(item.get("hit_maxiter", False) for item in load_solver_infos)),
    }


def _aggregate_load_timings(load_timings, primal_perf_counter_s):
    aggregate = {
        "load_count": int(len(load_timings)),
        "load_wall_s_sum": float(sum(item["load_wall_s"] for item in load_timings)),
        "load_wall_s_mean": float(np.mean([item["load_wall_s"] for item in load_timings])) if load_timings else 0.0,
        "rhs_build_s_sum": float(sum(item["rhs_build_s"] for item in load_timings)),
        "linear_solver_perf_counter_s_sum": float(
            sum((item["linear_solver_perf_counter_s"] or 0.0) for item in load_timings)
        ),
        "cg_iterations_sum": int(sum(item["cg_iterations"] for item in load_timings)),
        "cg_iterations_max": int(max((item["cg_iterations"] for item in load_timings), default=0)),
        "final_norm_res_max": float(max((item["final_norm_res"] for item in load_timings), default=np.nan)),
        "final_norm_res_mean": float(np.mean([item["final_norm_res"] for item in load_timings])) if load_timings else np.nan,
    }
    for label in PROFILE_COMPONENT_LABELS:
        aggregate[f"{label}_s_sum"] = float(sum(item["components"][label]["seconds"] for item in load_timings))
        aggregate[f"{label}_calls_sum"] = int(sum(item["components"][label]["calls"] for item in load_timings))
    return {
        "primaldual": "primal",
        "primal_dual_perf_counter_s": float(primal_perf_counter_s),
        "loads": load_timings,
        "aggregate": aggregate,
    }


def _print_timing_summary(profile_summary):
    agg = profile_summary["aggregate"]
    print(
        "[PROFILE] primal | loads={} | FN={:.3f}s | hG={:.3f}s | FiN={:.3f}s | A={:.3f}s | "
        "lin_solver={:.3f}s".format(
            agg["load_count"], agg["FN_s_sum"], agg["hG_s_sum"], agg["FiN_s_sum"], agg["A_s_sum"],
            agg["linear_solver_perf_counter_s_sum"],
        )
    )


_OFFLOAD_SAFETY_BYTES = 512*2**20
# Residency decisions per (field bytes, loads): the first solve of a process
# measures with a clean pool; later solves would see the pool's cached blocks
# as used and move solutions out needlessly.
_RESIDENCY = {}
_LOAD_LABELS = {}
# Short of memory, keep solutions as displacement coordinates (rebuilt bit
# for bit: Q.strain(z) + E is the same in-place add as the solve's).
_COMPRESS_SOLUTIONS = True


def _resident_limit(solutions, first, nloads):
    """Decide once, after the first load, how many solutions stay on device.

    The CuPy pool then holds that load's working set (high-water mark) and the
    first solution. Every further resident solution needs one field; one more
    field plus 512 MiB stay free for pool fragmentation and audit transients.
    Returns how many finished solutions may stay resident during a load.
    """
    value = getattr(solutions[first], 'val', None)
    if cp is None or not isinstance(value, cp.ndarray):
        return nloads
    free, _ = cp.cuda.runtime.memGetInfo()
    field = int(value.nbytes)
    return max(0, (int(free) - _OFFLOAD_SAFETY_BYTES)//field - 1)


def _compressed_budget(solution):
    """Compressed solutions that may stay resident during a load.

    Half-size coordinate blocks fragment the pool that the strain-sized
    working set reuses, so two strain fields plus 512 MiB stay free (measured
    after the first solution was compressed; one field failed at 360^3).
    """
    _, z = solution.cufhom_coordinates
    free, _ = cp.cuda.runtime.memGetInfo()
    field = 2*int(z.nbytes)  # a real strain field is about twice the coordinates
    return max(0, (int(free) - _OFFLOAD_SAFETY_BYTES - 2*field)//int(z.nbytes))


def _release_full_solution_coordinates(solution):
    """A retained physical field needs no second coordinate representation.

    Call only after choosing full-field residency. Compressed solutions keep
    their owned coordinates until restoration; direct solver callers keep the
    existing snapshot contract.
    """
    if solution.val is not None and hasattr(solution, 'cufhom_coordinates'):
        del solution.cufhom_coordinates


def _offload_solution(solutions, iL):
    """Exact copy of one device solution into pinned host memory."""
    started = time.perf_counter()
    value = solutions[iL].val
    memory = cp.cuda.alloc_pinned_memory(value.nbytes)
    host = np.frombuffer(memory, value.dtype, value.size).reshape(value.shape)
    value.get(out=host)
    solutions[iL].val = host
    _release_full_solution_coordinates(solutions[iL])
    return time.perf_counter()-started


def _compress_solution(solution, macro):
    """Keep the displacement coordinates of a solved load instead of its strain.

    Only solutions whose strain is exactly Q.strain(z) carry coordinates; they
    need half the memory and rebuild bit for bit (same FFT plan) before use.
    """
    coordinates = getattr(solution, 'cufhom_coordinates', None)
    if coordinates is None or solution.val is None:
        return False
    solution.cufhom_macro = macro
    solution.val = None
    return True


def _offload_coordinates(solution):
    """Exact copy of compressed coordinates into pinned host memory."""
    started = time.perf_counter()
    Q, z = solution.cufhom_coordinates
    memory = cp.cuda.alloc_pinned_memory(z.nbytes)
    host = np.frombuffer(memory, z.dtype, z.size).reshape(z.shape)
    z.get(out=host)
    solution.cufhom_coordinates = (Q, host)
    return time.perf_counter()-started


def _restore_offloaded_solutions(solutions, moved):
    started = time.perf_counter()
    for iL in moved:
        solutions[iL].val = cp.asarray(solutions[iL].val)
    for solution in solutions:
        if getattr(solution, 'val', 0) is None and hasattr(solution, 'cufhom_coordinates'):
            Q, z = solution.cufhom_coordinates
            solution.val = Q.strain(cp.asarray(z))
            solution.add_mean(solution.cufhom_macro)  # the same in-place add as the solve
            del solution.cufhom_coordinates, solution.cufhom_macro
    return time.perf_counter()-started


def _needs_bounded_assembly(pb, A, solutions):
    """Keep the fast resident path when all fields and reconstruction fit.

    Only the six-load energy-only indexed workflow can leave fields on host:
    field/stress/sensitivity consumers and partial loads retain their contract.
    Include both driver-free bytes and reusable pool bytes, so warm calls do
    not misinterpret the pool cache as live storage. This decision is fresh
    on every call, unlike the retained-load selector.
    """
    from cufhom_core.general.cupy_cfield_assembly import supports_streamed_assembly
    if (pb.solve.get('store_solution_fields', True)
            or pb.solve.get('partial_load_output', False)
            or pb.solve.get('affine_sensitivity_cfields') is not None
            or callable(pb.solve.get('stress_field_consumer'))
            or callable(pb.solve.get('solution_field_consumer'))
            or str(pb.solve.get('postprocess_assembly', 'gemm')) != 'fused'
            or not supports_streamed_assembly(A, solutions)):
        return False
    field = int(np.prod(A.N))*6*A.material_table.dtype.itemsize
    if any(isinstance(s.val, np.ndarray) for s in solutions):
        return True
    resident_full = sum(int(s.val.nbytes) for s in solutions if isinstance(s.val, cp.ndarray))
    reconstructing = any(getattr(s, 'val', 0) is None for s in solutions)
    # QR reconstruction needs a coordinate upload, a six-component spectrum
    # and a real field. The final real field is already in the six-field sum.
    reconstruction_scratch = (3*field//2) if reconstructing else 0
    needed = len(solutions)*field-resident_full+reconstruction_scratch+_OFFLOAD_SAFETY_BYTES
    pool = current_memory_pool()
    free, _ = cp.cuda.runtime.memGetInfo()
    reusable = pool.total_bytes()-pool.used_bytes()
    return needed > int(free)+int(reusable)


def _prepare_bounded_assembly(solutions):
    """Reconstruct and retain exact full strains on host one load at a time.

    This is retained-solution streaming, not an out-of-core FFT. A load's
    working set must still fit. Release device solutions first so QR never
    runs while five other full fields occupy the GPU.
    """
    started = time.perf_counter()
    for i, solution in enumerate(solutions):
        if isinstance(solution.val, cp.ndarray):
            _offload_solution(solutions, i)
        elif solution.val is None:
            _, z = solution.cufhom_coordinates
            if isinstance(z, cp.ndarray):
                _offload_coordinates(solution)
    # Drop only unused cached allocations, after outstanding transfers finish.
    cupy_synchronize()
    current_memory_pool().free_all_blocks()
    for i, solution in enumerate(solutions):
        if solution.val is None:
            Q, z = solution.cufhom_coordinates
            solution.val = Q.strain(cp.asarray(z))
            solution.add_mean(solution.cufhom_macro)
            del solution.cufhom_coordinates, solution.cufhom_macro
            _offload_solution(solutions, i)
    cupy_synchronize()
    return dict(strategy='host_fields_chunked_energy',
                retained_host_bytes=sum(int(s.val.nbytes) for s in solutions),
                staging_limit_bytes=2*6*6*solutions[0].val.dtype.itemsize*1_048_576,
                preparation_s=float(time.perf_counter()-started))


_PROJECTION_CONTINUATIONS = 3


def _projected_strain_solve(solve, Afun, B, x0, pb, GN):
    """Strain CG whose returned field is projected onto E_N and accepted by its own residual.

    The strain-CG iterate is compatible only up to the rounding that the
    recurrence accumulates, and that part enters the energy assembly of Ceff
    linearly. One projection removes it but raises the physical residual by
    about the incompatible part times the phase contrast. CG restarted from
    the projected field first evaluates its physical residual, returns the
    field unchanged when it passes and otherwise continues from it, as the
    compact route does after a failed audit.
    """
    if (cp is not None and isinstance(B.val,cp.ndarray) and B.val.dtype in (cp.float32,cp.float64)
            and (host_iterate_required(B.N,B.val.dtype)
                 or (B.val.dtype==cp.float64 and library_capacity_required(B.N,B.val.dtype)))):
        from cufhom_core.general.cupy_strain_capacity import supported
        if supported(Afun,B):
            B.cufhom_resident_cg=True
        else:
            B.cufhom_host_cg=True
    par = dict(pb.solver, check_true_residual=False)
    maxiter = int(par.get('maxiter', 1000))
    X, info = solve(solver=pb.solver['kind'], Afun=Afun, B=B, x0=x0, par=par)
    iterations, log, continuations = info['kit'], info['norm_res_log'], 0
    restarts, replacements = info.get('restarts'), info.get('replacements')
    while info['converged']:
        X = GN(X)
        X, info = solve(solver=pb.solver['kind'], Afun=Afun, B=B, x0=X,
                        par=dict(par, maxiter=max(0, maxiter-iterations)))
        iterations, log = iterations+info['kit'], log+info['norm_res_log']
        restarts = (None if restarts is None or info.get('restarts') is None else
                    restarts+int(info['kit'] > 0)+info['restarts'])
        replacements = (None if replacements is None or info.get('replacements') is None else
                        replacements+info['replacements'])
        if info['kit'] == 0 or continuations == _PROJECTION_CONTINUATIONS:
            break
        continuations += 1
    accepted = info['converged'] and info['kit'] == 0
    info.update(kit=iterations, norm_res_log=log, converged=accepted, true_norm_res_rel=info['norm_res_rel'],
                true_residual_converged=accepted, projection_continuations=continuations,
                restarts=restarts, replacements=replacements)
    return X, info


def load_macro_strain(pb, iL, D, real_dtype):
    """Macroscopic strain of load ``iL``: the unit Mandel load, or row ``iL`` of
    ``macro_strains`` when the caller prescribes arbitrary load directions."""
    directions = pb.solve.get('macro_strains', None)
    if directions is None:
        E = np.zeros(D, dtype=real_dtype)
        E[iL] = 1
        return E
    return np.asarray(directions[iL], dtype=real_dtype).copy()


def solve_load_elasticity(iL, D, Nbar, Afun, pb, GN, A, fft_form='c'):
    factory = pb.solver.get('elasticity_krylov_factory')
    profile_enabled = _profile_enabled(pb)
    timing_stats = _new_component_stats() if profile_enabled else None
    solve = linear_solver
    if factory is not None:
        factory_A = TimedCallable(A, 'A', timing_stats) if profile_enabled else A
        solve = factory(A=factory_A, GN=GN, pb=pb, fallback=linear_solver)
    real_dtype = _solver_real_dtype(pb)
    E = load_macro_strain(pb, iL, D, real_dtype)
    key = (D, tuple(E.tolist()), np.dtype(real_dtype).str)
    if key not in _LOAD_LABELS:  # NumPy array formatting costs ~0.1 ms per load
        _LOAD_LABELS[key] = 'macroscopic load E = ' + str(E)
    print(_LOAD_LABELS[key])
    # At very large FP64 grids, intercept before allocating a macroscopic
    # field/RHS that would exhaust the device. The private energy-only path
    # stages CG state but retains each route's original FFT operator.
    if (cp is not None and isinstance(getattr(A, 'val', None), cp.ndarray)
            and factory is not None and fft_form == 'r'
            and np.dtype(real_dtype) == np.dtype('float64')):
        from cufhom.fft.linear.active_capacity import eligible, ActiveCapacity
        if eligible(pb, A, tuple(Nbar), real_dtype):
            engine = pb.solver.get('_active_capacity_engine')
            if engine is None:
                constructor = getattr(solve, 'capacity_engine', None)
                engine = (constructor(tuple(Nbar)) if constructor is not None
                          else ActiveCapacity(A, tuple(Nbar), pb.Y))
                pb.solver['_active_capacity_engine'] = engine
            X, info = engine.solve(E, pb.solver)
            result = {'info': info}
            if profile_enabled:
                result['timing'] = _build_load_timing_payload(
                    iL, info['time'], 0., info, timing_stats)
            return iL, X, result
    EN = Tensor(name='EN', N=Nbar, shape=(D,), Fourier=False, fft_form=fft_form, dtype=real_dtype)
    # A new tensor is zero: adding E equals set_mean(E) without six reductions.
    EN.add_mean(E)
    init_fields = pb.solve.get('initial_solution_fields', None)
    compatible_start = init_fields is None
    if init_fields is not None:
        x0 = EN.zeros_like(name='x0')
        if hasattr(x0.val, "get") or type(x0.val).__module__.startswith("cupy"):
            val = cp.asarray(init_fields[iL], dtype=real_dtype)
        else:
            val = np.asarray(init_fields[iL], dtype=real_dtype)
        if val.size != x0.val.size:
            raise ValueError(
                'initial_solution_fields[{0}] has {1} entries; expected {2}.'.format(
                    iL, val.size, x0.val.size
                )
            )
        x0.val = val.reshape(x0.val.shape)
        # The coordinate adapter owns the normalized physical projection;
        # GN can carry the unscaled-FFT volume factor in that route.
        if pb.solve.get('project_initial_solution_fields', False) and factory is None:
            x0 = GN(x0)
            compatible_start = True
    else:
        x0 = None
    Afun_local = _instrument_afun_for_profile(Afun, GN, A, timing_stats) if profile_enabled else Afun
    rhs_t0 = time.perf_counter()
    build_rhs = getattr(solve, 'build_rhs', None)
    B = build_rhs(EN) if build_rhs is not None else None
    if B is None:
        B = Afun_local(-EN)
    if profile_enabled:  # timing only: otherwise the host keeps queuing work
        cupy_synchronize(B)
    rhs_build_s = time.perf_counter() - rhs_t0
    # The constant macroscopic field is a full six-component array; release
    # it before the solve.
    del EN
    load_t0 = time.perf_counter()
    if solve is linear_solver and compatible_start:
        # (A warm start outside E_N keeps its incompatible part: no projection.)
        X, info = _projected_strain_solve(solve, Afun_local, B, x0, pb, GN)
    else:
        X, info = solve(solver=pb.solver['kind'], Afun=Afun_local, B=B, x0=x0, par=pb.solver)
    load_wall_s = time.perf_counter() - load_t0
    result = {'info': info}
    if profile_enabled:
        result['timing'] = _build_load_timing_payload(iL, load_wall_s, rhs_build_s, info, timing_stats)
    if getattr(X, 'cufhom_host_energy_only', False):
        # These private energy-only fields are accepted fluctuations in pinned
        # memory. Add E on the staged GPU chunk, avoiding a CPU sweep over a
        # multi-GiB array. Consumers requiring total fields never use this path.
        X.cufhom_deferred_mean = E.copy()
    else:
        X.add_mean(E)
    return iL, X, result


def solve_affine_sensitivity_fields(D, Nbar, Afun, pb, GN, solutions, fft_form='c'):
    """Solve exact discrete affine sensitivities around the current primal anchor.

    The primal corrector satisfies ``G A(gamma) e = 0``.  Differentiating with
    respect to an affine coefficient ``gamma_q`` gives
    ``G A(gamma) de_q = -G dA_q e`` with zero macroscopic mean.
    """
    cfields = pb.solve.get('affine_sensitivity_cfields', None)
    if cfields is None:
        return None, {}

    names = list(pb.solve.get('affine_sensitivity_names', []))
    q_count = int(len(cfields))
    if not names:
        names = ['gamma_{0}'.format(q) for q in range(q_count)]
    if len(names) != q_count:
        raise ValueError("affine_sensitivity_names no coincide con affine_sensitivity_cfields.")

    active_load_ids = pb.solve.get('active_load_ids', None)
    if active_load_ids is None:
        load_ids = list(range(D))
    else:
        load_ids = sorted({int(value) for value in active_load_ids})
    if any(not hasattr(solutions[iL], 'val') for iL in load_ids):
        raise RuntimeError("No hay soluciones primales para calcular sensibilidades afines.")

    real_dtype = _solver_real_dtype(pb)
    batch_size = _resolve_affine_sensitivity_batch_size(
        pb,
        field_shape=tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist()),
        nloads=len(load_ids),
    )
    par = dict(pb.solver)

    field_shape = tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist())
    consumer = pb.solve.get('affine_sensitivity_consumer', None)
    progress = pb.solve.get('affine_sensitivity_progress', None)
    streaming = callable(consumer)
    fields = None if streaming else np.empty(
        (q_count, len(load_ids), D) + field_shape,
        dtype=np.dtype(real_dtype),
    )
    solver_infos = []
    consumed = []
    host_transfer_wall_s = 0.0
    consumer_wall_s = 0.0

    def make_sensitivity_tensor(q, spec):
        if isinstance(spec, dict):
            storage = str(spec.get('storage', spec.get('multype', ''))).strip().lower()
            if storage in {'sym21_indexed', 'indexed_sym21'}:
                index = spec.get('index', None)
                table = spec.get('table', None)
                if index is None or table is None:
                    raise ValueError("Descriptor sym21_indexed sin index/table.")
                index_shape = tuple(int(value) for value in getattr(index, 'shape', ()))
                if index_shape == field_shape:
                    val = index[None, ...]
                elif index_shape == (1,) + field_shape:
                    val = index
                else:
                    raise ValueError(
                        "Indice dA/dgamma incompatible: "
                        "se esperaba (*N) o (1, *N)."
                    )
                tensor = Tensor(
                    name='dA_dgamma_{0}'.format(q),
                    val=val,
                    order=1,
                    N=Nbar,
                    Y=pb.Y,
                    multype='sym21_indexed',
                    Fourier=False,
                    origin=0,
                    fft_form=fft_form,
                )
                tensor.material_table = table
                return tensor
            if 'cfield' in spec:
                spec = spec['cfield']
            else:
                raise ValueError("Descriptor dA/dgamma no soportado: {0}".format(storage))

        cfield_shape = tuple(int(value) for value in np.shape(spec))
        if cfield_shape[:1] == (21,):
            multype = 'sym21'
            order = 1
        elif cfield_shape[:2] == (D, D):
            multype = 21
            order = 2
        else:
            raise ValueError(
                "Campo dA/dgamma incompatible: "
                "se esperaba (21, *N) o (6, 6, *N)."
            )
        return Tensor(
            name='dA_dgamma_{0}'.format(q),
            val=spec,
            order=order,
            N=Nbar,
            Y=pb.Y,
            multype=multype,
            Fourier=False,
            origin=0,
            fft_form=fft_form,
        )

    def emit_or_store(q, storage_index, load_id, value):
        nonlocal host_transfer_wall_s, consumer_wall_s
        transfer_started = time.perf_counter()
        host_value = to_host_array(value).astype(np.dtype(real_dtype), copy=False)
        host_transfer_wall_s += float(time.perf_counter() - transfer_started)
        if streaming:
            consumer_started = time.perf_counter()
            consumer(int(q), str(names[q]), int(load_id), host_value)
            consumer_wall_s += float(time.perf_counter() - consumer_started)
            consumed.append((int(q), int(load_id)))
        else:
            fields[int(q), int(storage_index)] = host_value

    for q, cfield in enumerate(cfields):
        dA = make_sensitivity_tensor(q, cfield)

        for start in range(0, len(load_ids), batch_size):
            chunk_load_ids = load_ids[start:start + batch_size]
            if len(chunk_load_ids) == 1:
                solve_started = time.perf_counter()
                load_id = int(chunk_load_ids[0])
                stress_rhs = dA(solutions[load_id])
                B = -(GN(stress_rhs))
                x0 = B.zeros_like(name='x0_sens')
                X, info = linear_solver(
                    solver=pb.solver['kind'],
                    Afun=Afun,
                    B=B,
                    x0=x0,
                    par=par,
                )
                x_host = to_host_array(X.val).astype(np.dtype(real_dtype), copy=False)
                solver_wall_s = float(time.perf_counter() - solve_started)
                solver_infos.append({
                    'coefficient_index': int(q),
                    'coefficient_name': str(names[q]),
                    'load_ids': [load_id],
                    'info': info,
                })
                del X, x0, B, stress_rhs
                _release_affine_sensitivity_batch_memory(pb)
                emit_or_store(q, start, load_id, x_host)
                del x_host
                if callable(progress):
                    progress(
                        int(q), str(names[q]), [load_id],
                        solver_wall_s,
                    )
                continue

            solve_started = time.perf_counter()
            solution_values = [solutions[iL].val for iL in chunk_load_ids]
            xp = get_array_module(dA.val, *solution_values)
            if xp is np:
                val = np.stack([to_host_array(value) for value in solution_values], axis=1)
            else:
                val = xp.stack(
                    [to_backend_array(value, prefer_backend='cupy') for value in solution_values],
                    axis=1,
                )
            sol_batch = solutions[int(chunk_load_ids[0])].copy(
                name='sol_total_batch',
                val=val,
                order=2,
            )
            stress_rhs = dA(sol_batch)
            B = -(GN(stress_rhs))
            x0 = B.zeros_like(name='x0_sens_batch')
            batch_par = dict(par)
            batch_par['batched_rhs'] = True
            X, info = linear_solver(
                solver=pb.solver['kind'],
                Afun=Afun,
                B=B,
                x0=x0,
                par=batch_par,
            )
            x_host = to_host_array(X.val).astype(np.dtype(real_dtype), copy=False)
            solver_wall_s = float(time.perf_counter() - solve_started)
            solver_infos.append({
                'coefficient_index': int(q),
                'coefficient_name': str(names[q]),
                'load_ids': [int(value) for value in chunk_load_ids],
                'info': info,
            })
            del X, x0, B, stress_rhs, sol_batch, val, solution_values
            _release_affine_sensitivity_batch_memory(pb)
            for local, load_id in enumerate(chunk_load_ids):
                emit_or_store(q, start + local, load_id, x_host[:, local])
            del x_host
            if callable(progress):
                progress(
                    int(q), str(names[q]),
                    [int(value) for value in chunk_load_ids],
                    solver_wall_s,
                )

    rel_residuals = []
    convergence_flags = []
    iterations = []
    for item in solver_infos:
        info = item.get('info', {})
        if 'norm_res_rel_per_rhs' in info:
            rel_residuals.extend(float(value) for value in info['norm_res_rel_per_rhs'])
        elif 'norm_res_rel' in info:
            rel_residuals.append(float(info['norm_res_rel']))
        elif 'norm_res' in info and 'rhs_norm' in info:
            rhs_norm = max(float(info['rhs_norm']), np.finfo(float).tiny)
            rel_residuals.append(float(info['norm_res']) / rhs_norm)
        if 'converged_per_rhs' in info:
            convergence_flags.extend(bool(value) for value in info['converged_per_rhs'])
        elif 'converged' in info:
            convergence_flags.append(bool(info['converged']))
        if 'kit' in info:
            iterations.append(int(info['kit']))

    summary = {
        'coefficient_names': names,
        'load_ids': [int(value) for value in load_ids],
        'batch_size': int(batch_size),
        'solve_count': int(len(solver_infos)),
        'rhs_count': int(q_count * len(load_ids)),
        'all_converged': bool(all(convergence_flags)) if convergence_flags else False,
        'final_norm_res_rel_max': float(max(rel_residuals)) if rel_residuals else np.nan,
        'cg_iterations_max': int(max(iterations)) if iterations else -1,
        'streamed': bool(streaming),
        'host_transfer_wall_s': float(host_transfer_wall_s),
        'consumer_wall_s': float(consumer_wall_s),
    }
    output = {
        'coefficient_names': names,
        'load_ids': [int(value) for value in load_ids],
        'field_shape': [int(D)] + [int(value) for value in field_shape],
        'streamed': bool(streaming),
    }
    if streaming:
        output['consumed'] = consumed
    else:
        output['fields'] = fields
    return output, summary


def stream_primal_solution_fields_before_sensitivity(D, Nbar, pb, solutions, load_ids):
    consumer = pb.solve.get('solution_field_consumer', None)
    if not callable(consumer):
        return
    field_dtype = np.dtype(pb.solve.get('solution_field_dtype', _solver_real_dtype(pb)))
    field_shape = tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist())
    consumed = []
    for load_id in load_ids:
        load_id = int(load_id)
        if not hasattr(solutions[load_id], 'val'):
            raise RuntimeError("No hay solucion primal para emitir campos en streaming.")
        total = to_host_array(solutions[load_id].val).astype(field_dtype, copy=False)
        fluctuation = total.copy()
        E = load_macro_strain(pb, load_id, D, field_dtype)
        fluctuation -= E.reshape((-1,) + (1,) * (fluctuation.ndim - 1))
        consumer(load_id, fluctuation)
        consumed.append(load_id)
    pb.output['solution_fields_primal'] = {
        'load_ids': consumed,
        'field_shape': [int(D)] + [int(value) for value in field_shape],
        'dtype': str(field_dtype),
        'streamed': True,
    }


def stream_primal_equilibrated_stresses(D, Nbar, pb, solutions, A, GN, load_ids):
    """Emit statically admissible stresses without retaining the FOM fields.

    ``GN`` is the same zero-mean compatible-strain projector used by the
    primal Galerkin solve.  Thus ``(I-GN)sigma`` preserves the average stress
    and is orthogonal to every compatible fluctuation in this discrete
    formulation. A user-supplied consumer receives one stress field per load
    without retaining all fields in memory.
    """
    consumer = pb.solve.get('stress_field_consumer', None)
    if not callable(consumer):
        return
    field_dtype = np.dtype(pb.solve.get('stress_field_dtype', _solver_real_dtype(pb)))
    if field_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("stress_field_dtype debe ser float32 o float64.")
    equilibrate = bool(pb.solve.get('stress_field_equilibrate', True))
    field_shape = tuple(int(v) for v in np.asarray(Nbar, dtype=int).tolist())
    consumed, defects = [], []
    for load_id in load_ids:
        load_id = int(load_id)
        if not hasattr(solutions[load_id], 'val'):
            raise RuntimeError("No hay solucion primal para emitir tensiones.")
        stress = A(solutions[load_id])
        compatible = GN(stress)
        stress_val = stress.val
        compatible_val = compatible.val
        xp = get_array_module(stress_val, compatible_val)
        stress_norm = xp.linalg.norm(stress_val.reshape(-1))
        compatible_norm = xp.linalg.norm(compatible_val.reshape(-1))
        # Scalars must cross to the host for the manifest even on CUDA.
        stress_norm_host = float(to_host_array(stress_norm))
        compatible_norm_host = float(to_host_array(compatible_norm))
        defects.append(
            compatible_norm_host / max(stress_norm_host, np.finfo(float).tiny)
        )
        if equilibrate:
            field = stress - compatible
            del stress
        else:
            field = stress
        if bool(pb.solve.get('stress_field_device', False)) and hasattr(field.val, 'get'):
            # Device consumer: the equilibrated stress stays on the GPU.
            import cupy as cp
            value = cp.array(field.val, dtype=field_dtype)
        else:
            value = to_host_array(field.val).astype(field_dtype, copy=False)
        consumer(load_id, value)
        consumed.append(load_id)
        del field, compatible
    pb.output['stress_fields_primal'] = {
        'load_ids': consumed,
        'field_shape': [int(D)] + [int(value) for value in field_shape],
        'dtype': str(field_dtype),
        'field': 'total Mandel stress; zero-mean compatible projection removed',
        'equilibrated': equilibrate,
        'compatible_projection_relative_norms': defects,
        'streamed': True,
    }


def _solve_loads(D, Nbar, Afun, pb, GN, A, fft_form, load_ids, solutions, results,
                 load_solver_infos, load_timings, app_timing):
    """Solve the loads in order. When the device cannot hold every solution,
    keep the most recent ones and move older ones to coordinates or the host
    (exact copies), restoring them before postprocessing."""
    offloaded, offload_s, resident_limit = [], 0., len(load_ids)
    moved_full, compressed, budget, decision = [], [], 0, None
    key = None
    for position, iL in enumerate(load_ids):
        iL, sol, res = solve_load_elasticity(iL, D, Nbar, Afun, pb, GN, A, fft_form)
        solutions[iL] = sol
        results[iL] = res
        load_solver_infos.append(_solver_info_payload(res.get('info', {}), [iL]))
        if _profile_enabled(pb) and 'timing' in res:
            load_timings.append(res['timing'])
        last = position+1 == len(load_ids)
        if position == 0 and not last:
            key = (int(getattr(getattr(sol, 'val', None), 'nbytes', 0)), len(load_ids))
            decision = _RESIDENCY.get(key)
            if decision is None:
                resident_limit = _resident_limit(solutions, iL, len(load_ids))
                if (resident_limit < len(load_ids)-1 and _COMPRESS_SOLUTIONS
                        and hasattr(sol, 'cufhom_coordinates')):
                    resident_limit = -1  # compressed mode
            else:
                resident_limit, budget = decision
        if last:
            _release_full_solution_coordinates(sol)
            continue
        if resident_limit < 0:
            # Memory is short: keep coordinates (half a strain field) and
            # move the oldest beyond the budget to host. A solution without
            # coordinates moves to host whole.
            macro = load_macro_strain(pb, iL, D, _solver_real_dtype(pb))
            if _compress_solution(sol, macro):
                compressed.append(int(iL))
                if position == 0 and decision is None:
                    budget = _compressed_budget(sol)
            else:
                offload_s += _offload_solution(solutions, iL)
                moved_full.append(int(iL))
                offloaded.append(int(iL))
            resident = [j for j in compressed if not isinstance(
                solutions[j].cufhom_coordinates[1], np.ndarray)]
            for j in resident[:max(0, len(resident)-budget)]:
                offload_s += _offload_coordinates(solutions[j])
                offloaded.append(int(j))
        else:
            # Keep the most recent solutions; move the oldest out.
            _release_full_solution_coordinates(sol)
            resident = [j for j in load_ids[:position+1] if j not in offloaded]
            for j in resident[:max(0, len(resident)-resident_limit)]:
                offload_s += _offload_solution(solutions, j)
                moved_full.append(int(j))
                offloaded.append(int(j))
    if len(load_ids) > 1 and _RESIDENCY.get(key) is None:
        _RESIDENCY[key] = (resident_limit, budget)
    if _needs_bounded_assembly(pb, A, solutions):
        app_timing['solution_assembly'] = _prepare_bounded_assembly(solutions)
        restore_s = app_timing['solution_assembly']['preparation_s']
    else:
        restore_s = _restore_offloaded_solutions(solutions, moved_full)
        app_timing['solution_assembly'] = {'strategy': 'resident_fields'}
    app_timing['offloaded_solution_loads'] = sorted(offloaded)
    app_timing['offload_s'] = float(offload_s)
    app_timing['offload_restore_s'] = float(restore_s)


def elasticity(problem):
    """Homogenization of linear elasticity (GaNi scheme, primal formulation)."""
    print(' ')
    pb = problem
    print(pb)

    # Fourier projection: closed form on CUDA ('direct'), stored otherwise.
    projection_t0 = _stage_tic()
    fft_form = pb.solve.get('fft_form', 'c')
    real_dtype = _solver_real_dtype(pb)
    N = pb.solve['N']
    projection_storage = str(pb.solve.get("projection_storage", "full"))
    if projection_storage == 'direct':
        hG1N = proj.elasticity_direct(N, pb.Y, fft_form=fft_form, dtype=real_dtype)
    else:
        hG1N = proj.elasticity(N, pb.Y, fft_form=fft_form, dtype=real_dtype)
    FN = DFT(name='FN', inverse=False, N=N, fft_form=fft_form)
    FiN = DFT(name='FiN', inverse=True, N=N, fft_form=fft_form)
    GN = Operator(name='G1', mat=[[FiN, hG1N, FN]])
    projection_s = _stage_toc(projection_t0)
    app_timing = pb.output.setdefault('application_timing', {})
    app_timing['projection_s'] = float(projection_s)
    app_timing['projection_storage'] = projection_storage
    app_timing['fft_form'] = str(fft_form)

    primal_t0 = _stage_tic()
    tim = Timer(name='primal')
    print('\nproblem: primal')
    solutions = np.zeros(pb.shape).tolist()
    results = np.zeros(pb.shape).tolist()
    load_timings = []
    load_solver_infos = []

    material_t0 = _stage_tic()
    mat = Material(pb.material)
    A = mat.get_A_GaNi(N, fft_form=fft_form)
    Afun = Operator(name='FiGFA', mat=[[GN, A]])
    material_operator_s = _stage_toc(material_t0)

    solve_t0 = _stage_tic()
    D = int(pb.dim*(pb.dim+1)/2)
    active_load_ids = pb.solve.get('active_load_ids', None)
    if active_load_ids is None:
        load_ids = list(range(D))
    else:
        load_ids = sorted({int(value) for value in active_load_ids})
        if not load_ids or any(value < 0 or value >= D for value in load_ids):
            raise ValueError("active_load_ids contiene cargas inválidas.")
    _solve_loads(D, N, Afun, pb, GN, A, fft_form, load_ids, solutions, results,
                 load_solver_infos, load_timings, app_timing)
    tim.measure()
    solve_s = _stage_toc(solve_t0)
    if _profile_enabled(pb):
        profile_summary = _aggregate_load_timings(load_timings, tim.vals[-1][1])
        pb.output.setdefault('solver_timing', {})['primal'] = profile_summary
        _print_timing_summary(profile_summary)

    sensitivities = pb.solve.get('affine_sensitivity_cfields', None) is not None
    if sensitivities and bool(pb.solve.get('solution_field_stream_before_sensitivity', False)):
        stream_primal_solution_fields_before_sensitivity(D, N, pb, solutions, load_ids)
    if callable(pb.solve.get('stress_field_consumer', None)):
        stream_primal_equilibrated_stresses(D, N, pb, solutions, A, GN, load_ids)

    affine_sensitivity_summary = {}
    if sensitivities:
        sens_t0 = _stage_tic()
        sensitivity_output, affine_sensitivity_summary = solve_affine_sensitivity_fields(
            D, N, Afun, pb, GN, solutions, fft_form
        )
        affine_sensitivity_summary['wall_s'] = float(_stage_toc(sens_t0))
        affine_sensitivity_summary['solver_wall_excluding_consumer_s'] = float(
            max(
                0.0,
                affine_sensitivity_summary['wall_s']
                - float(affine_sensitivity_summary.get('consumer_wall_s', 0.0)),
            )
        )
        if sensitivity_output is not None:
            pb.output['affine_sensitivity_primal'] = sensitivity_output
            pb.output['affine_sensitivity_summary_primal'] = affine_sensitivity_summary

    if bool(pb.solve.get('partial_load_output', False)):
        partial_columns = {}
        for iL in load_ids:
            stress = A(solutions[iL])
            partial_columns[int(iL)] = np.asarray(stress.mean(), dtype=float)
        pb.output['partial_columns_primal'] = partial_columns
        pb.output['res_primal'] = results
        if pb.solve.get('store_solution_fields', False):
            pb.output['sol_primal'] = solutions
        postprocess_s = 0.0
    else:
        postprocess_t0 = _stage_tic()
        postprocess(pb, A, solutions, results)
        postprocess_s = _stage_toc(postprocess_t0)
    del Afun, GN
    app_timing.setdefault('primaldual', {})['primal'] = {
        'material_operator_s': float(material_operator_s),
        'load_solve_s': float(solve_s),
        'postprocess_s': float(postprocess_s),
        'total_primaldual_s': float(_stage_toc(primal_t0)),
        'load_solver_infos': load_solver_infos,
        'load_solver_summary': _summarize_solver_infos(
            load_solver_infos,
            rtol=pb.solver.get('rtol', pb.solver.get('tol', np.nan)),
            atol=pb.solver.get('atol', 0.0),
            maxiter=pb.solver.get('maxiter', 0),
        ),
        'affine_sensitivity_summary': affine_sensitivity_summary,
    }
