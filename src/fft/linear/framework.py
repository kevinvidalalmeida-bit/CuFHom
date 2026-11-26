"""Unified entry point with discretization-specific GPU execution plans.

The facade shares material preparation, options, audit/result schemas and final
energy validation. GaNi keeps its fused FFT/CG pipeline; HEX8 keeps nodal PCG
and its own kernels. No virtual dispatch is inserted into either Krylov loop.
"""

import contextlib
import io
import threading

import numpy as np

from .common import (
    MANDEL_ORDER,
    IndexedMaterial,
    SolveOptions,
    audit_record,
    cell_lengths,
    checked_energy_matrix,
    elastic_matrix,
    energy_report,
    energy_tolerance,
    isotropic_reference,
    reference_bounds,
    working_material_check,
)

_ENTRY_LOCK = threading.RLock()  # legacy GaNi owns process-wide FFT configuration


def homogenize(
    material,
    stiffness=None,
    *,
    discretization="gani",
    lengths=(1.0, 1.0, 1.0),
    precision=None,
    backend="auto",
    dtype=None,
    reference=None,
    rtol=None,
    atol=0.0,
    maxiter=3000,
    storage="auto",
    execution="optimized",
    working=None,
    energy_rtol=None,
    evaluation="df64",
    verify_tensor=False,
    verification_rtol=None,
):
    """Solve six unit Mandel loads on the same indexed coefficient image.

    Select precision='fp32', 'fp64' or 'mixed'; dtype and working need not be
    set together with it. FP32 rounds material storage and Krylov vectors to
    FP32. FP64 uses FP64 throughout. Mixed keeps the material and accumulated
    solution in FP64 and uses FP32 Krylov vectors, with fresh df64 evaluations.
    Mixed requires CUDA. backend='auto' selects CUDA when a device is available,
    otherwise NumPy. It never changes backend or precision after a failed solve.
    With no precision controls, raw inputs default to mixed on CUDA and FP64 on
    CPU. rtol defaults to 1e-5 for an explicit FP32 selection, otherwise 1e-6.

    material is an IndexedMaterial, or the integer index with stiffness supplied
    separately. dtype defaults to descriptor precision, otherwise FP64. Fields
    and tolerances retain their native meaning: GaNi projected stress RMS versus
    HEX8 volume-normalized nodal Linf. Equal numeric tolerances across those two
    norms do NOT certify equal continuum error.

    execution='reference' selects a staged HEX8 action with the same PCG and
    reference; it is an internal controlled ablation, separate from FANS software.
    GaNi currently accepts optimized/auto execution and its automatic residency:
    the first level that fits the device among all-device, the mixed FP64
    solution in pinned host memory, and the solution with the Krylov iterate
    there (the rows that fit stay on the device; coordinates.RESIDENCIES).
    No FFT backend, material law or solver is patched to dispatch a call.

    working is the Krylov precision. CUDA float64 problems default to
    'float32': reliable-update CG with FP32 kernels, an FP64 solution, FP64
    true-residual replacements and the unchanged fresh FP64 acceptance of each
    discretization; working='float64' runs every kernel in FP64. Both
    discretizations share these three configurations (mixed, FP32, FP64) on
    CUDA, with the same fresh acceptance and identity assembly.
    evaluation is the arithmetic of the fresh residual evaluations of the
    mixed and FP32 configurations after the load: 'df64' (default) pairs of
    FP32 at the FP32 rate (about 2^-46 relative per tested operation;
    accumulated residual error also depends on conditioning) or 'float64'
    (IEEE FP64). GaNi runs them in its fused pipeline; grids outside it, and
    the load itself, use cuFFT FP64. HEX8 evaluates its fresh nodal residual
    (stencil and interface gather) in it. working='float64' keeps every
    evaluation in FP64.
    energy_rtol replaces the residual criterion by a fresh energy-error bound
    on each load's effective energy, in every CUDA working precision of both
    discretizations, changing the estimated quantity used for acceptance.
    energy_error_bound estimates algebraic field-energy error; it does not
    include floating-point assembly, evaluation or retained-field rounding.
    verify_tensor=True retains the six fields and re-evaluates their full
    energy matrix and original-operator residuals on the CPU in FP64. It
    returns that matrix as C and preserves C_production. verification_rtol
    optionally requests a numerical diagonal energy target. Neither option
    supplies a rounding certificate. Prepared FP32 descriptors have already
    lost the coefficients preceding their rounding; this is reported.
    Numerical failures terminate this call. Working precision never escalates
    after a failed load or a failed residency check; another precision
    requires an explicit new call.
    """
    if precision is not None and precision not in ("fp32", "fp64", "mixed"):
        raise ValueError("precision must be fp32, fp64 or mixed")
    if backend == "auto":
        from .solver import _default_fft_backend

        backend = "cupy" if _default_fft_backend() == "cupy" else "numpy"
    if precision is not None:
        material_dtype = "float32" if precision == "fp32" else "float64"
        krylov_dtype = "float64" if precision == "fp64" else "float32"
        if precision == "mixed" and backend != "cupy":
            raise ValueError("precision='mixed' requires CUDA; use fp64 or fp32 on CPU")
        if (dtype is not None and np.dtype(dtype) != np.dtype(material_dtype)) or (
            working is not None and np.dtype(working) != np.dtype(krylov_dtype)
        ):
            raise ValueError("precision conflicts with dtype or working")
        dtype, working = material_dtype, krylov_dtype
    if rtol is None:
        rtol = 1e-5 if precision == "fp32" else 1e-6
    kind = str(discretization).lower()
    if kind not in ("gani", "hex8"):
        raise ValueError("discretization must be gani or hex8")
    if evaluation not in ("df64", "float64"):
        raise ValueError("evaluation must be df64 or float64")
    energy_rtol = energy_tolerance(energy_rtol)
    verification_rtol = energy_tolerance(verification_rtol)
    if not isinstance(verify_tensor, (bool, np.bool_)) or (
        verification_rtol is not None and not verify_tensor
    ):
        raise ValueError("verification_rtol requires verify_tensor=True (boolean)")
    if execution not in ("optimized", "auto", "reference"):
        raise ValueError("execution must be optimized, auto or reference")
    if isinstance(material, IndexedMaterial):
        if stiffness is not None or (
            dtype is not None and np.dtype(dtype) != material.dtype
        ):
            raise ValueError(
                "Prepared material already defines its stiffness and precision"
            )
        data = material
    else:
        if stiffness is None:
            raise ValueError("stiffness is required with an index array")
        data = IndexedMaterial(
            material, stiffness, dtype="float64" if dtype is None else dtype
        )
    verification_material = None
    if verify_tensor:
        verification_material = (
            IndexedMaterial(data.index, data.stiffness, dtype="float64")
            if isinstance(material, IndexedMaterial)
            else IndexedMaterial(material, stiffness, dtype="float64")
        )
    fields = [] if verify_tensor else None
    options = SolveOptions(backend, rtol, atol, maxiter, storage)
    lengths = cell_lengths(lengths)
    if kind == "gani" and (execution == "reference" or storage != "auto"):
        raise ValueError("GaNi currently uses its automatic execution/residency plan")
    if working is None:
        working = (
            "float32"
            if (backend, data.dtype) == ("cupy", np.dtype("float64"))
            else data.dtype.name
        )
    working = np.dtype(working)
    if working != data.dtype and (working, data.dtype, backend) != (
        np.dtype("float32"),
        np.dtype("float64"),
        "cupy",
    ):
        raise ValueError(
            "A distinct working precision requires CUDA, float64 and float32 Krylov kernels"
        )
    if energy_rtol is not None and backend != "cupy":
        raise ValueError("The energy criterion is implemented by the CUDA PCG")
    if reference is not None:
        reference = elastic_matrix(reference, name="reference")
    working_screen = (
        working_material_check(data.stiffness, working) if kind == "gani" else None
    )
    # CUDA GaNi runs one coordinate route in every configuration (as HEX8):
    # mixed, FP32 or FP64; the core strain route remains the CPU reference.
    configuration = "mixed" if working != data.dtype else working.name
    with _ENTRY_LOCK:
        residency = (
            _gani_residency(
                data,
                configuration,
                energy=energy_rtol is not None,
                library=fields is not None,
            )
            if kind == "gani" and backend == "cupy"
            else None
        )
        if kind == "gani" and backend == "cupy" and residency is None:
            raise MemoryError(
                f"The requested {configuration} GaNi configuration does not fit the available "
                "device/host residency budget; choose a different configuration or grid explicitly"
            )
        if kind == "gani" and backend == "cupy":
            result = _gani_mixed(
                data,
                lengths,
                options,
                reference,
                energy_rtol,
                residency,
                evaluation,
                fields,
                configuration=configuration,
            )
        elif kind == "gani":
            result = _gani(
                data, lengths, options, reference, working, energy_rtol, fields
            )
        else:
            result = _hex8(
                data,
                lengths,
                options,
                reference,
                execution,
                working,
                energy_rtol,
                evaluation,
                fields,
            )
    result["C"], defect = checked_energy_matrix(
        result["C"], symmetry_rtol=max(10 * rtol, 100 * np.finfo(data.dtype).eps)
    )
    result.update(
        discretization=kind,
        dtype=data.dtype.name,
        precision="mixed"
        if working != data.dtype
        else "fp32"
        if data.dtype == np.float32
        else "fp64",
        backend=backend,
        shape=data.shape,
        lengths=lengths,
        mandel_order=MANDEL_ORDER,
        symmetry_defect=defect,
        iterations=[a["iterations"] for a in result["audits"]],
        continuum_error_certified=False,
    )
    bounds = result["spectral_bounds"] = reference_bounds(
        data.stiffness, result["reference"]
    )
    fallback = None
    if kind == "gani":
        _, bulk, shear = isotropic_reference(data.stiffness, result["reference"])
        constant = max(1 / (2 * shear), 1 / (bulk + 4 * shear / 3))
        fallback = [
            constant
            * (
                a["native"]["final_norm_res"]
                / (int(np.prod(data.shape)) if backend == "cupy" else 1)
            )
            ** 2
            / bounds["gamma_minus"]
            for a in result["audits"]
        ]
    result["energy_error_bound"] = energy_report(
        result["C"],
        bounds,
        [a["native"].get("energy_G0") for a in result["audits"]],
        fallback_upper=fallback,
    )
    cpu_gani = kind == "gani" and backend != "cupy"
    result["working"] = working.name
    result["precision_contract"] = dict(
        material=data.dtype.name,
        fields=data.dtype.name,
        krylov=working.name,
        reductions="float64",
        residual_operator=(
            "production_precision"
            if cpu_gani
            else "float64_material"
            if data.dtype == np.float64
            else "float64_rounded_material"
        ),
        evaluation=(
            [data.dtype.name]
            if cpu_gani
            else sorted({i.get("evaluation", "float64") for i in result["infos"]})
        ),
        algebraic_energy_bound_certified=False,
    )
    result["precision_contract"].update(
        working_operator="rounded_integrated_element_tables"
        if kind == "hex8"
        else "rounded_constitutive_table",
        residual_storage="float64"
        if kind == "hex8" and backend == "numpy"
        else working.name,
        returned_tensor_rounding_certified=False,
        control=("device_lag1_polling" if backend == "cupy" else "host")
        if kind == "hex8"
        else "route_specific"
        if cpu_gani
        else (
            "device_prefix4_host_smoothing"
            if any(
                i.get("linear_cg_control") == "device_prefix4" for i in result["infos"]
            )
            else "host_scalar_per_iteration"
        ),
    )
    result["working_operator_diagnostics"] = (
        working_screen
        if kind == "gani"
        else result["native_report"]["working_diagnostics"]
    )
    for audit, info in zip(result["audits"], result["infos"]):
        audit.update(
            status="accepted",
            scope="production_solver",
            termination_reason="fresh_" + audit["metric"],
            problem_precision=data.dtype.name,
            working_precision=working.name,
            evaluation=info.get(
                "evaluation", result["precision_contract"]["evaluation"][0]
            ),
            recursive_metric=(
                "strain_residual_RMS" if backend == "numpy" else "coordinate_l2"
            )
            if kind == "gani"
            else "nodal_force_Linf_and_preconditioned_l2",
            recursive_norm=info.get("recursive_norm_res")
            if kind == "gani"
            else info.get("recursive_residual_inf"),
            replacements=info.get("replacements"),
            restarts=info.get("restarts"),
            counter_availability="recorded"
            if info.get("restarts") is not None and info.get("replacements") is not None
            else "not_recorded_by_legacy_route",
            recursive_residual_is_acceptance=False,
        )
        audit["recursive_norm_available"] = audit["recursive_norm"] is not None
    result["operator_contract"] = dict(
        boundary="periodic",
        energy="cell_average",
        admissible_modes="zero_mean_nyquist_zero_wavenumber"
        if kind == "gani"
        else "all_nodal_modes_except_translations",
        reference_policy="explicit"
        if reference is not None
        else (
            "isotropic_projection_of_phase_mean"
            if kind == "gani"
            else "phase_arithmetic_mean"
        ),
    )
    if verify_tensor:
        from .verification import verify_effective_tensor

        source = (
            "descriptor_coefficients_after_rounding"
            if isinstance(material, IndexedMaterial) and data.dtype == np.float32
            else "descriptor_float64"
            if isinstance(material, IndexedMaterial)
            else "original_input_float64"
        )
        report = verify_effective_tensor(
            verification_material,
            fields,
            kind=kind,
            lengths=lengths,
            reference=result["reference"],
            production=result["C"],
            rtol=rtol,
            atol=atol,
            tensor_rtol=verification_rtol
            if verification_rtol is not None
            else energy_rtol,
            energy_acceptance=energy_rtol is not None,
            coefficient_source=source,
        )
        result["C_production"] = result["C"]
        result["production_energy_error_bound"] = result["energy_error_bound"]
        result["production_spectral_bounds"] = result["spectral_bounds"]
        result["C"] = report["C"]
        if "C_stress" in result:
            result["C_stress_production"] = result["C_stress"]
        result["C_stress"] = report["C_stress"]
        result["tensor_verification"] = report
        result["energy_error_bound"] = report["energy_error_bound"]
        result["spectral_bounds"] = report["spectral_bounds"]
        result["operator_contract"]["returned_tensor_coefficients"] = source
        report["material_rounding_relative"] = float(
            np.linalg.norm(
                verification_material.stiffness - data.stiffness.astype(float)
            )
            / np.linalg.norm(verification_material.stiffness)
        )
        result["precision_contract"]["returned_tensor_evaluation"] = "float64"
    return result


def _gani(material, lengths, options, reference, working, energy_rtol, fields=None):
    from cufhom_core.tensors import fft

    from .solver import solve_homogenization

    c0, _, _ = isotropic_reference(material.stiffness, reference)
    # The core equations/audits use N*RMS on CUDA and RMS on CPU. Convert the
    # public absolute tolerance at entry, and normalize only the output metadata.
    scale = int(np.prod(material.shape)) if options.backend == "cupy" else 1
    parameters = dict(
        indexed_material=material,
        cell_lengths=lengths,
        reference_stiffness=c0 if reference is None else reference,
        fft_backend="cupy" if options.backend == "cupy" else "scipy",
        solver_real_dtype=material.dtype.name,
        solver_fp64_coordinates=True,
        solver_rtol=options.rtol,
        solver_atol=options.atol * scale,
        solver_maxiter=options.maxiter,
        require_convergence=True,
        check_true_residual=True,
        _return_solver_report=True,
    )
    if working != material.dtype:
        parameters["solver_working_dtype"] = working.name
    if energy_rtol is not None:
        parameters["solver_energy_rtol"] = float(energy_rtol)
    if fields is not None:
        parameters.update(
            solution_field_consumer=lambda a, field: fields.append(
                np.array(field, dtype=float, copy=True)
            ),
            solution_field_dtype="float64",
        )
    previous = fft.FFT_BACKEND
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            c = solve_homogenization(parameters)
    finally:
        fft.set_fft_backend(previous)
    report = parameters["_solver_report"]
    infos = report["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    if len(infos) != 6:
        raise RuntimeError("GaNi did not return six load audits")
    audits = _gani_audits(infos, scale, _gamma_minus(material, c0, reference))
    metadata = report["cufhom_core_application_timing"]
    fused = all(i.get("linear_fused_fft", False) for i in infos)
    for i in infos:  # native G0 uses the CUDA N*RMS scale squared on the coordinates
        if i.get("energy_G0") is not None:
            i["energy_G0"] = float(i["energy_G0"])
    return dict(
        C=c,
        audits=audits,
        infos=infos,
        reference=c0,
        implementation="CuFHom-GaNi-fused" if fused else "CuFHom-GaNi-library",
        execution="generated_fft" if fused else "library_fft",
        memory=dict(
            policy="auto",
            working_state="operator_specific",
            offloaded_loads=metadata.get("offloaded_solution_loads", []),
            assembly=metadata.get("solution_assembly", {}),
        ),
        native_report=report,
    )


def _gani_residency(material, configuration="mixed", energy=False, library=False):
    from .coordinates import mixed_residency

    return mixed_residency(
        material.shape,
        configuration,
        energy,
        phases=len(material.stiffness),
        library=library,
    )


def _gani_mixed(
    material,
    lengths,
    options,
    reference,
    energy_rtol,
    residency="device",
    evaluation="df64",
    fields=None,
    configuration="mixed",
):
    """CUDA GaNi with identity assembly in any configuration (coordinates.homogenize_gani_mixed)."""
    from cufhom_core.applications import _solver_info_payload

    from .coordinates import homogenize_gani_mixed

    c0, _, _ = isotropic_reference(material.stiffness, reference)
    native = homogenize_gani_mixed(
        material,
        lengths,
        c0 if reference is None else reference,
        rtol=options.rtol,
        atol=options.atol * int(np.prod(material.shape)),
        maxiter=options.maxiter,
        energy=energy_rtol,
        residency=residency,
        evaluation=evaluation,
        field_consumer=None if fields is None else fields.append,
        configuration=configuration,
    )
    # Same per-load schema as the core GaNi report of the FP64 route.
    infos = [_solver_info_payload(info, [a]) for a, info in enumerate(native["infos"])]
    audits = _gani_audits(
        infos, int(np.prod(material.shape)), _gamma_minus(material, c0, reference)
    )
    fused = all(i.get("linear_fused_fft", False) for i in infos)
    return dict(
        C=native["C"],
        C_stress=native["C_stress"],
        audits=audits,
        infos=infos,
        reference=c0,
        implementation="CuFHom-GaNi-fused" if fused else "CuFHom-GaNi-library",
        execution="generated_fft" if fused else "library_fft",
        memory=dict(
            policy="auto",
            working_state="operator_specific",
            retained=native["retained"],
            solution="device" if residency == "device" else "host",
            iterate="host" if residency == "iterate" else "device",
            residency=residency,
            assembly="identity_fp32_coordinates",
        ),
        native_report=native,
    )


def _gani_audits(infos, scale, gamma_minus):
    """Audits of GaNi load infos (core report schema; residuals in N*RMS on CUDA)."""
    return [
        audit_record(
            metric="energy_error_upper",
            residual=i["energy_G0"] / gamma_minus,
            rhs_norm=i["energy"],
            threshold=i["energy_threshold"],
            iterations=i["cg_iterations"],
            accepted=i["converged"],
            native=i,
        )
        if i.get("criterion") == "energy"
        else audit_record(
            metric="projected_stress_RMS",
            residual=i["final_norm_res"] / scale,
            rhs_norm=i["rhs_norm"] / scale,
            threshold=i["threshold"] / scale,
            iterations=i["cg_iterations"],
            accepted=i.get("true_residual_converged", i["converged"]),
            native=i,
        )
        for i in infos
    ]


def _gamma_minus(material, c0, reference):
    return reference_bounds(material.stiffness, c0 if reference is None else reference)[
        "gamma_minus"
    ]


def _hex8(
    material,
    lengths,
    options,
    reference,
    execution,
    working,
    energy_rtol,
    evaluation="df64",
    fields=None,
):
    from .hex8_general import Hex8Operator, homogenize_hex8

    if execution == "reference":
        from .hex8_reference import Hex8ReferenceOperator

        operator_type = Hex8ReferenceOperator
    else:
        operator_type = Hex8Operator
    native = homogenize_hex8(
        material.index,
        material.stiffness,
        lengths=lengths,
        reference=reference,
        backend=options.backend,
        dtype=material.dtype.name,
        rtol=options.rtol,
        atol=options.atol,
        maxiter=options.maxiter,
        storage=options.storage,
        operator_type=operator_type,
        working=working.name,
        energy_rtol=energy_rtol,
        return_fields=fields is not None,
        evaluation="float64" if working == np.float64 else evaluation,
    )
    if fields is not None:
        fields.extend(native.pop("solutions"))
    # The energy criterion accepts on the fresh energy-error bound of each load
    # (relative to its energy); the residual criterion on the nodal Linf norm.
    audits = [
        audit_record(
            metric="energy_error_upper",
            residual=i["energy_error_upper"],
            rhs_norm=i["energy"],
            threshold=i["energy_threshold"],
            iterations=i["iterations"],
            accepted=i["accepted"],
            native=i,
        )
        if i.get("criterion") == "energy"
        else audit_record(
            metric="nodal_force_Linf",
            residual=i["residual_inf"],
            rhs_norm=i["rhs_inf"],
            threshold=i["threshold"],
            iterations=i["iterations"],
            accepted=i["accepted"],
            native=i,
        )
        for i in native["infos"]
    ]
    return dict(
        C=native["C"],
        C_stress=native["C_stress"],
        audits=audits,
        infos=native["infos"],
        reference=native["reference"],
        implementation="CuFHom-HEX8-staged"
        if execution == "reference"
        else "CuFHom-HEX8",
        execution="staged_quadrature" if execution == "reference" else "nodal_gather",
        memory=dict(
            policy=options.storage,
            working_state="device" if options.backend == "cupy" else "host",
            retained=native["storage"],
            residency=native.get("residency", "device"),
        ),
        native_report=native,
    )
