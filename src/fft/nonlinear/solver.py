"""Periodic small-strain FFT homogenization with J2 or Drucker-Prager plasticity.

Mandel order: 11,22,33,23,13,12. Stress/moduli use GPa, strain is dimensionless.
This API is independent of the six-load linear homogenization entrypoint.
"""

import copy
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cufhom.fft.krylov import NonlinearConvergenceError, _cg
from cufhom.fft.linear.solver import (
    AFFINE_ORIENTATION_QUANTIZATION,
    _group_quantized_orientations,
    _group_quantized_orientations_gpu,
    _rotation_matrices,
)
from cufhom.fft.nonlinear.curves import (
    export_constitutive_curve,
    loading_component,
    plot_constitutive_curve,
)
from cufhom.fft.nonlinear.krylov import FusedCoordinatePCG, TangentPipeline
from cufhom.fft.nonlinear.loading import MixedLoadingPath
from cufhom.fft.operators import (
    DisplacementCoordinates,
    IsotropicReference,
    MixedProjector,
    ReferenceStiffness,
    Timings,
)
from cufhom.materials import drucker_prager as dp_material
from cufhom.materials import paraboloidal
from cufhom.materials.elasticity import (
    TI_stiffness_voigt,
    rotate_C_mandel_batch,
    voigt_to_mandel,
)
from cufhom.materials.gradient_damage import (
    CuPyGradientDamage,
    CuPyParaboloidalDamage,
    GradientDamageParameters,
)
from cufhom.materials.plasticity import (
    CuPyMaterial,
    J2Material,
    MaterialResponse,
    PlasticState,
    commit_state,
    initial_state,
    tangent_cpu,
    update_cpu,
)
from cufhom_core.general.cupy_capacity import current_memory_pool
from cufhom_core.general.cupy_warp_reduction import CupyWarpCGReduction
from cufhom_core.tensors import fft as core_fft
from cufhom_core.tensors.fft import cp


def _table(phase, ori, material, fiber, xp=np):
    """Material ids (0 matrix, 1+ fiber orientation group) and their elastic Mandel matrices.

    With CuPy the orientation groups (same keys and numbering) are formed on
    the device and the ids never cross PCIe; the matrices stay on the host.
    """
    if xp is not np:
        flat, groups, axes = _group_quantized_orientations_gpu(
            phase, ori, AFFINE_ORIENTATION_QUANTIZATION
        )
        ids = cp.zeros(phase.size, dtype=cp.int32)
        ids[flat] = groups + 1
        del flat, groups
        # The FP64 orientation upload and sort buffers are transient; do not
        # let the pool keep them reserved beside the solver fields.
        current_memory_pool().free_all_blocks()
        return ids.reshape(phase.shape), _fiber_matrices(material, fiber, axes)
    phase, ori = np.asarray(phase), np.asarray(ori)
    fibers = phase == 1
    groups, axes = (
        _group_quantized_orientations(
            ori[fibers].astype(np.float64), AFFINE_ORIENTATION_QUANTIZATION
        )
        if fibers.any()
        else (None, ())
    )
    ids = np.zeros(phase.shape, dtype=np.int32)
    if len(axes):
        ids[fibers] = groups + 1
    return ids, _fiber_matrices(material, fiber, axes)


def _fiber_matrices(material, fiber, axes):
    C = np.zeros((6, 6))
    C[:3, :3] = material.K - 2 * material.mu / 3
    C += np.eye(6) * 2 * material.mu
    matrices = C[None]
    if len(axes):
        Cf = voigt_to_mandel(TI_stiffness_voigt(**fiber))
        if np.min(np.linalg.eigvalsh(Cf)) <= 0:
            raise ValueError("Fiber stiffness must be positive definite")
        matrices = np.concatenate(
            [matrices, rotate_C_mandel_batch(Cf, _rotation_matrices(axes))]
        )
    return matrices


def _host(x):
    return (
        cp.asnumpy(x) if cp is not None and isinstance(x, cp.ndarray) else np.asarray(x)
    )


def _fingerprint(phase, ori, configuration):
    h = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode())
    for a in (phase, ori):
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(memoryview(np.ascontiguousarray(a)).cast("B"))
    return h.hexdigest()


def save_checkpoint(
    path, strain, state, fingerprint, history, loading=None, damage_state=None
):
    """Atomic replace of a confirmed state only; never serializes Python objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as f:
            temp = f.name
            metadata = dict(
                version=1 if loading is None else 2,
                fingerprint=fingerprint,
                history=history,
            )
            fields = {}
            if damage_state is not None:
                metadata.update(version=3, damage_model="implicit_gradient")
                fields.update(
                    damage=_host(damage_state.value),
                    damage_history=_host(damage_state.history),
                )
            if loading is not None:
                row = history[-1]
                metadata.update(
                    loading_path=loading.descriptor(),
                    loading_state=dict(
                        strain_control=loading.strain_control.tolist(),
                        parameter=row["target_parameter"],
                        strain=row["target_strain_mandel"],
                        stress=row["target_stress"],
                    ),
                )
            np.savez(
                f,
                strain=_host(strain),
                plastic_strain=_host(state.plastic_strain),
                equivalent=_host(state.equivalent),
                metadata=json.dumps(metadata, allow_nan=False),
                **fields,
            )
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
        eta = 0.1
    else:
        eta = 0.9 * (error / previous_error) ** 1.5
        safeguard = 0.9 * previous_eta**1.5
        if safeguard > 0.1:
            eta = max(eta, safeguard)
    if target is not None:
        # Avoid resolving a Newton equation far below the OUTER tolerance.
        # Both norms are physical/isometric, with the same fixed stress scale
        # (the fused pipeline measures the inner one with the reference root).
        # The nonlinear residual/line search is still evaluated independently.
        # Analogous to the final-target safeguard of PETSc's EW version 3.
        eta = max(eta, 0.5 * target / error)
    return min(0.3, max(floor, eta))


def annotate_pseudotime(
    history, strain_scale=0.01, stress_scale=None, strain_control=None
):
    """Monotone path clock, including unloading; no physical rate dependence.

    tau = integral |d prescribed_Mandel_strain| / strain_scale. Rejected increments are
    absent from confirmed history and consequently cannot advance this clock.

    For general mixed paths the arc length combines controlled Mandel strain
    divided by strain_scale and prescribed stress divided by stress_scale.
    The load factor is then target_parameter. Neither clock is physical time.
    """
    tau, last = 0.0, None
    if strain_control is not None:
        control = np.asarray(strain_control, dtype=bool)
        if stress_scale is None or not np.isfinite(stress_scale) or stress_scale <= 0:
            raise ValueError(
                "General loading pseudotime requires a positive finite stress scale"
            )
        for row in history:
            value = np.concatenate(
                (
                    np.asarray(row["target_strain_mandel"])[control] / strain_scale,
                    np.asarray(row["target_stress"])[~control] / stress_scale,
                )
            )
            if last is not None:
                tau += float(np.linalg.norm(value - last))
            row["pseudo_time"] = tau
            row["load_factor"] = row["target_parameter"]
            last = value
        return
    for row in history:
        value = row["target_strain"]
        if last is not None:
            tau += abs(value - last) / strain_scale
        row["pseudo_time"] = tau
        row["load_factor"] = value / strain_scale
        last = value


def export_curve(result, directory):
    """Export accepted loading history and constitutive curves into a directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = result["history"]
    axis = result["configuration"]["axis"]
    mixed = result["configuration"].get("loading")
    # Older Newton-only results may be re-exported without rerunning physics.
    for row in rows:
        row.setdefault("target_strain", row["strain"][axis])
        row.setdefault("ptc_iterations", 0)
        row.setdefault("ptc_retries", 0)
        row.setdefault("ptc_dt_final", 0.0)
    annotate_pseudotime(
        rows,
        result.get("pseudo_time_strain_scale", 0.01),
        result.get("pseudo_time_stress_scale_GPa"),
        None if mixed is None else mixed["strain_control"],
    )
    labels = ["11", "22", "33", "23", "13", "12"]
    scalar = [
        "pseudo_time",
        "load_factor",
        "equivalent_plastic_mean",
        "plastic_fraction",
        "residual",
        "macro_residual",
        "yield_violation",
        "hill_mandel_error",
        "plastic_dissipation",
        "newton_iterations",
        "cg_iterations",
        "ptc_iterations",
        "ptc_retries",
        "ptc_dt_final",
        "seconds",
    ]
    if result.get("model") == "drucker_prager":
        scalar += [
            "hardening_variable_mean",
            "plastic_volumetric_mean",
            "krylov_iterations",
        ]
    if result.get("damage"):
        scalar += [
            "damage_mean",
            "damage_matrix_mean",
            "damage_max",
            "damage_dissipation",
            "damage_gap",
            "damage_iterations",
            "nonlocal_iterations",
            "nonlocal_residual",
        ]
    with (directory / "curve.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        target_columns = (
            []
            if mixed is None
            else ["target_parameter"]
            + ["target_strain_Mandel_" + s for s in labels]
            + ["target_stress_Mandel_GPa_" + s for s in labels]
        )
        writer.writerow(
            ["step"]
            + ["strain_" + s for s in labels]
            + ["stress_GPa_" + s for s in labels]
            + scalar
            + target_columns
        )
        for i, r in enumerate(rows):
            targets = (
                []
                if mixed is None
                else [r["target_parameter"]]
                + r["target_strain_mandel"]
                + r["target_stress"]
            )
            writer.writerow(
                [i] + r["strain"] + r["stress"] + [r[k] for k in scalar] + targets
            )
    export_constitutive_curve(result, directory)
    serializable = {k: v for k, v in result.items() if k not in ("fields", "state")}
    (directory / "metadata.json").write_text(
        json.dumps(serializable, indent=2, allow_nan=False)
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_constitutive_curve(ax, result)
    name = "Drucker–Prager" if result.get("model") == "drucker_prager" else "J2"
    description = (
        "parámetros demostrativos"
        if mixed is None
        else "trayectoria mixta, proyección de componente"
    )
    qualifier = (
        "daño de matriz regularizado, sin calibrar"
        if result.get("damage")
        else "sin daño ni rotura"
    )
    ax.set_title(f"FFT {name} — {description}\n{qualifier}", fontsize=11)
    fig.tight_layout()
    fig.savefig(directory / "curve.png", dpi=150)
    plt.close(fig)


def solve_nonlinear_homogenization(parameters):
    """Periodic equilibrium under scalar or general mixed Mandel loading.

    axis=0,1,2,3,4,5 selects 11,22,33,23,13,12, respectively. For shear,
    path/increments/pseudo_time_strain_scale use gamma/sqrt(2); history and
    fields remain Mandel. constitutive.csv and curve.png use gamma and tau.

    Required: phase_array (0=matrix,1=fiber), ori_array (...,3). Optional:
    model ('j2' default, or 'drucker_prager'), matrix (material fields),
    fiber (TI E_L,E_T,nu_LT,nu_TT,G_LT), backend
    ('scipy'/'cupy'), dtype, path=(0,.01), axis=0, initial_increment=.00075,
    min_increment=1e-7, rtol, max_newton=25, max_cg=1000, output_dir,
    checkpoint_path, restart_from, checkpoint_every=0, return_fields=False,
    profile=False, on_step (called only after commit). Optional pseudotime
    controls: pseudo_time_strain_scale=.01, initial_pseudo_step, min_pseudo_step.
    A pseudo step multiplies the strain scale; it replaces, not supplements,
    the corresponding strain increment. integrator='midpoint' (default) is the
    second-order generalized midpoint return map; 'backward_euler' is the
    first-order radial return. Defaults use formulation='displacement'
    and preconditioner='matrix', the verified fast G09 path; pass
    formulation='strain', preconditioner='none' for the direct reference.
    coordinate_pcg='auto' uses fused reference PCG for CuPy float32 displacement
    with matrix preconditioning; other configurations keep classical CG.
    reuse_trial_response=True avoids reevaluating accepted Newton candidates.
    Default inner_forcing='ew_target' on that fused route avoids oversolving
    near the final tolerance; other routes retain 'ew'. Final acceptance is
    unchanged. For the historical path select coordinate_pcg='classic',
    reuse_trial_response=False, inner_forcing='ew'.
    diagnostic_reduction='auto' fuses accepted-trial GPU diagnostics in FP64
    for J2/DP without damage. 'reference' keeps the array-based checks;
    'fused' requires the supported GPU path. Constitutive/Krylov precision
    and acceptance thresholds are unchanged. Inputs are never mutated.
    dtype='float64' on the GPU defaults to working='float32': FP64 state,
    law, residual and acceptance with FP32 fused tangent solves (mixed
    precision); the paraboloidal and kinematic, saturating or viscous J2
    returns then run in double-float (pairs of FP32, ~46 bits, at the FP32 rate; see
    materials/double_float). working='float64' keeps IEEE FP64 throughout.
    Drucker-Prager uses backward_euler with complete cone/apex return.
    Independent friction/dilatancy coefficients select CG (associated) or
    BiCGStab(2) completed by GMRES (nonassociated, and the paraboloidal law),
    retaining the compact fused tangent; linear_solver='gmres' keeps GMRES.
    Its hardening variable is the accumulated multiplier, including apex.
    Optional damage={...} selects GPU implicit-gradient matrix damage with
    effective DP or paraboloidal stress and a staggered nonlocal solve;
    interphase_thickness adds a damaging fiber-matrix interphase. Plastic and damage
    histories remain frozen until the coupled increment is accepted.

    General loading replaces path with loading={strain_control: six booleans,
    strain: (knots,6), stress: (knots,6), parameter: optional increasing knots}.
    Strain is Mandel, stress is Mandel/GPa, and the mask is fixed throughout.
    All-strain and all-stress masks are permitted for well-posed trajectories.
    Inactive tensor entries are ignored. Parameter defaults to linspace(0,1)
    and is not time; initial_increment/min_increment now bound its increments
    (defaults .1/1e-8). initial_pseudo_step/min_pseudo_step instead bound the
    scaled mixed arc length. A virgin path must begin at zero prescribed
    tensors. A continuation starts at the checkpoint parameter and tensors;
    future knots may change. General loading disables exact elastic jumps.
    History keeps scalar target_strain for the selected display axis, adding
    target_strain_mandel, target_stress and target_parameter for full controls.
    Component exports remain physical projections of the multiaxial history.
    """
    old = core_fft.get_fft_backend_status()
    try:
        return _solve(parameters)
    finally:
        core_fft.set_fft_backend(old["requested"])
        core_fft.set_cupy_plan_mode(old["cupy_plan_mode"], preserve_cache=True)


def _solve(p):
    if any(
        k in p
        for k in ("fiber_fraction_array", "interface_normal_array", "Cfield_array")
    ):
        raise ValueError(
            "Nonlinear models accept binary voxels only; no composite constitutive mixing"
        )
    backend = p.get("backend", "scipy")
    if backend not in ("scipy", "cupy") or (backend == "cupy" and cp is None):
        raise ValueError("Use available backend scipy or cupy")
    xp = np if backend == "scipy" else cp
    dtype = np.dtype(p.get("dtype", "float64" if xp is np else "float32"))
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    phase = np.asarray(p["phase_array"]).view()
    ori = np.asarray(p["ori_array"]).view()
    if phase.ndim != 3 or min(phase.shape) < 2 or ori.shape != phase.shape + (3,):
        raise ValueError("Expected 3D phase and (...,3) orientations")
    if not np.isin(phase, [0, 1]).all() or not np.isfinite(ori).all():
        raise ValueError("Finite orientations and binary phases required")
    if np.any(np.linalg.norm(ori[phase == 1], axis=-1) < 1e-12):
        raise ValueError("Fiber orientations must be nonzero")
    phase.setflags(write=False)
    ori.setflags(write=False)
    model = p.get("model", "j2")
    if model not in ("j2", "drucker_prager", "paraboloidal"):
        raise ValueError("model must be 'j2', 'drucker_prager' or 'paraboloidal'")
    pressure_dependent = model != "j2"
    damage_parameters = None
    if p.get("damage") is not None:
        if model == "j2" or xp is np:
            raise ValueError(
                "Implicit-gradient damage requires a pressure-dependent matrix and the CuPy GPU backend"
            )
        try:
            damage_parameters = GradientDamageParameters(**p["damage"])
        except TypeError as exc:
            raise ValueError("Invalid damage parameter mapping") from exc
    # Constitutive law: CPU reference functions, GPU class and material type.
    law = dict(
        j2=SimpleNamespace(
            Material=J2Material,
            update_cpu=update_cpu,
            tangent_cpu=tangent_cpu,
            GPU=CuPyMaterial,
        ),
        drucker_prager=SimpleNamespace(
            Material=dp_material.DruckerPragerMaterial,
            update_cpu=dp_material.update_cpu,
            tangent_cpu=dp_material.tangent_cpu,
            GPU=dp_material.CuPyDruckerPrager,
        ),
        paraboloidal=SimpleNamespace(
            Material=paraboloidal.ParaboloidalMaterial,
            update_cpu=paraboloidal.update_cpu,
            tangent_cpu=paraboloidal.tangent_cpu,
            GPU=paraboloidal.CuPyParaboloidal,
        ),
    )[model]
    material = law.Material(**p.get("matrix", {}))
    nonsymmetric = pressure_dependent and not material.associated
    # Nonsymmetric tangents: BiCGStab, completed by GMRES when it breaks down.
    # linear_solver='gmres' keeps restarted GMRES alone, for comparisons.
    linear_solver = p.get("linear_solver", "bicgstab") if nonsymmetric else "cg"
    if linear_solver not in (("bicgstab", "gmres") if nonsymmetric else ("cg",)):
        raise ValueError(
            "linear_solver must be 'bicgstab' or 'gmres' for nonsymmetric tangents; "
            "symmetric tangents use CG"
        )
    krylov_fallbacks = 0
    gmres_restart = int(p.get("gmres_restart", 12))
    if not 2 <= gmres_restart <= 64:
        raise ValueError("gmres_restart must be between 2 and 64")
    fiber = p.get("fiber", dict(E_L=70.0, E_T=8.0, nu_LT=0.25, nu_TT=0.3, G_LT=5.0))
    axis = int(p.get("axis", 0))
    if "loading" in p and "path" in p:
        raise ValueError("Specify only one of loading and legacy scalar path")
    loading = MixedLoadingPath.from_mapping(p["loading"]) if "loading" in p else None
    path = (
        loading.parameter
        if loading is not None
        else np.asarray(p.get("path", [0.0, 0.01]), dtype=float)
    )
    if loading is not None and not p.get("restart_from"):
        loading.validate_start()
    lengths = tuple(p.get("lengths", [1.0, 1.0, 1.0]))
    pseudo_scale = float(p.get("pseudo_time_strain_scale", 0.01))
    if not np.isfinite(pseudo_scale) or pseudo_scale <= 0:
        raise ValueError("pseudo_time_strain_scale must be finite and positive")
    pseudo_stress_scale = float(
        p.get("pseudo_time_stress_scale_GPa", material.yield_stress)
    )
    if loading is not None and (
        not np.isfinite(pseudo_stress_scale) or pseudo_stress_scale <= 0
    ):
        raise ValueError("pseudo_time_stress_scale_GPa must be finite and positive")
    for a, b in [
        ("initial_increment", "initial_pseudo_step"),
        ("min_increment", "min_pseudo_step"),
    ]:
        if a in p and b in p:
            raise ValueError("Specify only one of " + a + " and " + b)
    # The second-order midpoint map is 25-90x more accurate than backward Euler
    # at equal steps; 3x the former 2.5e-4 default stays 2-30x more accurate.
    if loading is None:
        inc = float(
            p.get(
                "initial_increment",
                float(p.get("initial_pseudo_step", 0.075)) * pseudo_scale,
            )
        )
        minimum = float(
            p.get("min_increment", float(p.get("min_pseudo_step", 1e-5)) * pseudo_scale)
        )
    else:
        inc = float(p.get("initial_increment", p.get("initial_pseudo_step", 0.1)))
        minimum = float(p.get("min_increment", p.get("min_pseudo_step", 1e-8)))
    rtol = float(p.get("rtol", 1e-8 if dtype == np.float64 else 1e-5))
    max_newton, max_cg = int(p.get("max_newton", 25)), int(p.get("max_cg", 1000))
    integrator = p.get(
        "integrator", "backward_euler" if pressure_dependent else "midpoint"
    )
    if integrator not in ("midpoint", "backward_euler"):
        raise ValueError("integrator must be 'midpoint' or 'backward_euler'")
    if pressure_dependent and integrator != "backward_euler":
        raise ValueError("Pressure-dependent models require integrator=backward_euler")
    load_rate = p.get("load_rate")
    if material.viscosity:
        load_rate = float(load_rate) if load_rate is not None else np.nan
        if not np.isfinite(load_rate) or load_rate <= 0:
            raise ValueError(
                "Viscoplasticity requires a positive load_rate (path parameter per second)"
            )
    elif load_rate is not None:
        raise ValueError("load_rate applies to viscoplastic materials only")
    # A viscous committed state lies outside the static yield surface, so the
    # exact elastic reach from it is not defined: rate-dependent paths step.
    elastic_segments = bool(
        p.get(
            "elastic_segments",
            not pressure_dependent and loading is None and not material.viscosity,
        )
    )
    if elastic_segments and material.viscosity:
        raise ValueError(
            "Exact elastic-segment jumping requires a rate-independent material"
        )
    if elastic_segments and loading is not None:
        raise ValueError(
            "General mixed loading requires elastic_segments=False; scalar J2 reach is not applicable"
        )
    if pressure_dependent and elastic_segments:
        raise ValueError(
            "Exact elastic-segment jumping is currently available for J2 only"
        )
    cg_projection = p.get("cg_projection", "galerkin")
    fused_cg = bool(p.get("fused_cg", True))
    coordinate_pcg = p.get("coordinate_pcg", "auto")
    reuse_trial_response = bool(p.get("reuse_trial_response", True))
    diagnostic_reduction = p.get("diagnostic_reduction", "auto")
    if diagnostic_reduction not in ("auto", "reference", "fused"):
        raise ValueError("diagnostic_reduction must be auto, reference or fused")
    diagnostic_fused = (
        xp is cp
        and damage_parameters is None
        and model != "paraboloidal"
        and diagnostic_reduction != "reference"
    )
    if diagnostic_reduction == "fused" and not diagnostic_fused:
        raise ValueError("Fused diagnostics require CuPy J2/DP without damage")

    if coordinate_pcg not in ("auto", "classic", "fused", "pipeline"):
        raise ValueError("coordinate_pcg must be auto, classic, fused or pipeline")
    # Production default: the verified G09 fast path.  The strain coordinates
    # remain available as a transparent reference formulation.
    preconditioner = p.get("preconditioner", "matrix")
    # The historical double-projection diagnostic is defined in strain
    # coordinates.  Keep it usable when a caller selects it explicitly,
    # while the normal Galerkin route defaults to the measured fast path.
    formulation = p.get(
        "formulation", "strain" if cg_projection == "double" else "displacement"
    )
    if formulation not in ("strain", "displacement"):
        raise ValueError("formulation must be strain or displacement")
    if formulation == "displacement" and cg_projection != "galerkin":
        raise ValueError("displacement formulation requires Galerkin CG")
    if preconditioner not in ("none", "matrix"):
        raise ValueError("preconditioner must be 'none' or 'matrix'")
    # Mixed precision (Kelley 2022): with dtype float64 on the GPU the state,
    # constitutive law, Newton residual, line search and acceptance stay FP64
    # while the tangent solves run on the FP32 fused pipeline. An inexact
    # Newton solve only sets the convergence rate; the attainable accuracy is
    # that of the FP64 residual. working='float64' keeps FP64 Krylov solves.
    working = p.get("working")
    if working not in (None, "float32", "float64"):
        raise ValueError("working must be None, 'float32' or 'float64'")
    structural = (
        xp is cp
        and fused_cg
        and formulation == "displacement"
        and preconditioner == "matrix"
    )
    if working is None:
        working = (
            "float32"
            if dtype == np.float64
            and structural
            and coordinate_pcg in ("auto", "pipeline")
            and TangentPipeline.supported(phase.shape)
            else dtype.name
        )
    if working == "float64" and dtype == np.float32:
        raise ValueError("working precision cannot exceed dtype")
    mixed = dtype == np.float64 and working == "float32"
    if mixed and not (
        structural
        and TangentPipeline.supported(phase.shape)
        and coordinate_pcg in ("auto", "pipeline")
    ):
        raise ValueError(
            "Mixed precision needs the fused pipeline: CuPy, displacement, matrix, "
            "fused_cg and a supported grid"
        )
    fused_coordinate_compatible = structural and (dtype == np.float32 or mixed)
    if coordinate_pcg in ("fused", "pipeline") and not fused_coordinate_compatible:
        raise ValueError(
            "Fused coordinate PCG requires CuPy float32 (or mixed), displacement, matrix and fused_cg"
        )
    # 'pipeline': fused axis-split FFTs for Newton solves; PTC shifts keep 'fused'.
    pipeline_compatible = fused_coordinate_compatible and TangentPipeline.supported(
        phase.shape
    )
    if coordinate_pcg == "pipeline" and not pipeline_compatible:
        raise ValueError("The fused tangent pipeline needs FFT sizes with primes <= 13")
    coordinate_pcg_used = (
        "classic"
        if coordinate_pcg == "classic" or not fused_coordinate_compatible
        else "pipeline"
        if coordinate_pcg in ("auto", "pipeline") and pipeline_compatible
        else "fused"
    )
    inner_forcing = p.get(
        "inner_forcing", "ew" if coordinate_pcg_used == "classic" else "ew_target"
    )
    if inner_forcing not in ("legacy", "ew", "ew_target"):
        raise ValueError("inner_forcing must be legacy, ew or ew_target")
    if cg_projection not in ("double", "galerkin"):
        raise ValueError("cg_projection must be double or galerkin")
    globalization = p.get("globalization", "auto")
    ptc_initial = float(p.get("ptc_initial_dt", 10.0))
    ptc_min, ptc_max = (
        float(p.get("ptc_min_dt", 1e-8)),
        float(p.get("ptc_max_dt", 1e12)),
    )
    max_ptc = int(p.get("max_ptc", 60))
    explicit_steps = int(p.get("ptc_explicit_steps", 0))
    if not 0 <= explicit_steps <= 100:
        raise ValueError("ptc_explicit_steps must be between 0 and 100")
    if globalization not in ("auto", "newton", "ptc"):
        raise ValueError("globalization must be auto, newton or ptc")
    if (
        not np.isfinite([ptc_initial, ptc_min, ptc_max]).all()
        or not 0 < ptc_min <= ptc_initial <= ptc_max
        or max_ptc < 1
    ):
        raise ValueError("Invalid PTC pseudostep bounds")
    if int(p.get("checkpoint_every", 0)) < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    if (
        axis not in range(6)
        or path.ndim != 1
        or len(path) < 2
        or not np.isfinite(path).all()
    ):
        raise ValueError("Invalid strain path or Mandel axis (expected 0..5)")
    if len(lengths) != 3 or not np.isfinite(lengths).all() or min(lengths) <= 0:
        raise ValueError("Domain lengths must be finite positive values")
    if (
        not np.isfinite([inc, minimum, rtol]).all()
        or min(inc, minimum) <= 0
        or not 0 < rtol < 1
    ):
        raise ValueError("Invalid increment bounds or tolerance")
    if loading is None and minimum > inc:
        raise ValueError("Invalid increment bounds or tolerance")
    if min(max_newton, max_cg) < 1:
        raise ValueError("Iteration limits must be positive")
    segment_inc = np.full(len(path) - 1, inc)
    segment_minimum = np.full(len(path) - 1, minimum)
    if loading is not None:
        controls = loading.strain_control
        increments = np.concatenate(
            (
                np.diff(loading.strain[:, controls], axis=0) / pseudo_scale,
                np.diff(loading.stress[:, ~controls], axis=0) / pseudo_stress_scale,
            ),
            axis=1,
        )
        speeds = np.linalg.norm(increments, axis=1) / np.diff(path)
        if not np.isfinite(speeds).all():
            raise ValueError("Mixed path has a nonfinite scaled arc length")
        for name, bounds in [
            ("initial_pseudo_step", segment_inc),
            ("min_pseudo_step", segment_minimum),
        ]:
            if name in p:
                np.divide(float(p[name]), speeds, out=bounds, where=speeds > 0)
                bounds[speeds == 0] = min(float(p[name]), float(np.min(np.diff(path))))
        if (
            not np.isfinite(segment_inc).all()
            or not np.isfinite(segment_minimum).all()
            or np.any(segment_inc <= 0)
            or np.any(segment_minimum <= 0)
            or np.any(segment_minimum > segment_inc)
        ):
            raise ValueError(
                "Invalid parameter increment bounds after mixed pseudostep conversion"
            )
    timers = Timings(xp, p.get("profile", False))
    timers.sync()
    started = time.perf_counter()
    package_root = Path(__file__).resolve().parents[2]
    source_paths = [
        package_root / name
        for name in (
            "fft/nonlinear/solver.py",
            "fft/nonlinear/krylov.py",
            "fft/nonlinear/loading.py",
            "fft/nonlinear/curves.py",
            "fft/operators.py",
            "fft/krylov.py",
            "fft/linear/solver.py",
            "materials/plasticity.py",
            "materials/drucker_prager.py",
            "materials/paraboloidal.py",
            "materials/elasticity.py",
        )
    ]
    if diagnostic_fused:
        source_paths.append(package_root / "fft/nonlinear/diagnostics.py")
    if nonsymmetric:
        source_paths.append(package_root / "fft/nonlinear/gmres.py")
    if damage_parameters is not None:
        source_paths += [
            package_root / "materials/gradient_damage.py",
            package_root / "fft/nonlinear/nonlocal_damage.py",
        ]
    source_paths += [
        Path(core_fft.__file__),
        Path(core_fft.__file__).with_name("objects.py"),
    ]
    source_hashes = {
        str(f.relative_to(package_root.parent)): hashlib.sha256(
            f.read_bytes()
        ).hexdigest()
        for f in source_paths
    }
    core_fft.set_fft_backend(backend)
    core_fft.set_cupy_plan_mode("manual", preserve_cache=True)
    config = dict(
        matrix=material.configuration(),
        fiber=fiber,
        axis=axis,
        lengths=lengths,
        shape=list(phase.shape),
        conventions="Mandel 11,22,33,23,13,12; GPa",
        model=material.model_name,
        calibrated=False,
    )
    if material.viscosity:
        config["load_rate"] = load_rate
    if loading is not None:
        config["loading"] = loading.fingerprint_configuration(integrator)
    if damage_parameters is not None:
        config["damage"] = damage_parameters.configuration()
    fingerprint = _fingerprint(phase, ori, config)
    with timers.measure("geometry_and_transfer"):
        ids, table_cpu = _table(phase, ori, material, fiber, xp)
        # J2 positive-hardening tangents are bounded above by their elastic
        # stiffness. P is orthogonal, so this also bounds P C_alg P.
        elastic_bound = float(np.max(np.linalg.eigvalsh(table_cpu)))
        table = xp.asarray(table_cpu, dtype=dtype)
        state = initial_state(phase.shape, xp, dtype)
        strain = xp.zeros((6,) + phase.shape, dtype=dtype)
    del table_cpu
    if loading is None:
        control = np.zeros(6, dtype=bool)
        control[axis] = True
    else:
        control = loading.strain_control
    projector = MixedProjector(
        phase.shape,
        lengths,
        control,
        xp,
        timers,
        spectral_fusion=p.get("spectral_fusion", True),
    )
    reference = (
        None
        if preconditioner == "none"
        else IsotropicReference(projector, material.K, material.mu)
    )
    stiffness = ReferenceStiffness(
        projector, material.K, material.mu, dtype
    )  # PTC mass K0
    coordinates = (
        DisplacementCoordinates(projector) if formulation == "displacement" else None
    )
    # The pipeline leaves only PTC-shifted solves to the coordinate engine.
    coordinate_engine = (
        FusedCoordinatePCG(coordinates) if coordinate_pcg_used == "fused" else None
    )
    # Pipeline solves read complex64 coordinates; the FP64 coordinate dot
    # would reinterpret them. Both metrics accumulate products in FP64.
    tangent_pipeline = (
        TangentPipeline(projector, material, ids, table)
        if coordinate_pcg_used == "pipeline"
        else None
    )
    # Mixed route: iterative returns (general J2, paraboloidal), arithmetic-bound
    # at the 1:64 FP64 rate, run in double-float. The closed-form linear J2
    # and Drucker-Prager returns are cheaper than the FP64<->df64 conversions
    # and stay IEEE FP64.
    law_arithmetic = (
        "df64"
        if mixed and (model == "paraboloidal" or model == "j2" and material.general)
        else "ieee"
    )
    gpu = (
        None
        if xp is np
        else law.GPU(cp, material, ids, table, dtype, arithmetic=law_arithmetic)
        if model != "drucker_prager"
        else law.GPU(cp, material, ids, table, dtype)
    )
    damage = None
    if damage_parameters is not None:
        law_damage = (
            CuPyParaboloidalDamage if model == "paraboloidal" else CuPyGradientDamage
        )
        gpu = damage = law_damage(
            cp, material, ids, table, dtype, damage_parameters, lengths, timers
        )
    diagnostic_engine = None
    if diagnostic_fused:
        from cufhom.fft.nonlinear.diagnostics import FusedDiagnostics

        diagnostic_engine = FusedDiagnostics(
            cp, material, ids, dtype, pressure=pressure_dependent, rtol=rtol
        )
    history, rejected = [], []
    current = 0.0 if loading is None else float(path[0])
    previous_delta, previous_step = None, 0.0
    older_delta, older_step = None, 0.0
    if p.get("restart_from"):
        with np.load(p["restart_from"], allow_pickle=False) as ck:
            meta = json.loads(str(ck["metadata"]))
            expected_version = (
                3 if damage is not None else (1 if loading is None else 2)
            )
            if (
                meta["version"] != expected_version
                or meta["fingerprint"] != fingerprint
            ):
                raise ValueError("Checkpoint geometry/material/loading mismatch")
            for key, shape in [
                ("strain", strain.shape),
                ("plastic_strain", strain.shape),
                ("equivalent", phase.shape),
            ]:
                if ck[key].shape != shape or not np.isfinite(ck[key]).all():
                    raise ValueError("Invalid checkpoint field " + key)
            strain = xp.asarray(ck["strain"], dtype=dtype)
            state = PlasticState(
                xp.asarray(ck["plastic_strain"], dtype=dtype),
                xp.asarray(ck["equivalent"], dtype=dtype),
            )
            if np.min(ck["equivalent"]) < 0:
                raise ValueError("Checkpoint equivalent plastic strain is negative")
            if damage is not None:
                if meta.get("damage_model") != "implicit_gradient" or not {
                    "damage",
                    "damage_history",
                } <= set(ck.files):
                    raise ValueError("Checkpoint damage state mismatch")
                damage.restore(ck["damage"], ck["damage_history"])
            history = meta["history"]
            if loading is None:
                current = history[-1].get("target_strain", history[-1]["strain"][axis])
            else:
                current = loading.validate_restart(meta, history)
                prescribed, _ = loading.at(current)
                stored_mean = np.mean(ck["strain"], axis=(1, 2, 3), dtype=np.float64)
                tolerance = max(rtol, 32 * np.finfo(dtype).eps) * max(
                    pseudo_scale, np.linalg.norm(prescribed)
                )
                if np.linalg.norm((stored_mean - prescribed)[control]) > tolerance:
                    raise ValueError(
                        "Checkpoint field does not match prescribed mean strain"
                    )
    if abs(path[0] - current) > 1e-9:
        raise ValueError(
            "Path must begin at zero, or at the checkpoint prescribed Mandel strain"
        )
    current = float(path[0])
    for row in history:
        row.setdefault("target_strain", row["strain"][axis])
        if loading is None:
            row.setdefault("target_parameter", row["target_strain"])
            row.setdefault("target_stress", [0.0] * 6)

    def annotate_history():
        annotate_pseudotime(
            history,
            pseudo_scale,
            pseudo_stress_scale,
            None if loading is None else control,
        )

    def conditions(parameter):
        if loading is not None:
            return loading.at(parameter)
        prescribed = np.zeros(6)
        prescribed[axis] = parameter
        return prescribed, np.zeros(6)

    def stamp_targets(row, parameter):
        prescribed_strain, prescribed_stress = conditions(parameter)
        row.update(
            target_strain=float(prescribed_strain[axis]),
            target_stress=prescribed_stress.tolist(),
            target_parameter=float(parameter),
        )
        if loading is not None:
            row["target_strain_mandel"] = prescribed_strain.tolist()

    annotate_history()

    def dot(a, b):
        with timers.measure("reductions"):
            # float64 accumulation; multiplication is promoted too, without
            # retaining double-sized fields (einsum CPU, ReductionKernel GPU).
            if xp is np:
                return float(np.einsum("i,i->", a.ravel(), b.ravel(), dtype=np.float64))
            reduction.dot_kernel(
                (reduction.blocks,),
                (reduction.threads,),
                (
                    cp.ascontiguousarray(a),
                    cp.ascontiguousarray(b),
                    reduction.partial,
                    np.int64(a.size),
                ),
            )
            scalar = cp.sum(reduction.partial, dtype=cp.float64)
        # Keep the host transfer outside the device reduction timing.
        return float(scalar)

    def coordinate_dot(a, b):
        return dot(a.view(dtype), b.view(dtype))

    if mixed:
        from cufhom.fft.nonlinear.gmres import euclidean_dot as pipeline_dot
    else:
        pipeline_dot = coordinate_dot

    if nonsymmetric:
        from cufhom.fft.nonlinear.gmres import real_inner_products

        # Same Euclidean metric as the FP64 warp dot, with all Arnoldi
        # coefficients reduced in one batch and one scalar-vector transfer.
        dot.gmres_batch = coordinate_dot.gmres_batch = real_inner_products

    reduction = (
        None
        if xp is np
        else CupyWarpCGReduction(
            SimpleNamespace(N=phase.shape, val=strain), max_blocks=16384
        )
    )

    def norm(a):
        return np.sqrt(dot(a, a) / phase.size)

    def means(field):
        # FP64 component means; on the GPU one full reduction per component
        # is ~10x faster than CuPy's multi-axis mean.
        if xp is np:
            return np.mean(field, axis=(1, 2, 3), dtype=np.float64)
        return (
            _host(xp.stack([xp.sum(c, dtype=xp.float64) for c in field])) / phase.size
        )

    viscous = 0.0  # overstress modulus of the current attempt, 3/2 eta/dt

    def update(e):
        with timers.measure("constitutive"):
            # The midpoint map also reads the strain at the start of the step.
            start = strain if integrator == "midpoint" else None
            rate = dict(viscous=viscous) if material.viscosity else {}
            return (
                gpu.update(e, state, start, **rate)
                if gpu
                else law.update_cpu(e, state, material, ids, table, start, **rate)
            )

    def tangent(v, response):
        with timers.measure("tangent"):
            return (
                gpu.tangent(v, response)
                if gpu
                else law.tangent_cpu(v, response, material, ids, table)
            )

    def equilibrium(e, prescribed_stress=None):
        response = update(e)
        if coordinates is not None:
            # Q is an isometry with QQ* = P: the residual Q*sigma (one forward
            # transform) has the norm of P sigma and is the Newton right-hand
            # side itself. A prescribed uniform stress only moves the free
            # mean slots, by sqrt(N) times its components.
            residual = coordinates.adjoint(response.stress)
            if prescribed_stress is not None:
                residual[3 * coordinates.count :] -= (
                    np.sqrt(phase.size) * prescribed_stress.ravel()[projector.free]
                )
            return (
                response,
                residual,
                np.sqrt(coordinate_dot(residual, residual) / phase.size)
                / material.yield_stress,
            )
        residual = projector(response.stress)
        # A uniform prescribed stress is already in the range of P on free
        # means. Subtract it here, saving an extra field allocation/FFT pass.
        # The six-vector is prepared once per attempt; controlled entries are
        # canonical zeros. The legacy zero-stress path does no extra work.
        if prescribed_stress is not None:
            residual -= prescribed_stress
        # Fixed material stress scale avoids artificial convergence near zero.
        return response, residual, norm(residual) / material.yield_stress

    def solve_tangent(response, residual, inner, shift=0.0):
        """Newton correction: (P C_alg P + shift K0/E) de = -residual, to a relative tolerance.

        The PTC mass is the isotropic reference stiffness K0 of the
        preconditioner, scaled by 1/E (shift = E/dt), so the shifted system is
        preconditioned exactly by K0^-1/(1+shift/E) and is H + (shift/E) I in
        the pipeline's whitened coordinates. As dt -> 0 the PTC step tends to
        the basic (Moulinec-Suquet) step -dt K0^-1 P sigma.
        In displacement coordinates residual is Q*sigma and de = Q z.
        """
        mass = shift / material.E
        nonlocal coordinate_engine

        def nonsymmetric_krylov(op, rhs, inner_dot, precondition):
            nonlocal krylov_fallbacks
            from cufhom.fft.nonlinear.gmres import gmres, nonsymmetric_solve

            if linear_solver == "gmres":
                return gmres(
                    op,
                    rhs,
                    inner_dot,
                    inner,
                    max_cg,
                    precondition=precondition,
                    restart=gmres_restart,
                )
            x, count, fallback = nonsymmetric_solve(
                op,
                rhs,
                inner_dot,
                inner,
                max_cg,
                precondition=precondition,
                restart=gmres_restart,
            )
            krylov_fallbacks += fallback
            return x, count

        def action(v):
            out = tangent(v, response)
            if cg_projection == "double":
                out = projector(out)
            if mass:
                # K0 v voxelwise: 2 mu v + (K - 2 mu/3) tr(v) on the normal components.
                out += (mass * 2 * material.mu) * v
                out[:3] += (mass * (material.K - 2 * material.mu / 3)) * (
                    v[0] + v[1] + v[2]
                )
            return out

        with timers.measure("cg_total_inclusive"):
            if tangent_pipeline is not None and (not mass or nonsymmetric):
                return tangent_pipeline.solve(
                    response,
                    residual,
                    inner,
                    max_cg,
                    pipeline_dot,
                    method=linear_solver,
                    restart=gmres_restart,
                    shift=mass,
                )
            if reference is not None:
                reference.set_shift(mass, dtype)
            if coordinates is None:
                if nonsymmetric:
                    return nonsymmetric_krylov(
                        lambda v: projector(action(projector(v))),
                        -residual,
                        dot,
                        reference,
                    )
                return _cg(
                    action,
                    -residual,
                    dot,
                    inner,
                    max_cg,
                    xp,
                    project=projector,
                    fused=fused_cg,
                    precondition=reference,
                )

            def reduced_action(z):
                out = coordinates.adjoint(tangent(coordinates.strain(z), response))
                if mass:
                    out += coordinates.precondition(z, stiffness.scaled(mass))
                return out

            rhs = -residual
            if (
                coordinate_engine is None
                and coordinate_pcg_used == "pipeline"
                and not mixed
            ):
                coordinate_engine = FusedCoordinatePCG(coordinates)
            if nonsymmetric:
                reduced_reference = (
                    None
                    if reference is None
                    else lambda z: coordinates.precondition(z, reference)
                )
                z, count = nonsymmetric_krylov(
                    reduced_action, rhs, coordinate_dot, reduced_reference
                )
            elif coordinate_engine is None:
                reduced_reference = (
                    None
                    if reference is None
                    else lambda z: coordinates.precondition(z, reference)
                )
                z, count = _cg(
                    reduced_action,
                    rhs,
                    coordinate_dot,
                    inner,
                    max_cg,
                    xp,
                    fused=fused_cg,
                    precondition=reduced_reference,
                )
            else:
                z, count = coordinate_engine.solve(
                    reduced_action, rhs, coordinate_dot, inner, max_cg, reference
                )
            return coordinates.strain(z), count

    def elastic_reach(direction):
        """Exact elastic extent of the prescribed strain from the committed state.

        With the plastic state frozen the response is linear in the prescribed
        component: one elastic solve gives the strain rate d(e)/dE of the mixed
        problem, and in each matrix voxel 1.5|s - X + t ds|^2 = R(alpha)^2
        is a quadratic in the load t (X: backstress). Returns (t*, strain rate, CG iterations):
        t* is the largest admissible elastic increment (inf if none yields).
        """
        elastic = MaterialResponse(
            None,
            xp.zeros_like(strain),
            xp.full(phase.shape, 2 * material.mu, dtype=dtype),
            xp.zeros(phase.shape, dtype=dtype),
            xp.zeros(phase.shape, dtype=dtype),
        )
        unit = xp.zeros_like(strain)
        unit[axis] = direction
        loaded = tangent(unit, elastic)
        # The jump is only a Newton guess: FP32 mixed solves stop at their floor.
        correction, count = solve_tangent(
            elastic,
            projector(loaded) if coordinates is None else coordinates.adjoint(loaded),
            max(rtol, 1e-5) if mixed else rtol,
        )
        del loaded
        rate = unit + (projector(correction) if coordinates is None else correction)
        sigma, dsigma = update(strain).stress, tangent(rate, elastic)
        s, ds = sigma.copy(), dsigma
        s[:3] -= xp.sum(s[:3], axis=0) / 3
        ds[:3] -= xp.sum(ds[:3], axis=0) / 3
        backstress = material.backstress(state.plastic_strain, xp)
        if backstress is not None:
            s -= backstress
        a = 1.5 * xp.sum(ds * ds, axis=0, dtype=xp.float64)
        b = 3 * xp.sum(s * ds, axis=0, dtype=xp.float64)
        radius = material.radius(state.equivalent.astype(xp.float64), xp)
        c = xp.minimum(
            1.5 * xp.sum(s * s, axis=0, dtype=xp.float64) - radius * radius, 0
        )
        root = (-b + xp.sqrt(b * b - 4 * a * c)) / xp.where(a > 0, 2 * a, 1)
        reach = xp.where((ids == 0) & (a > 0), root, np.inf)
        return float(xp.min(reach)), rate, count

    def admissible(e, target):
        # Fixed macroscopic strain and admissibility must not drift over a path.
        e = projector(e)
        if loading is None:
            e[axis] += target
        else:
            e += target
        return e

    def mechanical_attempt(target, use_ptc=False, guess=None):
        nonlocal viscous
        delta = target - current
        if material.viscosity:
            viscous = material.viscous_modulus(abs(delta) / load_rate)
        # Per-attempt boundary conditions are frozen alongside the confirmed
        # plastic state. Line searches/PTC never interpolate a different load
        # or commit a constitutive trial; a rejected attempt discards them.
        if loading is None:
            prescribed_strain, prescribed_stress = target, None
        else:
            target_strain, target_stress = conditions(target)
            prescribed_strain = xp.asarray(target_strain, dtype=dtype).reshape(
                (6, 1, 1, 1)
            )
            prescribed_stress = (
                xp.asarray(target_stress, dtype=dtype).reshape((6, 1, 1, 1))
                if np.any(target_stress)
                else None
            )

        def evaluate(value):
            return equilibrium(value, prescribed_stress)

        quadratic, first = None, None
        if guess is not None:
            e = guess
        elif (
            p.get("predictor", True)
            and previous_delta is not None
            and delta * previous_step > 0
        ):
            # Extrapolate the strain path linearly and, when the last three
            # states go the same way, quadratically (Lagrange, any spacing).
            e = strain + (delta / previous_step) * previous_delta
            if older_delta is not None and older_step * previous_step > 0:
                curvature = (
                    delta * (delta + previous_step) / (older_step + previous_step)
                )
                quadratic = e + (curvature / previous_step) * previous_delta
                quadratic -= (curvature / older_step) * older_delta
        else:
            e = strain.copy()
            if loading is None:
                e[axis] += delta
        e = admissible(e, prescribed_strain)
        if quadratic is not None:
            # The quadratic guess wins once the plastic zone stops spreading
            # and overshoots while it still changes: start Newton from the
            # smaller residual (one extra constitutive evaluation per step).
            first = evaluate(e)
            quadratic = admissible(quadratic, prescribed_strain)
            candidate = evaluate(quadratic)
            if candidate[2] < first[2]:
                e, first = quadratic, candidate
            del quadratic, candidate
        cg_total = 0
        previous_error, previous_eta = None, None
        linear_solves = []
        dt, retries = ptc_initial, 0
        explicit_accepted, explicit_reductions = 0, 0
        if use_ptc and explicit_steps:
            # Optional forward-Euler relaxation for the fictitious evolution
            # E de/dtau = -P sigma. Conservative spectral cap, then residual
            # backtracking; never update the physical plastic state here.
            explicit_dt = min(dt, 0.8 * material.E / elastic_bound)
            for _ in range(explicit_steps):
                response, residual, error = evaluate(e)
                if error <= rtol:
                    break
                for backtrack in range(12):
                    trial_dt = explicit_dt * 0.5**backtrack
                    strain_residual = (
                        residual
                        if coordinates is None
                        else coordinates.strain(residual)
                    )
                    candidate = e - (trial_dt / material.E) * strain_residual
                    del strain_residual
                    candidate = admissible(candidate, prescribed_strain)
                    trial_response, trial_residual, trial_error = evaluate(candidate)
                    accepted = np.isfinite(trial_error) and (
                        trial_error < error or trial_error <= rtol
                    )
                    del trial_response, trial_residual
                    if accepted:
                        e = candidate
                        explicit_accepted += 1
                        explicit_reductions += backtrack
                        explicit_dt = min(
                            trial_dt * 1.5, 0.8 * material.E / elastic_bound
                        )
                        break
                else:
                    break  # Switch to implicit PTC; no false convergence.
        limit = max_ptc if use_ptc else max_newton
        # Only reuse an accepted evaluation within this attempt, against the
        # same confirmed plastic state. Never carry it across commits/rejects.
        accepted_equilibrium = None if use_ptc and explicit_steps else first
        del first
        for iteration in range(limit + 1):
            if accepted_equilibrium is None:
                response, residual, error = evaluate(e)
            else:
                response, residual, error = accepted_equilibrium
                accepted_equilibrium = None
            if not np.isfinite(error):
                raise NonlinearConvergenceError("Nonfinite Newton residual")
            if error <= rtol:
                diagnostics = dict(
                    globalization="ptc" if use_ptc else "newton",
                    ptc_iterations=iteration if use_ptc else 0,
                    ptc_retries=retries,
                    ptc_dt_final=dt if use_ptc else 0.0,
                    explicit_steps=explicit_accepted,
                    linear_solves=linear_solves,
                    explicit_backtracks=explicit_reductions,
                )
                return e, response, error, iteration, cg_total, diagnostics
            if iteration == limit:
                break
            inner = min(
                0.01, max(1e-5 if working == "float32" else 1e-10, 0.1 * np.sqrt(error))
            )
            if inner_forcing in ("ew", "ew_target") and not use_ptc:
                inner = _ew_tolerance(
                    error,
                    previous_error,
                    previous_eta,
                    1e-5 if working == "float32" else 1e-10,
                    rtol if inner_forcing == "ew_target" else None,
                )
            for retry in range(9 if use_ptc else 1):
                # PTC: (P C_alg P + K0/dt) de = -P sigma, K0 the reference medium.
                # This mass term regularizes the iteration, NOT the constitutive
                # law or the equilibrium acceptance test above.
                shift = material.E / dt if use_ptc else 0.0
                try:
                    correction, count = solve_tangent(response, residual, inner, shift)
                    cg_total += count
                    linear_solves.append(
                        dict(
                            residual_before=error,
                            relative_tolerance=inner,
                            cg_iterations=count,
                            shift=shift,
                        )
                    )
                    if pressure_dependent:
                        linear_solves[-1].update(
                            method=linear_solver, krylov_iterations=count
                        )
                    if coordinates is None:
                        correction = projector(correction)
                    accepted = False
                    for backtrack in range(12):
                        candidate = e + (0.5**backtrack) * correction
                        trial_response, trial_residual, trial_error = evaluate(
                            candidate
                        )
                        if (
                            trial_error <= (1 - 1e-4 * 0.5**backtrack) * error
                            or trial_error <= rtol
                        ):
                            e, accepted = candidate, True
                            if reuse_trial_response:
                                accepted_equilibrium = (
                                    trial_response,
                                    trial_residual,
                                    trial_error,
                                )
                            del trial_response, trial_residual
                            break
                        del trial_response, trial_residual
                    if not accepted:
                        raise NonlinearConvergenceError(
                            "Equilibrium line search failed"
                        )
                except NonlinearConvergenceError:
                    if not use_ptc or retry == 8 or dt * 0.25 < ptc_min:
                        raise
                    dt *= 0.25
                    retries += 1
                    continue
                if use_ptc:
                    # Switched evolution relaxation; bounded growth avoids
                    # removing stabilization abruptly after a lucky iteration.
                    dt = min(
                        ptc_max,
                        dt * min(10.0, max(1.0, error / max(trial_error, rtol))),
                    )
                break
            previous_error, previous_eta = error, inner
        raise NonlinearConvergenceError(
            "PTC iteration limit" if use_ptc else "Newton iteration limit"
        )

    def attempt(target, use_ptc=False, guess=None):
        if damage is None:
            return mechanical_attempt(target, use_ptc, guess)
        damage.begin()
        total_cg, total_newton, total_nonlocal = 0, 0, 0
        linear_solves = []
        try:
            for coupled in range(1, damage_parameters.max_iterations + 1):
                e, response, error, it, cg, diagnostics = mechanical_attempt(
                    target, use_ptc, guess
                )
                total_cg += cg
                total_newton += it
                linear_solves.extend(diagnostics["linear_solves"])
                with timers.measure("damage_nonlocal_total"):
                    proposed, nonlocal_info = damage.propose(e)
                total_nonlocal += nonlocal_info["iterations"]
                if nonlocal_info["damage_gap"] <= damage_parameters.rtol:
                    # Stress/tangent and d remain paired. Updating d here
                    # would invalidate the verified mechanical equilibrium.
                    diagnostics.update(
                        damage_iterations=coupled,
                        nonlocal_iterations=total_nonlocal,
                        nonlocal_residual=nonlocal_info["relative_residual"],
                        damage_gap=nonlocal_info["damage_gap"],
                        linear_solves=linear_solves,
                    )
                    return e, response, error, total_newton, total_cg, diagnostics
                damage.set_trial(proposed)
                guess = e
                del response
            raise NonlinearConvergenceError("Damage coupling iteration limit")
        except NonlinearConvergenceError:
            damage.begin()
            raise

    def record(e, response, error, it, cg, elapsed, target=None):
        with timers.measure("diagnostics_and_transfer"):
            if diagnostic_engine is None:
                mean_e, mean_s = means(e), means(response.stress)
            else:
                detail = diagnostic_engine.evaluate(e, response, state, viscous)
                mean_e, mean_s = detail["mean_strain"], detail["mean_stress"]
            if loading is None:
                macro_error = np.linalg.norm(mean_s[~control]) / material.yield_stress
            else:
                prescribed_strain, prescribed_stress = conditions(
                    current if target is None else target
                )
                macro_error = (
                    np.linalg.norm((mean_s - prescribed_stress)[~control])
                    / material.yield_stress
                )
                strain_error = (
                    np.linalg.norm((mean_e - prescribed_strain)[control]) / pseudo_scale
                )
            if diagnostic_engine is not None:
                yield_error, dissipation = (
                    detail["yield_violation"],
                    detail["dissipation"],
                )
                hm = abs(detail["stress_work"] - float(mean_s @ mean_e))
                hm /= material.yield_stress * max(detail["strain_norm"], 1e-12)
                equivalent_mean, plastic_fraction = (
                    detail["equivalent_mean"],
                    detail["plastic_fraction"],
                )
                plastic_volume = detail["plastic_volumetric_mean"]
            else:
                plastic_volume_increment = None
                if model == "paraboloidal":
                    # Plastic work and the volume change of the potential's flow.
                    if gpu:
                        violation, irreversible, plastic_volume_increment = gpu.audit(
                            response, state
                        )
                    else:
                        violation, irreversible = paraboloidal.audit_cpu(
                            response, state, material, ids
                        )
                        plastic_volume_increment = xp.sum(
                            paraboloidal.plastic_increment(
                                response.stress, response.dp, material
                            )[:3],
                            axis=0,
                        )
                elif pressure_dependent and gpu:
                    violation, irreversible = gpu.audit(response, state)
                else:
                    dev = response.stress.copy()
                    mean = xp.sum(dev[:3], axis=0) / 3
                    dev[:3] -= mean
                    if material.kinematic_hardening:
                        dev -= material.backstress(
                            state.plastic_strain
                            + np.sqrt(1.5) * response.direction * response.dp,
                            xp,
                        )
                    q = xp.sqrt(1.5 * xp.sum(dev * dev, axis=0))
                    if pressure_dependent or not (
                        material.nonlinear_hardening or material.viscosity
                    ):
                        f = (
                            q
                            - material.yield_stress
                            - material.hardening * (state.equivalent + response.dp)
                        )
                    else:
                        # Perzyna: q - R equals the overstress viscous*dp on the plastic set.
                        f = (
                            q
                            - material.radius(state.equivalent + response.dp, xp)
                            - viscous * response.dp
                        )
                    if pressure_dependent:
                        f += material.friction * mean
                        irreversible = response.dp * (
                            material.yield_stress
                            + (material.beta - material.friction) * mean
                        )
                    violation = xp.where(ids == 0, xp.maximum(f, 0), 0)
                yield_error = float(xp.max(violation)) / material.yield_stress
                hm = abs(dot(response.stress, e) / phase.size - float(mean_s @ mean_e))
                hm /= material.yield_stress * max(norm(e), 1e-12)
                # Hardening is stored energy: J2 dissipates sigma_y dp plus the overstress work.
                dissipation = (
                    float(xp.mean(irreversible, dtype=xp.float64))
                    if pressure_dependent
                    else material.yield_stress
                    * float(xp.mean(response.dp, dtype=xp.float64))
                )
                if viscous:
                    dissipation += viscous * float(
                        xp.mean(response.dp * response.dp, dtype=xp.float64)
                    )
                if pressure_dependent:
                    # Positive mean alone could hide a locally inadmissible flow.
                    # Normalize by the local multiplier; ignore only roundoff.
                    work_scale = (
                        material.yield_stress
                        if damage is None
                        else material.yield_stress * damage.degradation
                    )
                    bad_work = irreversible < -rtol * work_scale * response.dp
                    if bool(xp.any(bad_work)):
                        raise NonlinearConvergenceError(
                            "Negative local plastic dissipation"
                        )
                equivalent_mean = float(
                    xp.mean(state.equivalent + response.dp, dtype=xp.float64)
                )
                plastic_fraction = float(
                    xp.mean((state.equivalent + response.dp) > 1e-10)
                )
                if plastic_volume_increment is not None:
                    plastic_volume = float(
                        xp.mean(
                            xp.sum(state.plastic_strain[:3], axis=0)
                            + plastic_volume_increment,
                            dtype=xp.float64,
                        )
                    )
                elif pressure_dependent:
                    plastic_volume = float(
                        xp.mean(
                            xp.sum(state.plastic_strain[:3], axis=0)
                            + np.sqrt(1.5)
                            * xp.sum(response.direction[:3], axis=0)
                            * response.dp,
                            dtype=xp.float64,
                        )
                    )
            if macro_error > rtol or yield_error > max(
                rtol, 5e-6 if dtype == np.float32 else 1e-12
            ):
                raise NonlinearConvergenceError(
                    f"Macro stress or local yield audit failed (macro {macro_error:.2e}, yield {yield_error:.2e})"
                )
            if (
                not np.isfinite([macro_error, yield_error, dissipation, hm]).all()
                or dissipation < 0
                or hm > 5 * rtol
            ):
                raise NonlinearConvergenceError(
                    f"Invalid energy diagnostic (plastic dissipation {dissipation:.6e}, "
                    f"Hill-Mandel {hm:.6e}, limit {5 * rtol:.6e})"
                )
            if loading is not None and (
                not np.isfinite(strain_error) or strain_error > rtol
            ):
                raise NonlinearConvergenceError("Prescribed mean strain audit failed")
            row = dict(
                strain=mean_e.tolist(),
                stress=mean_s.tolist(),
                residual=error,
                macro_residual=macro_error,
                yield_violation=yield_error,
                hill_mandel_error=hm,
                plastic_dissipation=dissipation,
                equivalent_plastic_mean=equivalent_mean,
                plastic_fraction=plastic_fraction,
                newton_iterations=it,
                cg_iterations=cg,
                seconds=elapsed,
                globalization="initial",
                ptc_iterations=0,
                ptc_retries=0,
                ptc_dt_final=0.0,
            )
            if loading is not None:
                row["macro_strain_residual"] = strain_error
            if pressure_dependent:
                row.update(
                    linear_solver=linear_solver,
                    krylov_iterations=cg,
                    hardening_variable_mean=row["equivalent_plastic_mean"],
                    plastic_volumetric_mean=plastic_volume,
                )
            if damage is not None:
                damage_work = damage.damage_work(e, state, response)
                work = float(xp.mean(damage_work, dtype=xp.float64))
                if (
                    not np.isfinite(work)
                    or work < 0
                    or bool(xp.any(~xp.isfinite(damage_work)))
                    or bool(xp.any(damage_work < 0))
                ):
                    raise NonlinearConvergenceError("Invalid damage dissipation")
                matrix_count = int(np.count_nonzero(phase == 0))
                damage_sum = float(xp.sum(damage.trial, dtype=xp.float64))
                if damage.interphase is not None:
                    weak = damage.interphase != 0
                    row["damage_interphase_mean"] = float(
                        xp.sum(damage.trial * weak, dtype=xp.float64)
                        / max(int(xp.count_nonzero(weak)), 1)
                    )
                    row["interphase_fraction"] = (
                        float(xp.count_nonzero(weak)) / phase.size
                    )
                row.update(
                    damage_mean=damage_sum / phase.size,
                    damage_matrix_mean=damage_sum / max(matrix_count, 1),
                    damage_max=float(xp.max(damage.trial)),
                    damage_dissipation=work,
                    damage_gap=0.0,
                    damage_iterations=0,
                    nonlocal_iterations=0,
                    nonlocal_residual=0.0,
                )
            return row

    if not history:
        response, _, error = equilibrium(strain)
        history.append(record(strain, response, error, 0, 0, 0.0))
        stamp_targets(history[-1], current)
        annotate_history()
        del response
    checkpoint = p.get("checkpoint_path")
    final_stress = None
    for segment, endpoint in enumerate(path[1:]):
        increment_limit = float(segment_inc[segment])
        minimum_limit = float(segment_minimum[segment])
        step = increment_limit
        if loading is not None:
            # Parameter always increases, even when tensors unload or rotate.
            # A knot changes the segment slope: do not extrapolate the old one.
            previous_delta, previous_step = None, 0.0
            older_delta, older_step = None, 0.0
        direction = float(np.sign(endpoint - current))
        # The response is linear until some voxel yields. From a virgin state
        # or after a load reversal, reach the first yield (or the endpoint) in
        # one exact elastic step instead of many small increments.
        jump, extra_cg, extra_seconds = None, 0, 0.0
        if (
            elastic_segments
            and direction
            and (previous_step == 0 or previous_step * direction < 0)
        ):
            timers.sync()
            jump_started = time.perf_counter()
            try:
                reach, rate, extra_cg = elastic_reach(direction)
            except NonlinearConvergenceError:
                reach, rate = 0.0, None  # ordinary increments remain available
            extra_seconds = time.perf_counter() - jump_started
            # The elastic solve costs about one plastic increment.
            if reach >= inc:
                jump = (min(reach, abs(endpoint - current)), rate)
            del rate
        while (
            current < endpoint
            if loading is not None
            else abs(endpoint - current) > 1e-12
        ):
            guess = None
            if jump is not None:
                length, rate = jump
                jump = None
                target = current + direction * length
                guess = strain + length * rate
                del rate
            else:
                distance = abs(endpoint - current)
                rounding = (
                    32
                    * np.finfo(float).eps
                    * max(abs(endpoint), abs(current), distance)
                    if loading is not None
                    else 0.0
                )
                target = (
                    float(endpoint)
                    if step >= distance - rounding
                    else current + direction * step
                )
                if target == current:
                    raise ValueError(
                        "Loading parameter increment is below floating-point resolution"
                    )
            timers.sync()
            step_started = time.perf_counter()
            try:
                fallback_reason = None
                try:
                    e, response, error, it, cg, diagnostics = attempt(
                        target, globalization == "ptc", guess
                    )
                except NonlinearConvergenceError as exc:
                    if globalization != "auto":
                        raise
                    fallback_reason = str(exc)
                # Leave the except block before PTC: its traceback can retain
                # several full GPU fields belonging to the failed Newton solve.
                if fallback_reason is not None:
                    e, response, error, it, cg, diagnostics = attempt(target, True)
                    diagnostics["ptc_fallback_reason"] = fallback_reason
                timers.sync()
                row = record(
                    e,
                    response,
                    error,
                    it,
                    cg + extra_cg,
                    time.perf_counter() - step_started + extra_seconds,
                    target,
                )
                row.update(diagnostics)
                stamp_targets(row, target)
                row["elastic_step"] = guess is not None
                extra_cg, extra_seconds = 0, 0.0  # the elastic solve is counted once
            except NonlinearConvergenceError as exc:
                rejected.append(
                    dict(from_strain=current, target=float(target), reason=str(exc))
                )
                if loading is not None:
                    rejected[-1].update(
                        from_parameter=current, target_parameter=float(target)
                    )
                if p.get("on_reject"):
                    p["on_reject"](dict(rejected[-1]))
                step = abs(target - current) / 2
                if step < minimum_limit:
                    if checkpoint:
                        save_checkpoint(
                            checkpoint,
                            strain,
                            state,
                            fingerprint,
                            history,
                            loading,
                            damage_state=None if damage is None else damage.confirmed,
                        )
                    raise NonlinearConvergenceError(
                        f"{exc}; minimum increment reached at {current}; {len(history) - 1} confirmed steps"
                    ) from exc
                continue
            older_delta, older_step = previous_delta, previous_step
            previous_delta, previous_step = e - strain, float(target - current)
            if pressure_dependent and gpu:
                state = gpu.commit(state, response)
            elif model == "paraboloidal":
                state = paraboloidal.commit_cpu(state, response, material)
            else:
                state = commit_state(state, response)
            if damage is not None:
                damage.accept()
            strain, current = e, float(target)
            history.append(row)
            annotate_history()
            if p.get("return_fields", False):
                # The accepted stress; re-evaluating a viscous state would flow again.
                final_stress = response.stress
            del response
            if p.get("on_step"):
                p["on_step"](copy.deepcopy(row))
            every = int(p.get("checkpoint_every", 0))
            if checkpoint and every > 0 and (len(history) - 1) % every == 0:
                save_checkpoint(
                    checkpoint,
                    strain,
                    state,
                    fingerprint,
                    history,
                    loading,
                    damage_state=None if damage is None else damage.confirmed,
                )
            # Grow only back to the requested resolution, never skip the path.
            if it <= 6:
                step = min(increment_limit, step * 1.5)
    if checkpoint:
        with timers.measure("checkpoint_transfer_io"):
            save_checkpoint(
                checkpoint,
                strain,
                state,
                fingerprint,
                history,
                loading,
                damage_state=None if damage is None else damage.confirmed,
            )
    timers.sync()
    result = dict(
        history=history,
        rejected_steps=rejected,
        configuration=config,
        loading=loading_component(axis),
        fingerprint=fingerprint,
        backend=backend,
        dtype=dtype.name,
        rtol=rtol,
        source_sha256=source_hashes,
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            cupy=cp.__version__ if cp is not None else None,
        ),
        predictor=bool(p.get("predictor", True)),
        integrator=integrator,
        model=model,
        linear_solver=linear_solver,
        gmres_restart=gmres_restart if nonsymmetric else None,
        krylov_fallbacks=(
            krylov_fallbacks + (tangent_pipeline.fallbacks if tangent_pipeline else 0)
            if linear_solver == "bicgstab"
            else None
        ),
        hardening_variable=dict(
            j2="equivalent_von_Mises_plastic_strain",
            drucker_prager="accumulated_plastic_multiplier",
            paraboloidal="equivalent_plastic_strain_nu_p",
        )[model],
        cg_projection=cg_projection,
        fused_cg=fused_cg,
        coordinate_pcg=coordinate_pcg_used,
        coordinate_pcg_requested=coordinate_pcg,
        working=working,
        law_arithmetic=law_arithmetic,
        reuse_trial_response=reuse_trial_response,
        diagnostic_reduction="fused" if diagnostic_engine is not None else "reference",
        diagnostic_calls=0 if diagnostic_engine is None else diagnostic_engine.calls,
        diagnostic_workspace_bytes=0
        if diagnostic_engine is None
        else diagnostic_engine.workspace_bytes,
        preconditioner=preconditioner,
        formulation=formulation,
        spectral_fusion=projector.spectral_fusion,
        inner_forcing=inner_forcing,
        initial_increment=inc,
        min_increment=minimum,
        path=path.tolist(),
        pseudo_time_strain_scale=pseudo_scale,
        load_rate=load_rate,
        globalization=globalization,
        ptc=dict(
            initial_dt=ptc_initial,
            min_dt=ptc_min,
            max_dt=ptc_max,
            max_iterations=max_ptc,
            explicit_steps=explicit_steps,
        ),
        wall_seconds=time.perf_counter() - started,
        timings=timers.seconds,
        timing_calls=timers.calls,
        timings_synchronized=bool(timers.enabled),
        sampled_device_peak_bytes=timers.sampled_device_peak,
        gpu_pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes())
        if xp is not np
        else 0,
        stress_scale_GPa=material.yield_stress,
    )
    if loading is not None:
        result["loading"].update(
            kind="piecewise_linear_mixed",
            strain_control=control.tolist(),
            other_mean_stresses="prescribed on stress-controlled components",
            component_export="projection of the multiaxial history",
        )
        result.update(
            loading_path=loading.descriptor(),
            elastic_segments=False,
            pseudo_time_stress_scale_GPa=pseudo_stress_scale,
            parameter_increment_limits=dict(
                initial=segment_inc.tolist(), minimum=segment_minimum.tolist()
            ),
        )
    if damage is not None:
        result["damage"] = dict(
            damage_parameters.configuration(),
            law=f"effective_stress_{'paraboloidal' if model == 'paraboloidal' else 'DP'}_exponential_nonlocal_total_strain",
            calibrated=False,
            fracture_energy_calibrated=False,
            interface=(
                "damaging_interphase"
                if damage_parameters.interphase_thickness
                else "perfectly_bonded"
            ),
            nonlocal_dtype="float64",
            damage_history_dtype="float64",
            mechanical_dtype=dtype.name,
        )
    if p.get("return_fields", False):
        if final_stress is None:
            final_stress = update(strain).stress
        result["fields"] = dict(strain=_host(strain), stress=_host(final_stress))
        result["state"] = PlasticState(
            _host(state.plastic_strain), _host(state.equivalent)
        )
        if damage is not None:
            result["fields"].update(
                damage=_host(damage.confirmed.value),
                damage_history=_host(damage.confirmed.history),
            )
    if p.get("output_dir"):
        export_curve(result, p["output_dir"])
    return result
