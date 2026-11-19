"""Run nonlinear constitutive curves for a homogeneous, layered or supplied geometry.

Built-in material parameters are uncalibrated demonstrations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from cufhom.fft import solve_nonlinear_homogenization
from cufhom.fft.nonlinear.curves import (
    CONSTITUTIVE_COMPONENTS,
    MANDEL_COMPONENTS,
    export_constitutive_curves,
    loading_component,
)


def main():
    """Parse command-line arguments and run a nonlinear loading case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["homogeneous", "laminate", "geometry"], default="homogeneous"
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        help="geometry.npz or directory containing phase.npy and ori.npy",
    )
    parser.add_argument("--material", type=Path, help="Material JSON for case geometry")
    parser.add_argument(
        "--loading",
        type=Path,
        help="General mixed loading JSON: six Mandel strains/stresses and boolean strain_control",
    )
    parser.add_argument(
        "--loading-increment",
        type=float,
        default=0.05,
        help="Maximum increment of the general loading parameter (not physical time)",
    )
    parser.add_argument(
        "--damage",
        type=Path,
        help="Implicit-gradient damage parameters JSON; requires Drucker-Prager and CuPy",
    )
    parser.add_argument(
        "--lengths",
        type=float,
        nargs=3,
        help="Physical periodic cell lengths, in the same length unit as damage.length_scale",
    )
    parser.add_argument(
        "--model",
        choices=["j2", "drucker_prager", "paraboloidal"],
        default=None,
        help="Matrix model: j2 by default; drucker_prager (cone) or paraboloidal "
        "(Tschoegl/Melro polymer) for pressure dependence",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=None,
        help="Drucker-Prager pressure coefficient alpha (dimensionless, not an angle); overrides JSON",
    )
    parser.add_argument(
        "--dilatancy",
        type=float,
        default=None,
        help="Drucker-Prager flow coefficient beta (0 <= beta <= alpha); overrides JSON",
    )
    parser.add_argument(
        "--load-rate",
        type=float,
        default=None,
        help="Rate of the path parameter, 1/s; required by a viscoplastic J2 matrix (viscosity > 0)",
    )
    parser.add_argument(
        "--component",
        choices=[*CONSTITUTIVE_COMPONENTS, "all"],
        default="11",
        help="Normal or shear experiment; all runs six independent virgin states",
    )
    parser.add_argument("--backend", choices=["scipy", "cupy"], default="scipy")
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument(
        "--increment",
        type=float,
        default=0.00075,
        help="Increment of epsilon_ii or engineering gamma_ij (dimensionless)",
    )
    parser.add_argument(
        "--maximum",
        type=float,
        default=0.01,
        help="Final epsilon_ii or engineering gamma_ij; demo .01, not failure strain",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--globalization", choices=["auto", "newton", "ptc"], default="auto"
    )
    parser.add_argument(
        "--integrator",
        choices=["midpoint", "backward_euler"],
        default=None,
        help="Return map: midpoint by default for j2; backward_euler only for drucker_prager",
    )
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--restart", type=Path)
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Initial epsilon_ii or engineering gamma_ij, matching the checkpoint",
    )
    loading = parser.add_mutually_exclusive_group()
    loading.add_argument(
        "--unload",
        action="store_true",
        help="Append unloading to zero strain (optional verification)",
    )
    args = parser.parse_args()
    if args.material is not None and args.case != "geometry":
        parser.error("--material is only used with --case geometry")
    if args.loading and (args.component == "all" or args.unload or args.start != 0.0):
        parser.error(
            "--loading requires one display component; specify the complete path/restart in its JSON"
        )
    model = args.model or "j2"
    if model != "drucker_prager" and (
        args.friction is not None or args.dilatancy is not None
    ):
        parser.error("--friction and --dilatancy require --model drucker_prager")
    if model != "j2" and args.integrator == "midpoint":
        parser.error(
            f"{model} requires --integrator backward_euler; midpoint is only available for j2"
        )
    integrator = args.integrator or ("midpoint" if model == "j2" else "backward_euler")
    if args.component == "all" and (args.restart or args.start != 0.0):
        parser.error(
            "--component all requires independent virgin states; restart a single component instead"
        )
    components = (
        CONSTITUTIVE_COMPONENTS if args.component == "all" else (args.component,)
    )
    output = args.output or Path.cwd() / "runs" / (
        "nonlinear_"
        + args.case
        + ("" if model == "j2" else "_" + model)
        + "_"
        + args.backend
        + ("_cycle" if args.unload else "_loading")
        + ("" if args.component == "11" else "_" + args.component)
    )
    run_dirs = {
        c: output / c if args.component == "all" else output for c in components
    }
    if (output / "metadata.json").exists() or any(
        (d / "metadata.json").exists() for d in run_dirs.values()
    ):
        parser.error("Output already contains a run; choose another --output directory")
    damage = None
    general_loading = None
    if args.loading:
        try:
            general_loading = json.loads(args.loading.read_text())
        except (OSError, ValueError) as exc:
            parser.error(f"Invalid --loading JSON: {exc}")
    if args.case == "geometry":
        if args.geometry is None or args.material is None:
            parser.error(
                "case geometry requires explicit --geometry and --material; no external campaign is assumed"
            )
        try:
            mat = json.loads(args.material.read_text())
            matrix, fiber = dict(mat["matrix"]), dict(mat["fiber"])
            damage = mat.get("damage")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Invalid --material JSON: {exc}")
        if mat.get("model", model) != model:
            parser.error(
                "Material JSON model disagrees with --model; choose the matching material explicitly"
            )
        if model == "j2" and any(k in matrix for k in ("friction", "dilatancy")):
            parser.error(
                "J2 material must not contain friction or dilatancy; choose the matching --model and JSON"
            )
        if args.geometry.is_file():
            with np.load(args.geometry, allow_pickle=False) as data:
                phase, ori = data["phase"], data["ori"]
        else:
            phase = np.load(args.geometry / "phase.npy", mmap_mode="r")
            ori = np.load(args.geometry / "ori.npy", mmap_mode="r")
    else:
        phase = np.zeros((8, 8, 8), dtype=np.uint8)
        if args.case == "laminate":
            phase[3:5, :, :] = 1
        ori = np.zeros(phase.shape + (3,))
        ori[..., 0] = 1
        matrix = dict(E=3.0, nu=0.3, yield_stress=0.015, hardening=0.03)
        fiber = dict(E_L=70.0, E_T=8.0, nu_LT=0.25, nu_TT=0.3, G_LT=5.0)
        if model == "drucker_prager":
            matrix.update(
                friction=0.3, dilatancy=None
            )  # Demonstration only, uncalibrated.
        if model == "paraboloidal":
            matrix = dict(
                E=3.0,
                nu=0.3,
                plastic_poisson=0.32,
                tension_curve=[[0, 0.015]],
                compression_curve=[[0, 0.035]],
            )  # Demonstration only, uncalibrated.
    if args.damage:
        try:
            damage_json = json.loads(args.damage.read_text())
            damage = damage_json.get("damage", damage_json)
        except (OSError, ValueError, AttributeError) as exc:
            parser.error(f"Invalid --damage JSON: {exc}")
    if damage is not None and (model != "drucker_prager" or args.backend != "cupy"):
        parser.error(
            "Implicit-gradient damage requires --model drucker_prager --backend cupy"
        )
    if model == "drucker_prager":
        if args.friction is not None:
            matrix["friction"] = args.friction
        if args.dilatancy is not None:
            matrix["dilatancy"] = args.dilatancy
        alpha, beta = matrix.get("friction", 0.3), matrix.get("dilatancy")
        beta = alpha if beta is None else beta
        try:
            valid_flow = bool(
                np.isfinite(alpha) and np.isfinite(beta) and 0 <= beta <= alpha < 3
            )
        except (TypeError, ValueError):
            valid_flow = False
        if not valid_flow:
            parser.error(
                "Drucker-Prager requires finite 0 <= dilatancy <= friction < 3; null dilatancy means associated flow"
            )
    linear_solver = (
        "bicgstab"
        if model == "paraboloidal" or model == "drucker_prager" and beta != alpha
        else "cg"
    )
    parameters = dict(
        phase_array=phase,
        ori_array=ori,
        matrix=matrix,
        fiber=fiber,
        backend=args.backend,
        model=model,
        profile=args.profile,
        globalization=args.globalization,
        integrator=integrator,
        formulation="displacement",
        preconditioner="matrix",
    )
    if damage is not None:
        parameters["damage"] = damage
    if args.load_rate is not None:
        parameters["load_rate"] = args.load_rate
    if args.lengths is not None:
        parameters["lengths"] = args.lengths
    if general_loading is not None:
        parameters["loading"] = general_loading
    if args.dtype:
        parameters["dtype"] = args.dtype
    if args.restart:
        parameters["restart_from"] = args.restart
    results = {}
    for component in components:
        axis = MANDEL_COMPONENTS.index(component)
        loading = loading_component(axis)
        scale = loading["strain_from_mandel_scale"]
        count = [0]

        def progress(row):
            count[0] += 1
            details = {
                k: row[k]
                for k in (
                    "damage_mean",
                    "damage_max",
                    "damage_iterations",
                    "nonlocal_residual",
                    "target_parameter",
                )
                if k in row
            }
            print(
                json.dumps(
                    dict(
                        component=component,
                        model=model,
                        linear_solver=linear_solver,
                        step=count[0],
                        strain_quantity=loading["strain_quantity"],
                        strain=row["strain"][axis] * scale,
                        stress_quantity=loading["stress_quantity"],
                        stress_GPa=row["stress"][axis]
                        * loading["stress_from_mandel_scale"],
                        newton=row["newton_iterations"],
                        **{("cg" if model == "j2" else "krylov"): row["cg_iterations"]},
                        seconds=row["seconds"],
                        residual=row["residual"],
                        plastic_fraction=row["plastic_fraction"],
                        **details,
                    )
                ),
                flush=True,
            )

        def rejected(row):
            physical = dict(row)
            if general_loading is None:
                for key in ("from_strain", "target"):
                    physical[key] *= scale
            print(json.dumps(dict(component=component, rejected=physical)), flush=True)

        run = dict(
            parameters,
            axis=axis,
            path=[
                v / scale
                for v in [args.start, args.maximum] + ([0.0] if args.unload else [])
            ],
            initial_increment=args.increment / scale,
            pseudo_time_strain_scale=0.01 / scale,
            output_dir=run_dirs[component],
            on_step=progress,
            on_reject=rejected,
        )
        if general_loading is not None:
            run.pop("path")
            run["initial_increment"] = args.loading_increment
            run["pseudo_time_strain_scale"] = 0.01
        if args.checkpoint:
            run.update(
                checkpoint_path=run_dirs[component] / "checkpoint.npz",
                checkpoint_every=10,
            )
        result = solve_nonlinear_homogenization(run)
        results[component] = result
        print(
            json.dumps(
                dict(
                    component=component,
                    model=model,
                    linear_solver=result.get("linear_solver", linear_solver),
                    output=str(run_dirs[component]),
                    wall_seconds=result["wall_seconds"],
                    steps=len(result["history"]) - 1,
                    rejected=len(result["rejected_steps"]),
                    max_residual=max(r["residual"] for r in result["history"]),
                )
            ),
            flush=True,
        )
    if args.component == "all":
        export_constitutive_curves(results, output)
        print(
            json.dumps(
                dict(
                    output=str(output),
                    components=list(components),
                    wall_seconds=sum(r["wall_seconds"] for r in results.values()),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
