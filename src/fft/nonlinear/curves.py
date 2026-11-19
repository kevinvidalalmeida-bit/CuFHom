"""Physical normal/shear curves from confirmed Mandel histories.

The solver stores e_M=sqrt(2)*epsilon_ij=gamma_ij/sqrt(2) and
s_M=sqrt(2)*tau_ij for shear. Keep those histories intact for restart/audits.
"""

import csv
import json
from pathlib import Path

import numpy as np

MANDEL_COMPONENTS = ("11", "22", "33", "23", "13", "12")
CONSTITUTIVE_COMPONENTS = ("11", "22", "33", "12", "13", "23")


def loading_component(axis):
    """Describe the physical quantities and conversion for one Mandel axis."""
    if axis not in range(6):
        raise ValueError("axis must be a Mandel index in 0..5")
    component = MANDEL_COMPONENTS[axis]
    shear = axis >= 3
    return dict(
        component=component,
        axis=axis,
        strain_quantity=("gamma_" if shear else "epsilon_") + component,
        stress_quantity=("tau_" if shear else "sigma_") + component,
        strain_from_mandel_scale=float(np.sqrt(2.0) if shear else 1.0),
        stress_from_mandel_scale=float(1 / np.sqrt(2.0) if shear else 1.0),
        other_mean_stresses="zero",
        strain_units="dimensionless",
        stress_units="GPa",
    )


def constitutive_values(result):
    """Return physical mean epsilon/gamma and sigma/tau (GPa), in path order."""
    loading = loading_component(result["configuration"]["axis"])
    axis = loading["axis"]
    strain = np.asarray([r["strain"][axis] for r in result["history"]])
    stress = np.asarray([r["stress"][axis] for r in result["history"]])
    return (
        strain * loading["strain_from_mandel_scale"],
        stress * loading["stress_from_mandel_scale"],
    )


def export_constitutive_curve(result, directory):
    """Write an explicitly named engineering-strain/physical-stress CSV."""
    loading = loading_component(result["configuration"]["axis"])
    strain, stress = constitutive_values(result)
    with (Path(directory) / "constitutive.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                loading["strain_quantity"],
                loading["stress_quantity"] + "_GPa",
                loading["stress_quantity"] + "_MPa",
                "pseudo_time",
            ]
        )
        for step, (e, s, row) in enumerate(zip(strain, stress, result["history"])):
            writer.writerow([step, e, s, s * 1000, row["pseudo_time"]])


def plot_constitutive_curve(ax, result):
    """Plot branches without sorting or joining independent experiments."""
    loading = loading_component(result["configuration"]["axis"])
    strain, stress = constitutive_values(result)
    directions = np.sign(np.diff(strain))
    boundaries = (
        [0]
        + (np.flatnonzero(np.diff(directions) != 0) + 1).tolist()
        + [len(strain) - 1]
    )
    labels_used = set()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        increasing = strain[end] >= strain[start]
        label = "Carga / recarga" if increasing else "Descarga"
        ax.plot(
            strain[start : end + 1] * 100,
            stress[start : end + 1] * 1000,
            ".-",
            markersize=3,
            color="tab:blue" if increasing else "tab:orange",
            label=label if label not in labels_used else None,
        )
        labels_used.add(label)
    component = loading["component"]
    strain_symbol, stress_symbol = (
        ("gamma", "tau") if loading["axis"] >= 3 else ("varepsilon", "sigma")
    )
    ax.set(
        xlabel=rf"$\{strain_symbol}_{{{component}}}$ [%]",
        ylabel=rf"$\{stress_symbol}_{{{component}}}$ [MPa]",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()


def export_constitutive_curves(results, directory):
    """Summarize six independent runs; their full metadata lives in subfolders."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # A missing or mislabelled experiment must not produce a complete suite.
    for component in CONSTITUTIVE_COMPONENTS:
        if (
            loading_component(results[component]["configuration"]["axis"])["component"]
            != component
        ):
            raise ValueError("Mismatched constitutive component: " + component)
    with (directory / "constitutive_curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "component",
                "step",
                "strain_quantity",
                "strain",
                "stress_quantity",
                "stress_GPa",
                "stress_MPa",
                "pseudo_time",
            ]
        )
        for component in CONSTITUTIVE_COMPONENTS:
            result = results[component]
            loading = loading_component(result["configuration"]["axis"])
            strain, stress = constitutive_values(result)
            for step, (e, s, row) in enumerate(zip(strain, stress, result["history"])):
                writer.writerow(
                    [
                        component,
                        step,
                        loading["strain_quantity"],
                        e,
                        loading["stress_quantity"],
                        s,
                        s * 1000,
                        row["pseudo_time"],
                    ]
                )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for ax, component in zip(axes.flat, CONSTITUTIVE_COMPONENTS):
        plot_constitutive_curve(ax, results[component])
    models = {r.get("model", "j2") for r in results.values()}
    name = (
        "Drucker–Prager"
        if models == {"drucker_prager"}
        else "J2"
        if models == {"j2"}
        else "plasticidad"
    )
    qualifier = (
        "Daño de matriz regularizado, sin calibrar"
        if any(r.get("damage") for r in results.values())
        else "Parámetros demostrativos, sin daño ni rotura"
    )
    fig.suptitle(f"FFT {name} — seis ensayos independientes\n{qualifier}")
    fig.savefig(directory / "constitutive_curves.png", dpi=180)
    fig.savefig(directory / "constitutive_curves.pdf")
    plt.close(fig)
    metadata = dict(
        independent_initial_states=True,
        other_mean_stresses="zero",
        strain_units="dimensionless (epsilon for normal, engineering gamma for shear)",
        stress_units="GPa; MPa columns = 1000 times GPa",
        components={
            c: dict(
                loading=loading_component(results[c]["configuration"]["axis"]),
                metadata=f"{c}/metadata.json",
                curve=f"{c}/constitutive.csv",
                points=len(results[c]["history"]),
            )
            for c in CONSTITUTIVE_COMPONENTS
        },
    )
    if any(r.get("damage") for r in results.values()):
        metadata.update(
            damage_model="implicit_gradient",
            calibrated=False,
            interface="perfectly_bonded",
        )
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False)
    )
