"""Tabulate new campaign records and quantify reference-operator sensitivity."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path, rows):
    """Write a deterministic CSV without silently replacing previous results."""
    if not rows:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(directory):
    """Extract first-call evidence and warm-call distributions from each point."""
    rows = []
    for source in sorted(directory.glob("point_*.json")):
        record = json.loads(source.read_text())
        point = record["point"]
        calls = record.get("calls", [])
        first = next((item for item in calls if item["state"] == "first"), {})
        warm = [item["time_s"] for item in calls if item["state"] == "warm"]
        audits = first.get("audits", [])
        rows.append(
            dict(
                source=source.name,
                status=record["status"],
                case=point["case"],
                discretization=point["discretization"],
                precision=point["precision"],
                n=point["n"],
                eta=point["eta"],
                contrast=point["contrast"],
                nu0=point["nu0"],
                session=point["session"],
                mean_updates=first.get("mean_updates"),
                iterations=json.dumps(first.get("iterations")),
                first_s=first.get("time_s"),
                warm_median_s=statistics.median(warm) if warm else None,
                warm_min_s=min(warm) if warm else None,
                warm_max_s=max(warm) if warm else None,
                fresh_relative=json.dumps([item["relative"] for item in audits]),
                fresh_max=max((item["relative"] or 0 for item in audits), default=None),
                direct_error=first.get("error_direct_fro"),
                corrected_error=first.get("error_corrected_fro"),
                symmetrized_error=first.get("error_symmetrized_fro"),
                failure=record.get("error"),
            )
        )
    return rows


def sensitivity(rows):
    """Normalize each contrast curve by its own minimum, excluding chi=1."""
    curves = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["status"] != "accepted" or row["nu0"] is None:
            continue
        key = tuple(
            row[name]
            for name in ("case", "discretization", "precision", "n", "eta", "contrast")
        )
        curves[key][row["nu0"]].append(row["mean_updates"])
    normalized, diagnostics = [], []
    for key, values in curves.items():
        curve = {nu: statistics.median(items) for nu, items in values.items()}
        minimum = min(curve.values())
        # Zero-work curves, including chi=1, cannot be divided by their minimum.
        denominator = None if key[-1] == 1 or minimum == 0 else minimum
        identity = dict(
            zip(
                ("case", "discretization", "precision", "n", "eta", "contrast"),
                key,
                strict=True,
            )
        )
        for nu, mean in sorted(curve.items()):
            normalized.append(
                identity
                | {
                    "nu0": nu,
                    "mean_updates": mean,
                    "normalized_by_minimum": mean / denominator
                    if denominator
                    else None,
                }
            )
        diagnostics.append(
            identity
            | {
                "minimum_updates": minimum,
                "nu0_at_sampled_minimum": json.dumps(
                    sorted(nu for nu, mean in curve.items() if mean == minimum)
                ),
                "penalty_nu048_over_minimum": curve.get(0.48) / denominator
                if denominator and 0.48 in curve
                else None,
                "penalty_nu030_over_minimum": curve.get(0.3) / denominator
                if denominator and 0.3 in curve
                else None,
                "scope": "sampled nu0 values only; session medians; chi=1 excluded from normalization",
            }
        )
    return normalized, diagnostics


def compare_curves(normalized):
    """Compare sampled reciprocal contrasts and FP64/mixed curve shapes."""
    curves = defaultdict(dict)
    for row in normalized:
        key = tuple(
            row[name]
            for name in ("case", "discretization", "precision", "n", "eta", "contrast")
        )
        curves[key][row["nu0"]] = row
    comparisons = []
    for key, curve in curves.items():
        candidates = []
        if key[2] == "fp64":
            candidates.append(("mixed_vs_fp64", key[:2] + ("mixed",) + key[3:]))
        if key[-1] is not None and 0 < key[-1] < 1:
            inverse = next(
                (
                    other
                    for other in curves
                    if other[:-1] == key[:-1]
                    and other[-1] is not None
                    and abs(other[-1] * key[-1] - 1) < 1e-12
                ),
                None,
            )
            if inverse:
                candidates.append(("reciprocal_contrasts", inverse))
        for purpose, other in candidates:
            if other not in curves:
                continue
            shared = sorted(set(curve) & set(curves[other]))
            pairs = [(curve[nu], curves[other][nu]) for nu in shared]
            relative = [
                abs(b["mean_updates"] - a["mean_updates"]) / a["mean_updates"]
                for a, b in pairs
                if a["mean_updates"] > 0
            ]
            shapes = [
                abs(b["normalized_by_minimum"] - a["normalized_by_minimum"])
                for a, b in pairs
                if a["normalized_by_minimum"] is not None
                and b["normalized_by_minimum"] is not None
            ]
            comparisons.append(
                {
                    "comparison": purpose,
                    "case": key[0],
                    "discretization": key[1],
                    "n": key[3],
                    "eta": key[4],
                    "precision_a": key[2],
                    "contrast_a": key[-1],
                    "precision_b": other[2],
                    "contrast_b": other[-1],
                    "shared_nu0": json.dumps(shared),
                    "max_absolute_relative_update_difference": max(
                        relative, default=None
                    ),
                    "max_absolute_normalized_shape_difference": max(
                        shapes, default=None
                    ),
                    "interpretation": "sampled paired differences; no claim of identical shapes or contrast independence",
                }
            )
    return comparisons


def main():
    """Export CSVs without replacing the published paper or previous summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = collect(args.directory)
    if not rows:
        parser.error("No point_*.json campaign records found")
    output = args.output or args.directory / "tables"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "points.csv", rows)
    normalized, diagnostics = sensitivity(rows)
    write_csv(output / "reference_curves.csv", normalized)
    write_csv(output / "reference_diagnostics.csv", diagnostics)
    write_csv(output / "reference_comparisons.csv", compare_curves(normalized))
    print(
        f"Saved {len(rows)} conditions and {len(diagnostics)} sensitivity curves in {output}"
    )


if __name__ == "__main__":
    main()
