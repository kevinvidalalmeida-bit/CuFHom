"""Independent analytical and end-to-end checks of the runnable Hashin example."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from cufhom.fft import select_grid
from cufhom.fft.grids import is_fft_size

from benchmarks.paper.run import plot_convergence, run_convergence
from examples.hashin_convergence import hashin_bulk

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/hashin_convergence.py"


@pytest.mark.parametrize(
    "core,coating,shear,fraction",
    [(10, 1, 0.6, 0.14), (0.2, 4, 2, 0.35), (4, 4, 3, 0.6)],
)
def test_bulk_matches_radial_boundary_value_problem(core, coating, shear, fraction):
    # Independent radial solution: u_core=A*r, u_coating=B*r+D/r**2.
    # Enforce displacement and traction continuity and unit outer displacement.
    matrix = np.array(
        [
            [1, -1, -1 / fraction],
            [3 * core, -3 * coating, 4 * shear / fraction],
            [0, 1, 1],
        ],
        dtype=float,
    )
    _, b, d = np.linalg.solve(matrix, [0, 0, 1])
    bulk_from_outer_traction = coating * b - 4 * shear * d / 3
    assert hashin_bulk(core, coating, shear, fraction) == pytest.approx(
        bulk_from_outer_traction, rel=1e-13
    )


@pytest.mark.parametrize("fraction,expected", [(0, 1), (1, 10)])
def test_single_phase_limits(fraction, expected):
    assert hashin_bulk(10, 1, 0.6, fraction) == pytest.approx(expected)


def test_default_grids_have_no_prime_factors_above_13():
    for method in ("gani", "hex8"):
        for requested in (16, 24, 32, 48, 64, 96, 128, 192):
            size = select_grid(requested, discretization=method)
            assert size >= requested and size % 2 == (method == "gani")
            remaining = size
            divisor = 2
            while divisor * divisor <= remaining:
                while remaining % divisor == 0:
                    assert divisor <= 13
                    remaining //= divisor
                divisor += 1
            assert remaining <= 13


@pytest.mark.parametrize(
    "requested,policy,method,expected",
    [
        (17, "odd", "gani", 17),
        (17, "odd-13", "gani", 21),
        (33, "even", "hex8", 34),
        (33, "even-13", "hex8", 36),
        (32, "auto", "gani", 33),
        (32, "auto", "hex8", 32),
        (21, "odd-13", "hex8", 21),
    ],
)
def test_grid_policy_selects_the_smallest_allowed_size(
    requested, policy, method, expected
):
    assert select_grid(requested, policy, discretization=method) == expected


@pytest.mark.parametrize("invalid", [True, 0, -2, 3, 8.5, "16"])
def test_grid_policy_rejects_invalid_sizes(invalid):
    with pytest.raises(ValueError, match="integer >= 4"):
        select_grid(invalid)


def test_prime_check_handles_one_and_repeated_factors():
    assert all(is_fft_size(n) for n in (1, 8, 21, 26, 169))
    assert not any(is_fft_size(n) for n in (0, -1, 17, 34, 289))
    with pytest.raises(ValueError, match="Unknown grid policy"):
        select_grid(16, "typo")
    with pytest.raises(ValueError, match="discretization"):
        select_grid(16, discretization="typo")


def run_example(output, *arguments):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "numpy",
            "--precision",
            "fp64",
            "--grids",
            "8",
            "14",
            "--repeats",
            "1",
            "--output",
            str(output),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.integration
def test_cpu_run_keeps_calls_and_exports_exactly_one_figure(tmp_path):
    output = tmp_path / "run"
    completed = run_example(output)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    study = json.loads((output / "results.json").read_text())
    assert study["status"] == "completed" and study["backend"] == "numpy"
    assert study["K_reference"] == pytest.approx(1.237735849056604)
    assert study["requested_grids"] == [8, 14]
    assert study["selected_grids"] == {"gani": [9, 15], "hex8": [8, 14]}
    assert "-> selected" in completed.stdout
    assert len(study["points"]) == 4
    for point in study["points"]:
        assert np.isfinite(point["K"]) and point["K"] > 0
        assert len(point["calls"]) == 2
        assert 0 < point["warm_min_s"] <= point["warm_median_s"] <= point["warm_max_s"]
        for call in point["calls"]:
            assert len(call["audits"]) == 6
            assert all(audit["accepted"] for audit in call["audits"])
    with (output / "convergence.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert {p.name for p in output.iterdir()} == {
        "results.json",
        "convergence.csv",
        "hashin_convergence.png",
    }
    assert (
        (output / "hashin_convergence.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )
    # A second invocation must preserve the complete earlier run.
    original = (output / "results.json").read_bytes()
    rejected = run_example(output)
    assert (
        rejected.returncode != 0 and "previous results are preserved" in rejected.stderr
    )
    assert (output / "results.json").read_bytes() == original


def test_invalid_policy_is_rejected_before_solving(tmp_path):
    output = tmp_path / "bad-grid"
    rejected = run_example(output, "--grid-policy", "typo")
    assert rejected.returncode != 0 and "invalid choice" in rejected.stderr
    assert not output.exists()


@pytest.mark.integration
def test_shared_runner_and_plot_work_for_a_different_case_without_analytic_reference(
    tmp_path,
):
    study, section, output = run_convergence(
        "laminate",
        grids=(8, 12),
        grid_policy="even",
        backend="numpy",
        precision="fp64",
        repeats=1,
        output=tmp_path / "laminate",
    )
    assert study["case"] == "laminate" and study["K_reference"] is None
    assert study["selected_grids"] == {"gani": [8, 12], "hex8": [8, 12]}
    assert all(p["relative_error"] is None for p in study["points"])
    assert section.shape == (12, 12)
    plot_convergence(study, section, output / "laminate.png")
    assert (output / "laminate.png").is_file()


@pytest.mark.gpu
def test_odd_policy_runs_both_methods_in_mixed_precision(tmp_path):
    study, _, _ = run_convergence(
        "laminate",
        grids=(8, 14),
        grid_policy="odd-13",
        precision="mixed",
        backend="cupy",
        repeats=1,
        output=tmp_path / "odd-laminate",
    )
    assert study["selected_grids"] == {"gani": [9, 15], "hex8": [9, 15]}
    assert study["status"] == "completed"
    assert all(
        a["accepted"] for p in study["points"] for c in p["calls"] for a in c["audits"]
    )
