"""Validate portable campaign planning and numerical report conventions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

CAMPAIGN = Path(__file__).resolve().parents[1] / "benchmarks" / "reproduce"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(CAMPAIGN))
    spec = importlib.util.spec_from_file_location(
        "benchmark_runner", CAMPAIGN / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_protocol_expands_and_preserves_requested_conditions(runner):
    for path in (CAMPAIGN / "protocols").glob("*.json"):
        protocol = json.loads(path.read_text())
        if "cases" not in protocol:
            continue  # External source pins are metadata, not a campaign matrix.
        points = runner.expand(protocol, "cupy", False)
        assert points
        assert {p["precision"] for p in points} == set(protocol["precisions"])
        assert {p["session"] for p in points} == set(range(1, protocol["sessions"] + 1))
        assert all(p["reference_eta"] == protocol["reference_eta"] for p in points)


@pytest.mark.parametrize(
    "key,value",
    [
        ("etas", [0]),
        ("etas", [float("nan")]),
        ("sessions", 0),
        ("warm_calls", -1),
        ("nu0", [0.5]),
        ("contrasts", [-1]),
    ],
)
def test_invalid_physical_protocol_is_rejected_before_solving(runner, key, value):
    protocol = json.loads((CAMPAIGN / "protocols/public.json").read_text())
    protocol[key] = value
    with pytest.raises(ValueError):
        runner.expand(protocol, "cupy", False)


def test_strict_json_preserves_unavailable_history_without_acceptance_changes(runner):
    record = {
        "audits": [
            {
                "accepted": True,
                "relative": 1e-7,
                "history": np.array([np.nan, np.inf, 0.25]),
            }
        ]
    }
    converted = runner.strict_record(record)
    assert converted["audits"][0]["relative"] == 1e-7
    assert converted["audits"][0]["accepted"] is True
    assert converted["audits"][0]["history"] == [None, None, 0.25]
    assert len(converted["nonfinite_native_metadata_encoded_as_null"]) == 2
    json.dumps(converted, allow_nan=False)


def test_reference_changes_bulk_without_changing_selected_shear(runner):
    shear = 7.5
    for nu in (-0.2, 0.0, 0.3, 0.48):
        matrix = runner.isotropic(2 * shear * (1 + nu), nu)
        np.testing.assert_allclose(np.diag(matrix)[3:], 2 * shear)
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_sensitivity_excludes_homogeneous_curve_from_normalization(monkeypatch):
    monkeypatch.syspath_prepend(str(CAMPAIGN))
    spec = importlib.util.spec_from_file_location(
        "benchmark_summary", CAMPAIGN / "summarize.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = dict(
        status="accepted",
        case="public",
        discretization="gani",
        precision="fp64",
        n=189,
        eta=1e-6,
    )
    rows = [
        base | dict(contrast=chi, nu0=nu, mean_updates=count)
        for chi in (1, 10)
        for nu, count in ((0.3, 0 if chi == 1 else 20), (0.48, 0 if chi == 1 else 60))
    ]
    normalized, diagnostics = module.sensitivity(rows)
    assert all(
        row["normalized_by_minimum"] is None
        for row in normalized
        if row["contrast"] == 1
    )
    curve = next(row for row in diagnostics if row["contrast"] == 10)
    assert curve["minimum_updates"] == 20
    assert curve["penalty_nu048_over_minimum"] == 3
    assert json.loads(curve["nu0_at_sampled_minimum"]) == [0.3]
