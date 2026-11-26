"""Check that paired reports reject incompatible inputs and false acceptance."""

from copy import deepcopy

import numpy as np
import pytest

from benchmarks.paper.ablations import compare_records
from benchmarks.paper.run import host_memory


@pytest.fixture
def record():
    audit = dict(
        accepted=True, residual=1e-7, threshold=1e-6, relative=1e-7, evaluation="df64"
    )
    call = dict(
        state="warm",
        time_s=0.2,
        iterations=[12] * 6,
        audits=[deepcopy(audit) for _ in range(6)],
        C=np.eye(6).tolist(),
    )
    return dict(
        pair="gani_loads",
        precision="mixed",
        n=21,
        input_hashes={"index": "geometry", "stiffness_solver": "material"},
        reference=np.eye(6).tolist(),
        status="accepted",
        calls=[deepcopy(call) for _ in range(5)],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("n", 189),
        ("precision", "fp32"),
        (
            "input_hashes",
            {"index": "different-geometry", "stiffness_solver": "material"},
        ),
        ("reference", (2 * np.eye(6)).tolist()),
    ],
)
def test_mismatched_physical_problem_cannot_be_reported_as_a_pair(record, key, value):
    variant = deepcopy(record)
    variant[key] = value
    with pytest.raises(ValueError, match="disagree"):
        compare_records(record, variant)


@pytest.mark.parametrize("residual", [2e-6, np.nan, np.inf])
def test_accepted_flag_does_not_override_fresh_physical_failure(record, residual):
    variant = deepcopy(record)
    variant["calls"][0]["audits"][2]["residual"] = residual
    with pytest.raises(ValueError, match="acceptance"):
        compare_records(record, variant)


def test_count_changes_and_tensor_differences_remain_visible(record):
    variant = deepcopy(record)
    variant["calls"][0]["iterations"][0] += 1
    variant["calls"][0]["C"][0][0] += 1e-9
    diagnostic = compare_records(record, variant)
    assert diagnostic["variant"]["iterations_per_call"][0][0] == 13
    assert diagnostic["control"]["iterations_per_call"][0][0] == 12
    assert diagnostic["max_relative_tensor_difference"] > 0


def test_intentional_reference_ablation_reports_preconditioner_change(record):
    control = deepcopy(record)
    control["pair"] = "gani_reference"
    variant = deepcopy(control)
    variant["reference"] = (2 * np.eye(6)).tolist()
    diagnostic = compare_records(control, variant)
    assert diagnostic["reference_changed"] is True
    assert diagnostic["max_relative_tensor_difference"] == 0


def test_host_memory_keeps_capacity_and_lifetime_high_water_distinct(monkeypatch):
    def read(path):
        return (
            "MemTotal: 100 kB\nMemAvailable: 80 kB\nSwapTotal: 0 kB\n"
            if str(path) == "/proc/meminfo"
            else "VmRSS: 12 kB\nVmHWM: 20 kB\nVmLck: 4 kB\n"
        )

    monkeypatch.setattr("benchmarks.paper.run.Path.read_text", read)
    result = host_memory()
    assert result["MemTotal_bytes"] == 102400
    assert result["VmRSS_bytes"] == 12288
    assert result["VmHWM_bytes"] == 20480
    assert "not an isolated per-call peak" in result["scope"]
