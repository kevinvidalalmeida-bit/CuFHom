"""Check reference-driver contracts without importing optional solver packages."""

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.reproduce import external
from benchmarks.reproduce.external import _source_path, fans, ffthompy, unpack
from benchmarks.reproduce.profile import summarize_intervals


@pytest.mark.parametrize("grid", [4, 8, 16])
def test_ffthompy_rejects_unmatched_even_grids_before_loading_dependencies(grid):
    with pytest.raises(ValueError, match="odd grids"):
        ffthompy(np.zeros((grid,) * 3, np.int32), np.zeros((1, 21)), 1e-6)


@pytest.mark.parametrize("grid", [3, 9, 21])
def test_fans_rejects_odd_grids_before_initializing_cuda(grid):
    with pytest.raises(ValueError, match="even cubic grids"):
        fans(np.zeros((grid,) * 3, np.int32), np.zeros((1, 21)), 1e-6)


@pytest.mark.parametrize("eta", [0, -1, 1, np.inf, np.nan])
def test_reference_drivers_reject_invalid_tolerances(eta):
    with pytest.raises(ValueError, match="eta"):
        ffthompy(np.zeros((3,) * 3, np.int32), np.zeros((1, 21)), eta)
    with pytest.raises(ValueError, match="eta"):
        fans(np.zeros((4,) * 3, np.int32), np.zeros((1, 21)), eta)


def test_reference_path_requires_an_upstream_package(tmp_path):
    with pytest.raises(ValueError, match="__init__.py"):
        _source_path(tmp_path, "ffthompy")


def test_failed_reference_is_saved_without_fabricating_load_audits(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(external, "environment", lambda: {"test": True})
    destination = tmp_path / "failed.json"
    result = external.main(
        [
            "ffthompy",
            "--grid",
            "9",
            "--source",
            str(tmp_path / "missing"),
            "--warm",
            "0",
            "--output",
            str(destination),
        ]
    )
    record = json.loads(destination.read_text())
    assert result == 1
    assert record["status"] == "failed"
    assert record["calls"] == []
    assert "ValueError" in record["error"]
    assert record["input"]["n"] == 9


def test_reference_cli_cannot_overwrite_a_published_record(tmp_path):
    destination = tmp_path / "published.json"
    destination.write_text("unchanged evidence")
    with pytest.raises(SystemExit) as error:
        external.main(["ffthompy", "--grid", "9", "--output", str(destination)])
    assert error.value.code == 2
    assert destination.read_text() == "unchanged evidence"


def test_unpack_preserves_anisotropic_work_and_symmetry():
    # Construct a non-isotropic SPD tensor with normal/shear coupling.
    rng = np.random.default_rng(7)
    factor = rng.normal(size=(6, 6))
    reference = factor.T @ factor + np.eye(6)
    packed = np.asarray([[reference[i, j] for i in range(6) for j in range(i, 6)]])
    actual = unpack(packed)[0]
    strain = rng.normal(size=6)
    assert np.allclose(actual @ strain, reference @ strain)
    assert np.allclose(actual, actual.T)


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([], 0),
        ([(0, 5)], 5),
        ([(4, 7), (0, 5)], 7),
        ([(0, 10), (2, 3), (10, 12)], 12),
        ([(1, 2), (5, 9)], 5),
    ],
)
def test_gpu_interval_union_does_not_double_count_overlap(events, expected):
    assert summarize_intervals(events) == expected


def test_negative_event_duration_is_rejected():
    with pytest.raises(ValueError, match="finish"):
        summarize_intervals([(3, 2)])


def test_campaign_execution_has_no_historical_path_dependencies():
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "reproduce"
    forbidden = ("PAPER_CUFHOM/", "CuFHom_PRISM/", "research/", str(Path.home()) + "/")
    # Plotting archive identifiers intentionally retain original source paths.
    # Execution drivers and their component controls must not depend on them.
    assert root.is_dir()
    sources = list(root.glob("*.py")) + list((root / "adapters").glob("*.py"))
    assert sources
    for source in sources:
        assert not any(token in source.read_text() for token in forbidden), source


def test_rejected_reference_call_returns_failure_and_preserves_audit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(external, "environment", lambda: {"test": True})
    audit = {"relative": 2e-6, "accepted": False}
    monkeypatch.setattr(
        external,
        "fans",
        lambda *args, **kwargs: {
            "status": "residual_not_met",
            "audits": [audit],
            "C": np.eye(6).tolist(),
        },
    )
    destination = tmp_path / "rejected.json"
    result = external.main(
        [
            "fans",
            "--grid",
            "4",
            "--warm",
            "0",
            "--eta",
            "1e-6",
            "--output",
            str(destination),
        ]
    )
    record = json.loads(destination.read_text())
    assert result == 1
    assert record["status"] == "residual_not_met"
    assert record["calls"][0]["audits"] == [audit]
    np.testing.assert_array_equal(record["calls"][0]["C"], np.eye(6))
