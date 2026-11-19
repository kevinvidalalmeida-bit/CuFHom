"""CPU-safe probe tests; CUDA cases opt in for an authorized serial GPU slot.

CUFHOM_P2_GPU_TESTS=1 python -B -m pytest --gpu -m gpu -q \
  tests/fft/linear/test_fft_operator_probe.py
"""

import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[3] / "tests/fft/linear/operator_probe.py"
_SPEC = importlib.util.spec_from_file_location("p2_operator_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_host_cli_does_not_import_cuda(tmp_path, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("cupy", "cufhom", "cufhom_core"):
            pytest.fail("host-only probe attempted a CUDA/project import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    path = tmp_path / "host.json"
    probe.main(
        [
            "--working",
            "float32",
            "--evaluation",
            "df64",
            "--shape",
            "15",
            "17",
            "19",
            "--output",
            str(path),
        ]
    )
    record = json.loads(path.read_text())
    assert record["mode"] == "host_only"
    assert record["cases"][0]["shape"] == [15, 17, 19]
    assert "grid_supported" in record["cases"][0]["definitions"]
    assert "tests/fft/linear/operator_probe.py" in record["source_sha256"]
    assert all(len(h) == 64 for h in record["source_sha256"].values())


@pytest.mark.parametrize(
    "shape",
    [
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (15, 17, 19),
        (24, 30, 40),
        (17, 19, 22),
    ],
)
def test_host_shapes_and_memory_estimate(shape):
    record = probe.host_report(shape, "float32", "float64")
    assert record["hybrid_spectrum_estimate_bytes"] == 48 * shape[0] * shape[1] * (
        shape[2] // 2
    )
    assert record["previous_full_solver_estimate_126N_bytes"] == 126 * np.prod(shape)
    assert record["evaluation_requested"] == "float64"


@pytest.mark.parametrize(
    "args",
    [
        (["--working", "df64", "--evaluation", "df64"]),
        (["--working", "float32", "--evaluation", "float32"]),
        (["--working", "float32", "--evaluation", "df64", "--shape", "0", "4", "4"]),
        (["--working", "float32", "--evaluation", "df64", "--repeats", "0"]),
        (
            [
                "--working",
                "float32",
                "--evaluation",
                "df64",
                "--lengths",
                "nan",
                "1",
                "1",
            ]
        ),
        (
            [
                "--working",
                "float32",
                "--evaluation",
                "df64",
                "--max-memory-fraction",
                "1.1",
            ]
        ),
    ],
)
def test_cli_rejects_invalid_precision_or_resource_inputs(args, tmp_path):
    with pytest.raises(SystemExit):
        probe.main([*args, "--output", str(tmp_path / "bad.json")])


def test_equivalence_complex_and_nonfinite():
    ref = np.array([1 + 2j, -3j])
    assert probe.error_metrics(ref, ref)["relative_l2"] == 0
    assert probe.error_metrics(ref + 1e-6, ref)["absolute_linf"] == pytest.approx(1e-6)
    for bad in ([np.nan], [np.inf], [1]):
        with pytest.raises(ValueError):
            probe.error_metrics(bad, ref)


def test_inventory_deduplicates_backing_allocation():
    block = SimpleNamespace(ptr=123, size=512)

    def array(nbytes, shape):
        return SimpleNamespace(
            data=SimpleNamespace(mem=block),
            nbytes=nbytes,
            shape=shape,
            dtype=np.dtype("float64"),
        )

    report = probe.array_inventory(
        {"state": array(64, (8,)), "energy_view": array(8, (1,)), "none": None}
    )
    assert report["logical_bytes"] == 72
    assert report["unique_allocation_bytes"] == 512
    assert report["unique_allocations"] == 1


def test_compile_instrument_restores_on_failure():
    def compile_failed(specs):
        raise RuntimeError("injected compile failure")

    module = SimpleNamespace(_compile_many=compile_failed)
    with pytest.raises(RuntimeError), probe.compile_watch(module):
        module._compile_many([("source", "kernel", 0)])
    assert module._compile_many is compile_failed


def test_case_failure_is_saved_with_traceback(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected probe failure")

    monkeypatch.setattr(probe, "host_report", fail)
    path = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="diagnostics saved"):
        probe.main(
            ["--working", "float32", "--evaluation", "df64", "--output", str(path)]
        )
    record = json.loads(path.read_text())
    assert record["cases"][0]["status"] == "failed"
    assert "injected probe failure" in record["cases"][0]["traceback"]


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("CUFHOM_P2_GPU_TESTS") != "1",
    reason="CUDA requires explicit serial GPU slot: CUFHOM_P2_GPU_TESTS=1",
)
@pytest.mark.parametrize(
    "shape,generated",
    [
        ((16, 16, 16), True),
        ((24, 30, 40), True),
        ((15, 17, 19), False),
        ((17, 19, 22), False),
    ],
)
@pytest.mark.parametrize(
    "working,evaluation", [("float32", "df64"), ("float64", "float64")]
)
def test_generated_and_library_equivalent_actions(
    shape, generated, working, evaluation, monkeypatch
):
    pytest.importorskip("cupy")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    args = SimpleNamespace(
        working=working,
        evaluation=evaluation,
        lengths=(0.8, 1.3, 1.1),
        seed=1701,
        max_memory_fraction=0.5,
        repeats=1,
        batch=1,
    )
    case = probe.gpu_case(shape, args)
    assert case["generated_supported"] is generated
    assert case["library"]["median_event_ms"] > 0
    if not generated:
        assert case["status"] == "library_only"
        return
    assert case["operator_error"]["relative_l2"] < (
        5e-6 if working == "float32" else 2e-12
    )
    assert case["fresh_evaluation"]["error"]["relative_l2"] < 3e-7
    assert len(case["kernels"]) == 5
    assert all(k["attributes"]["num_regs"] > 0 for k in case["kernels"])
    assert case["generated_buffers"]["unique_allocation_bytes"] > 0
