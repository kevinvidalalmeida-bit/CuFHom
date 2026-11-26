import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from cufhom import runtime


def test_save_json_concurrent_writers_use_distinct_atomic_temporaries(
    tmp_path, monkeypatch
):
    destination = tmp_path / "result.json"
    writers = 8
    barrier = threading.Barrier(writers, timeout=10)
    real_replace = runtime.os.replace
    temporary_paths = []
    paths_lock = threading.Lock()

    def synchronized_replace(source, target):
        # All complete documents must coexist before any writer can replace
        # the destination; a shared temporary cannot satisfy this condition.
        with paths_lock:
            temporary_paths.append(source)
        barrier.wait()
        json.loads(source.read_text(encoding="utf8"))
        real_replace(source, target)

    monkeypatch.setattr(runtime.os, "replace", synchronized_replace)
    documents = [{"writer": i, "values": [i] * 1000} for i in range(writers)]
    with ThreadPoolExecutor(max_workers=writers) as pool:
        list(pool.map(lambda value: runtime.save_json(destination, value), documents))

    assert len(set(temporary_paths)) == writers
    assert all(path.parent == destination.parent for path in temporary_paths)
    assert json.loads(destination.read_text(encoding="utf8")) in documents
    assert sorted(tmp_path.iterdir()) == [destination]


def test_save_json_cleans_temporary_on_serialization_failure(tmp_path):
    destination = tmp_path / "result.json"
    destination.write_text('{"old": true}', encoding="utf8")

    with pytest.raises(ValueError):
        runtime.save_json(destination, {"value": float("nan")})

    assert json.loads(destination.read_text(encoding="utf8")) == {"old": True}
    assert sorted(tmp_path.iterdir()) == [destination]


def test_save_json_serializes_arrays_scalars_and_paths(tmp_path):
    """Portable output preserves array values and scalar JSON types."""
    import numpy as np

    destination = tmp_path / "nested" / "result.json"
    runtime.save_json(
        destination,
        {
            "tensor": np.array([[1.25, 2.5]], dtype=np.float32),
            "iterations": np.int64(6),
            "accepted": np.bool_(True),
            "source": tmp_path / "input.npz",
        },
    )

    assert json.loads(destination.read_text(encoding="utf8")) == {
        "tensor": [[1.25, 2.5]],
        "iterations": 6,
        "accepted": True,
        "source": str(tmp_path / "input.npz"),
    }
    assert list(destination.parent.iterdir()) == [destination]


def test_save_json_preserves_previous_result_when_replace_fails(tmp_path, monkeypatch):
    """An OS replacement error must not damage an existing result."""
    destination = tmp_path / "result.json"
    original = '{"accepted": true}\n'
    destination.write_text(original, encoding="utf8")

    def reject_replace(source, target):
        raise PermissionError("injected replacement failure")

    monkeypatch.setattr(runtime.os, "replace", reject_replace)
    with pytest.raises(PermissionError, match="injected replacement failure"):
        runtime.save_json(destination, {"accepted": False})

    assert destination.read_text(encoding="utf8") == original
    assert list(tmp_path.iterdir()) == [destination]
