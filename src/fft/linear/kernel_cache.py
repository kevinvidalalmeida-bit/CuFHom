"""Device properties and a persistent record of compiled fused kernels.

CuPy keeps compiled kernels in its own disk cache. This JSON file only
records that a grid's kernels are already there, so a later process can use
them at once instead of compiling them in the background (fused.py).
"""

import json
import os
import tempfile
from pathlib import Path

from cufhom_core.tensors import fft

_DEVICE_PROPERTIES = {}


def device_properties():
    """Return cached CUDA device properties to avoid repeated driver queries."""
    cp = fft.cp
    device = cp.cuda.runtime.getDevice()
    if device not in _DEVICE_PROPERTIES:
        _DEVICE_PROPERTIES[device] = cp.cuda.runtime.getDeviceProperties(device)
    return _DEVICE_PROPERTIES[device]


def cache_path():
    """Return the configured kernel-cache path or its user-cache default."""
    custom = os.environ.get("CUFHOM_KERNEL_CACHE")
    if custom:
        return Path(custom).expanduser()
    return (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "cufhom"
        / "kernels_v1.json"
    )


def read_records(path):
    """Read cache records, returning an empty mapping for missing or invalid data."""
    try:
        content = json.loads(path.read_text())
        return content if isinstance(content, dict) else {}
    except (OSError, ValueError):
        return {}


def save_record(path, key, record):
    """Atomic replace plus a file lock prevents lost updates across processes."""
    temporary = None
    try:
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            content = read_records(path)
            content[key] = record
            with tempfile.NamedTemporaryFile(
                "w", dir=path.parent, prefix=path.name + ".", delete=False
            ) as f:
                temporary = Path(f.name)
                json.dump(content, f, indent=2, allow_nan=False)
                f.write("\n")
            os.replace(temporary, path)
        return True
    except (OSError, ImportError):
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
