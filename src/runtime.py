"""Small portable I/O helpers; no campaign configuration or environment switch."""

import json
import os
import tempfile
import time
from pathlib import Path


def progress(message):
    """Print a timestamped progress message and flush the output stream."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(path, value):
    """Atomically write finite JSON data, converting NumPy values and paths.

    Existing output is replaced only after serialization succeeds. Temporary
    files are created beside the destination and removed after failures.
    """
    import numpy as np

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A unique sibling keeps replacement atomic on the same filesystem and
    # prevents concurrent writers from sharing a temporary filename.
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".pending",
            delete=False,
        ) as stream:
            pending = Path(stream.name)
            json.dump(value, stream, indent=2, default=convert, allow_nan=False)
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
