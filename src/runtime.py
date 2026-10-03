"""Small portable I/O helpers; no campaign configuration or environment switch."""
import json
from pathlib import Path
import time


def progress(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save_json(path, value):
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
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf8')
    pending.replace(path)
