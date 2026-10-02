"""Tune complete immutable H applications once per compatible FFT configuration.

This chooses execution batching, not frequencies, materials or tolerances.
Explicit FFT policies bypass it. Cold tuning and cache provenance are returned
to the per-load solver info; warm solver timing never hides the initial cost.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from cufhom_core.tensors import fft

_CHOICES = {}
_SCHEMA = 1
_POLICIES = ('full', 'component', 'batch2', 'batch3')


def configuration(shape, dtype):
    cp = fft.cp
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device['name']
    if isinstance(name, bytes):
        name = name.decode()
    return dict(schema=_SCHEMA, shape=list(shape), field_dtype=str(np.dtype(dtype)),
        batch=6, layout='C contiguous components first', axes=[1, 2, 3],
        inverse='raw C2R consumable input', gpu=name,
        gpu_identity=str(device.get('uuid', (device.get('pciBusID'), device.get('pciDeviceID')))),
        compute_capability=[device['major'], device['minor']],
        cupy=cp.__version__, cufft=cp.cuda.cufft.getVersion(),
        runtime=cp.cuda.runtime.runtimeGetVersion(), driver=cp.cuda.runtime.driverGetVersion())


def cache_path():
    custom = os.environ.get('CUFHOM_FFT_TUNING_CACHE')
    if custom:
        return Path(custom).expanduser()
    return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'cufhom' / 'fft_tuning_v1.json'


def read_cache(path):
    try:
        content = json.loads(path.read_text())
        return (content if isinstance(content, dict) and content.get('schema') == _SCHEMA
                and isinstance(content.get('choices'), dict) else dict(schema=_SCHEMA, choices={}))
    except (OSError, ValueError):
        return dict(schema=_SCHEMA, choices={})


def save_choice(path, key, record):
    """Atomic replace plus a file lock prevents lost updates across processes."""
    temporary = None
    try:
        import fcntl
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            content = read_cache(path)
            content.setdefault('choices', {})[key] = record
            with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name + '.', delete=False) as f:
                temporary = Path(f.name)
                json.dump(content, f, indent=2, allow_nan=False)
                f.write('\n')
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


def valid_record(record, metadata):
    return (isinstance(record, dict) and record.get('configuration') == metadata
            and record.get('policy') in _POLICIES)


def tune(action, probe, shape):
    """Return the decision and leave the selected policy active for this solve."""
    requested = fft.CUPY_RFFT_BATCH_MODE
    try:
        return _tune(action, probe, shape)
    except Exception:
        fft.set_cupy_rfft_batch_mode(requested)
        raise


def _tune(action, probe, shape):
    cp = fft.cp
    requested = fft.CUPY_RFFT_BATCH_MODE
    if requested != 'auto':
        return dict(status='explicit_policy', policy=requested, tuning_s=0.)
    if fft.CUPY_PLAN_MODE != 'manual' or probe.dtype != cp.complex64 or np.prod(shape) % 2:
        return dict(status='incompatible_configuration', policy='auto', tuning_s=0.)
    metadata = configuration(shape, probe.real.dtype)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path = cache_path()
    # Include the cache location so isolated runs and a user's overrides do
    # not accidentally share decisions through this process-global table.
    memory_key = (str(path.resolve()), key)
    cached = _CHOICES.get(memory_key)
    source = 'memory_cache'
    if not valid_record(cached, metadata):
        cached = read_cache(path).get('choices', {}).get(key)
        source = 'disk_cache'
    if valid_record(cached, metadata):
        _CHOICES[memory_key] = cached
        fft.set_cupy_rfft_batch_mode(cached['policy'])
        return dict(status=source, policy=cached['policy'], tuning_s=0., cache_key=key)
    stream = cp.cuda.get_current_stream()
    stream.synchronize()
    started = time.perf_counter()
    plans_before = set(fft._CUPY_PLAN_CACHE)
    reference = action(probe)
    stream.synchronize()
    baseline = 'component' if tuple(shape) == (240, 240, 240) and fft._component_fft_device_supported() else 'full'
    candidates = {}
    for policy in _POLICIES:
        fft.set_cupy_rfft_batch_mode(policy)
        try:
            candidate = action(probe)
            stream.synchronize()
            numerator = denominator = 0.
            for first in range(0, probe.size, 1024 * 1024):
                x = candidate[first:first + 1024 * 1024]
                y = reference[first:first + 1024 * 1024]
                numerator += float(cp.sum(cp.abs(x - y)**2, dtype=cp.float64))
                denominator += float(cp.sum(cp.abs(y)**2, dtype=cp.float64))
            error = np.sqrt(numerator / denominator) if denominator else np.sqrt(numerator)
            if not np.isfinite(error) or error > 2e-6:
                candidates[policy] = dict(status='operator_mismatch', relative_error=float(error))
            else:
                candidates[policy] = dict(status='valid', relative_error=float(error), samples_s=[])
            del candidate
        except (cp.cuda.memory.OutOfMemoryError, cp.cuda.cufft.CuFFTError) as error:
            if policy == baseline:
                fft.set_cupy_rfft_batch_mode(requested)
                raise
            candidates[policy] = dict(status='unsupported', reason=str(error))
    valid = [policy for policy in _POLICIES if candidates[policy]['status'] == 'valid']
    for repetition in range(5):
        order = valid[repetition % len(valid):] + valid[:repetition % len(valid)] if valid else []
        for policy in order:
            fft.set_cupy_rfft_batch_mode(policy)
            stream.synchronize()
            start = time.perf_counter()
            result = action(probe)
            stream.synchronize()
            candidates[policy]['samples_s'].append(time.perf_counter() - start)
            del result
    del reference
    if not valid or baseline not in valid:
        fft.set_cupy_rfft_batch_mode(requested)
        raise RuntimeError('FFT autotuning could not validate the existing operator')
    for policy in valid:
        candidates[policy]['median_s'] = float(np.median(candidates[policy]['samples_s']))
    best = min(valid, key=lambda policy: candidates[policy]['median_s'])
    # A small noisy difference cannot justify replacing the established policy.
    if candidates[best]['median_s'] > .97 * candidates[baseline]['median_s']:
        best = baseline
    fft.set_cupy_rfft_batch_mode(best)
    chosen_batch = {'full': 6, 'component': 1, 'batch2': 2, 'batch3': 3}[best]
    # Drop only newly-created losing plans; retain plans that preceded tuning.
    for plan_key in set(fft._CUPY_PLAN_CACHE) - plans_before:
        if plan_key[1][0] != chosen_batch:
            del fft._CUPY_PLAN_CACHE[plan_key]
    elapsed = time.perf_counter() - started
    record = dict(configuration=metadata, policy=best, baseline=baseline, candidates=candidates,
                  tuning_s=elapsed, scope='five rotated warm complete-H timings; verified frozen output; no CG timing claim')
    _CHOICES[memory_key] = record
    persisted = save_choice(path, key, record)
    return dict(status='tuned', policy=best, tuning_s=elapsed, cache_key=key,
                persisted=persisted, candidates=candidates)
