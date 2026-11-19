"""Independent finite-segment periodic audit for SAM's capsule contact model."""

import itertools

import numpy as np


def segment_distance(delta, u, v):
    """Minimum of a convex quadratic: interior solution plus four boundaries."""
    w = -delta - 0.5 * u + 0.5 * v
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d = np.sum(u * w, axis=-1)
    e = np.sum(v * w, axis=-1)

    def distance2(s, t):
        r = w + np.asarray(s)[..., None] * u - np.asarray(t)[..., None] * v
        return np.sum(r * r, axis=-1)

    best = np.minimum.reduce(
        [
            distance2(0.0, np.clip(e / c, 0, 1)),
            distance2(1.0, np.clip((e + b) / c, 0, 1)),
            distance2(np.clip(-d / a, 0, 1), 0.0),
            distance2(np.clip((b - d) / a, 0, 1), 1.0),
        ]
    )
    det = a * c - b * b
    safe = np.maximum(det, 1e-30)
    s = (b * e - c * d) / safe
    t = (a * e - b * d) / safe
    interior = (det > 1e-12 * a * c) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    return np.sqrt(
        np.maximum(np.where(interior, np.minimum(best, distance2(s, t)), best), 0.0)
    )


def audit_periodic_fibers(fibers, box, tolerance_relative, *, block_pairs=65536):
    """Audit every possible periodic contact, including nonzero self images.

    Lengths/diameters and box use the same units. The diameter-relative
    tolerance follows SAM's segment/capsule model, not flat cylinder surfaces.
    Enumerate the upper triangle in bounded blocks, independent of optimizer
    neighbor lists. Memory is O(block_pairs + number_of_fibers), even for dense
    or long-fiber configurations. A nearest-center box bound only rejects pairs
    that cannot meet in ANY periodic image; all remaining images are checked.
    """
    f = np.asarray(fibers, dtype=float)
    box = np.broadcast_to(np.asarray(box, dtype=float), (3,))
    if f.ndim != 2 or f.shape[1] != 8 or not np.isfinite(f).all() or len(f) == 0:
        raise ValueError("Expected finite nonempty fiber table with eight columns")
    if (
        not np.isfinite(box).all()
        or not np.isfinite(tolerance_relative)
        or np.any(box <= 0)
        or np.any(f[:, 6:] <= 0)
        or tolerance_relative < 0
    ):
        raise ValueError(
            "Positive box/fiber dimensions and nonnegative tolerance required"
        )
    if (
        isinstance(block_pairs, bool)
        or not isinstance(block_pairs, (int, np.integer))
        or block_pairs < 1
    ):
        raise ValueError("block_pairs must be a positive integer")
    if not np.allclose(np.linalg.norm(f[:, 3:6], axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Fiber orientations must be unit vectors")
    limits = np.ceil((np.max(f[:, 6]) + np.max(f[:, 7])) / box).astype(int)
    count = 0
    missed = 0
    maximum = 0.0
    worst = None
    worst_key = None
    n = len(f)
    row_ends = np.cumsum(np.arange(n, 0, -1, dtype=np.int64))
    row_starts = np.r_[0, row_ends[:-1]]
    for start in range(0, int(row_ends[-1]), block_pairs):
        flat = np.arange(
            start, min(start + block_pairs, int(row_ends[-1])), dtype=np.int64
        )
        i = np.searchsorted(row_ends, flat, side="right")
        j = i + flat - row_starts[i]
        delta = f[j, :3] - f[i, :3]
        delta -= np.rint(delta / box) * box
        u = f[i, 3:6] * f[i, 6, None]
        v = f[j, 3:6] * f[j, 6, None]
        diam = 0.5 * (f[i, 7] + f[j, 7])
        ext = 0.5 * (np.abs(u) + np.abs(v)) + diam[:, None]
        possible = np.all(np.abs(delta) <= ext + 1e-12, axis=1)
        if not possible.any():
            continue
        i, j, delta, u, v, diam, ext = (
            a[possible] for a in (i, j, delta, u, v, diam, ext)
        )
        for shift in itertools.product(*(range(-m, m + 1) for m in limits)):
            other = delta + box * np.array(shift)
            keep = np.all(np.abs(other) <= ext + 1e-12, axis=1)
            if shift == (0, 0, 0):
                keep &= i != j
            if not keep.any():
                continue
            distances = segment_distance(other[keep], u[keep], v[keep])
            relative = (diam[keep] - distances) / diam[keep]
            bad = relative > tolerance_relative + 1e-10
            count += int(bad.sum())
            if shift != (0, 0, 0):
                missed += int(bad.sum())
            at = int(np.argmax(relative))
            value = float(relative[at])
            chosen = np.flatnonzero(keep)[at]
            key = (*shift, int(i[chosen]), int(j[chosen]))
            # Preserve the exhaustive reference's tie order: image, then pair.
            if value > maximum or (value > 0 and value == maximum and key < worst_key):
                maximum = value
                worst_key = key
                worst = dict(
                    i=int(i[chosen]),
                    j=int(j[chosen]),
                    image_shift=list(shift),
                    distance=float(distances[at]),
                )
    return dict(
        accepted=count == 0,
        violating_image_contacts=count,
        nonnearest_violating_image_contacts=missed,
        maximum_relative_overlap=maximum,
        tolerance_relative=float(tolerance_relative),
        worst=worst,
        model="periodic finite segments minus radii, same capsule criterion as SAM; self images included",
    )
