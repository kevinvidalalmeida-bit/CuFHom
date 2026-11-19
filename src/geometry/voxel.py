"""Periodic rasterization, physical volumes and voxel orientation moments."""

from itertools import combinations_with_replacement, permutations

import numpy as np
from numba import njit, prange


def fiber_volume(length, diameter, shape="capsule"):
    """Return the volume of a finite cylinder or a cylinder with spherical caps."""
    if shape not in ("capsule", "cylinder"):
        raise ValueError("Shape must be capsule or cylinder")
    if not np.isfinite([length, diameter]).all() or min(length, diameter) <= 0:
        raise ValueError("Positive finite fiber dimensions required")
    volume = np.pi * diameter**2 * length / 4
    return volume + (np.pi * diameter**3 / 6 if shape == "capsule" else 0.0)


@njit(cache=True)
def _paint_fibers(centers, directions, length, radius, phase, ori, capsule):
    """Slice along the dominant axis: O(length * radius²) samples per fiber.

    Intersect each grid plane with a bounding rectangle of the infinite-cylinder
    ellipse, then apply the exact axial and radial tests. Enumerate unwrapped
    points before applying periodic indices, including multiple cell crossings.
    Cylinders sample the strict interior; capsules include their boundary.
    """
    n = len(phase)
    count = 0
    for fiber in range(len(centers)):
        c, u = centers[fiber], directions[fiber]
        k = int(np.argmax(np.abs(u)))
        i, j = (k + 1) % 3, (k + 2) % 3
        half = length / 2
        extent = half * abs(u[k]) + radius * np.sqrt(max(0.0, 1 - u[k] * u[k]))
        if capsule:
            extent = half * abs(u[k]) + radius
        ri = radius * np.sqrt(1 + (u[i] / u[k]) ** 2)
        rj = radius * np.sqrt(1 + (u[j] / u[k]) ** 2)
        for xk in range(int(np.floor(c[k] - extent)), int(np.ceil(c[k] + extent)) + 1):
            dk = xk - c[k]
            ci, cj = c[i] + dk * u[i] / u[k], c[j] + dk * u[j] / u[k]
            for xi in range(int(np.floor(ci - ri)), int(np.ceil(ci + ri)) + 1):
                di = xi - c[i]
                for xj in range(int(np.floor(cj - rj)), int(np.ceil(cj + rj)) + 1):
                    dj = xj - c[j]
                    axial = dk * u[k] + di * u[i] + dj * u[j]
                    if not capsule and abs(axial) >= half:
                        continue
                    if capsule:
                        axial = max(-half, min(half, axial))
                    radial = (
                        (dk - axial * u[k]) ** 2
                        + (di - axial * u[i]) ** 2
                        + (dj - axial * u[j]) ** 2
                    )
                    if (
                        radial > radius * radius
                        if capsule
                        else radial >= radius * radius
                    ):
                        continue
                    if k == 0:
                        x, y, z = xk % n, xi % n, xj % n
                    elif k == 1:
                        x, y, z = xj % n, xk % n, xi % n
                    else:
                        x, y, z = xi % n, xj % n, xk % n
                    if phase[x, y, z] == 0:
                        count += 1
                    phase[x, y, z] = 1
                    ori[x, y, z] = u
    return count


def rasterize_fibers(centers, directions, length, radius, grid, shape="capsule"):
    """Rasterize every fiber; coordinates and dimensions are in voxel units."""
    c = np.ascontiguousarray(centers, dtype=float)
    u = np.ascontiguousarray(directions, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or u.shape != c.shape:
        raise ValueError("Centers and directions must have shape (n, 3)")
    if (
        not np.isfinite(c).all()
        or not np.isfinite(u).all()
        or not np.allclose(np.linalg.norm(u, axis=1), 1.0, rtol=0, atol=1e-10)
    ):
        raise ValueError("Finite centers and unit directions required")
    fiber_volume(length, 2 * radius, shape)
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError("Integer grid >= 2 required")
    phase = np.zeros((grid, grid, grid), dtype=np.uint8)
    ori = np.zeros((grid, grid, grid, 3), dtype=np.float32)
    count = _paint_fibers(
        c, u, float(length), float(radius), phase, ori, shape == "capsule"
    )
    return phase, ori, count, len(c)


def rasterize_cylinders(centers, directions, length, radius, grid):
    """Rasterize periodic cylinders at voxel centers on the requested grid."""
    return rasterize_fibers(centers, directions, length, radius, grid, "cylinder")


def rasterize_capsules(centers, directions, length, radius, grid):
    """Rasterize periodic capsules at voxel centers on the requested grid."""
    return rasterize_fibers(centers, directions, length, radius, grid, "capsule")


_A4_COMPONENTS = np.array(
    list(combinations_with_replacement(range(3), 4)), dtype=np.int64
)


@njit(cache=True, parallel=True)
def _voxel_a4_components(phase, orientations):
    """15 independent components, O(blocks) scratch rather than O(voxels*81)."""
    block_size = 16384
    blocks = (len(phase) + block_size - 1) // block_size
    partial = np.zeros((blocks, 16), dtype=np.float64)
    for block in prange(blocks):
        for at in range(block * block_size, min((block + 1) * block_size, len(phase))):
            if phase[at] == 0:
                continue
            norm2 = 0.0
            for axis in range(3):
                norm2 += np.float64(orientations[at, axis]) ** 2
            if not np.isfinite(norm2) or norm2 <= 1e-24:
                continue
            for component in range(15):
                value = 1.0
                for axis in range(4):
                    value *= np.float64(
                        orientations[at, _A4_COMPONENTS[component, axis]]
                    )
                partial[block, component] += value / (norm2 * norm2)
            partial[block, 15] += 1.0
    total = np.zeros(16)
    for block in range(blocks):
        for component in range(16):
            total[component] += partial[block, component]
    return total


def voxel_a4(phase, orientations):
    """Full symmetric fourth moment of occupied voxels, unit-axis convention."""
    total = _voxel_a4_components(
        np.asarray(phase).reshape(-1), np.asarray(orientations).reshape(-1, 3)
    )
    result = np.zeros((3, 3, 3, 3))
    if total[15]:
        for component, indices in enumerate(_A4_COMPONENTS):
            for perm in set(permutations(indices)):
                result[perm] = total[component] / total[15]
    return result


@njit(cache=True, parallel=True)
def _voxel_a2(phase, orientations):
    block_size = 16384
    blocks = (len(phase) + block_size - 1) // block_size
    partial = np.zeros((blocks, 7), dtype=np.float64)
    for block in prange(blocks):
        xx = 0.0
        yy = 0.0
        zz = 0.0
        xy = 0.0
        xz = 0.0
        yz = 0.0
        count = 0.0
        for at in range(block * block_size, min((block + 1) * block_size, len(phase))):
            if phase[at] == 0:
                continue
            x = np.float64(orientations[at, 0])
            y = np.float64(orientations[at, 1])
            z = np.float64(orientations[at, 2])
            norm = np.sqrt(x * x + y * y + z * z)
            if norm <= 1e-12 or np.isnan(norm):
                continue
            x /= norm
            y /= norm
            z /= norm
            xx += x * x
            yy += y * y
            zz += z * z
            xy += x * y
            xz += x * z
            yz += y * z
            count += 1.0
        partial[block, 0] = xx
        partial[block, 1] = yy
        partial[block, 2] = zz
        partial[block, 3] = xy
        partial[block, 4] = xz
        partial[block, 5] = yz
        partial[block, 6] = count
    totals = np.zeros(7, dtype=np.float64)
    for block in range(blocks):
        for component in range(7):
            totals[component] += partial[block, component]
    result = np.zeros((3, 3), dtype=np.float64)
    if totals[6] > 0:
        result[0, 0] = totals[0] / totals[6]
        result[1, 1] = totals[1] / totals[6]
        result[2, 2] = totals[2] / totals[6]
        result[0, 1] = result[1, 0] = totals[3] / totals[6]
        result[0, 2] = result[2, 0] = totals[4] / totals[6]
        result[1, 2] = result[2, 1] = totals[5] / totals[6]
    return result


def voxel_a2(phase, orientations):
    """Second moment of occupied voxels, normalizing each valid direction."""
    return _voxel_a2(
        np.asarray(phase).reshape(-1), np.asarray(orientations).reshape(-1, 3)
    )
