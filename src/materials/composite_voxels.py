"""Sub-voxel geometry and laminate interface voxels for Cartesian FFT grids.

The FFT solver must use a uniform periodic grid; refining only a few cells would
destroy the circulant structure used by the FFT.  This module therefore attacks
the geometric error instead: a cell cut by a smooth two-phase interface stores
its fibre volume fraction and the interface normal, and receives the exact
rank-one-laminate stiffness for those data.

The construction is useful for analytically defined capsules (short fibres),
but the stiffness routine is independent of how fractions and normals were
obtained.  Fractions/normals estimated from a high-resolution image can be
used as well.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import numba

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - pure NumPy fallback remains supported.
    numba = None
    _NUMBA_AVAILABLE = False


_MANDEL_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_MANDEL_FACTOR = np.array((1.0, 1.0, 1.0, np.sqrt(2.0), np.sqrt(2.0), np.sqrt(2.0)))
_SYM21_PAIRS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 3),
    (3, 4),
    (3, 5),
    (4, 4),
    (4, 5),
    (5, 5),
)


if _NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def _rasterize_capsules_composite_numba(
        centers: np.ndarray,
        directions: np.ndarray,
        fiber_length: float,
        fiber_radius: float,
        grid_size: int,
        offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rasterize the union serially, touching only each fibre’s bounding box."""
        fraction = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
        orientation = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
        normal = np.zeros((grid_size, grid_size, grid_size, 3), dtype=np.float32)
        radius_sq = fiber_radius * fiber_radius
        half_length = 0.5 * fiber_length
        n_offsets = offsets.shape[0]
        for fibre in range(centers.shape[0]):
            cx, cy, cz = centers[fibre]
            ux, uy, uz = directions[fibre]
            p0x = cx - half_length * ux
            p0y = cy - half_length * uy
            p0z = cz - half_length * uz
            p1x = cx + half_length * ux
            p1y = cy + half_length * uy
            p1z = cz + half_length * uz
            ax, ay, az = p1x - p0x, p1y - p0y, p1z - p0z
            axis_sq = ax * ax + ay * ay + az * az
            lo0 = int(np.floor(min(p0x, p1x) - (fiber_radius + 0.5)))
            lo1 = int(np.floor(min(p0y, p1y) - (fiber_radius + 0.5)))
            lo2 = int(np.floor(min(p0z, p1z) - (fiber_radius + 0.5)))
            hi0 = int(np.ceil(max(p0x, p1x) + (fiber_radius + 0.5)))
            hi1 = int(np.ceil(max(p0y, p1y) + (fiber_radius + 0.5)))
            hi2 = int(np.ceil(max(p0z, p1z) + (fiber_radius + 0.5)))
            for ix in range(lo0, hi0 + 1):
                for iy in range(lo1, hi1 + 1):
                    for iz in range(lo2, hi2 + 1):
                        inside = 0
                        for sample in range(n_offsets):
                            px = float(ix) + offsets[sample, 0]
                            py = float(iy) + offsets[sample, 1]
                            pz = float(iz) + offsets[sample, 2]
                            t = (
                                (px - p0x) * ax + (py - p0y) * ay + (pz - p0z) * az
                            ) / axis_sq
                            if t < 0.0:
                                t = 0.0
                            elif t > 1.0:
                                t = 1.0
                            dx = px - (p0x + t * ax)
                            dy = py - (p0y + t * ay)
                            dz = pz - (p0z + t * az)
                            if dx * dx + dy * dy + dz * dz <= radius_sq:
                                inside += 1
                        local_fraction = inside / n_offsets
                        wx, wy, wz = ix % grid_size, iy % grid_size, iz % grid_size
                        if local_fraction <= fraction[wx, wy, wz] or inside == 0:
                            continue
                        t = (
                            (float(ix) - p0x) * ax
                            + (float(iy) - p0y) * ay
                            + (float(iz) - p0z) * az
                        ) / axis_sq
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0
                        nx = float(ix) - (p0x + t * ax)
                        ny = float(iy) - (p0y + t * ay)
                        nz = float(iz) - (p0z + t * az)
                        nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
                        if nrm > 1.0e-14:
                            nx, ny, nz = nx / nrm, ny / nrm, nz / nrm
                        else:
                            # A cut cell centred on the axis has no radial
                            # normal at its centre.  The axial fallback is
                            # appropriate for a capsule end and stable below.
                            nx, ny, nz = ux, uy, uz
                        fraction[wx, wy, wz] = local_fraction
                        orientation[wx, wy, wz, 0] = ux
                        orientation[wx, wy, wz, 1] = uy
                        orientation[wx, wy, wz, 2] = uz
                        normal[wx, wy, wz, 0] = nx
                        normal[wx, wy, wz, 1] = ny
                        normal[wx, wy, wz, 2] = nz
        return fraction, orientation, normal


def _capsule_distance_and_normal(
    points: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return distance to a capsule axis and outward normal at ``points``."""
    axis = p1 - p0
    length_sq = float(axis @ axis)
    if length_sq <= 1.0e-20:
        raise ValueError("A capsule must have non-zero axis length.")
    relative = points - p0
    t = np.clip(relative @ axis / length_sq, 0.0, 1.0)
    radial = relative - t[:, None] * axis
    distance = np.linalg.norm(radial, axis=1)
    normal = np.zeros_like(radial)
    valid = distance > 1.0e-14
    normal[valid] = radial[valid] / distance[valid, None]
    return distance, normal


def rasterize_capsules_composite(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    grid_size: int,
    *,
    samples_per_axis: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize periodic capsules with deterministic sub-voxel quadrature.

    Parameters are expressed in voxel units.  The result is ``(fraction, ori,
    normal)`` with shapes ``(N,N,N)``, ``(N,N,N,3)`` and ``(N,N,N,3)``.
    Fractions are the fibre volume in a cell.  At cut cells, normals point from
    the fibre axis outwards; their sign does not affect laminate stiffness.

    This deliberately evaluates only each capsule bounding box.  It is
    intended for coarse grids, where correcting the geometry is much cheaper
    than globally increasing the FFT grid resolution.
    """
    centers = np.asarray(centers, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or directions.shape != centers.shape:
        raise ValueError("centers and directions must have shape (n_fibres, 3).")
    if grid_size < 2 or fiber_length <= 0.0 or fiber_radius <= 0.0:
        raise ValueError("grid_size, fiber_length and fiber_radius must be positive.")
    samples_per_axis = int(samples_per_axis)
    if samples_per_axis < 1:
        raise ValueError("samples_per_axis must be at least one.")

    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1.0e-14):
        raise ValueError("All fibre directions must be non-zero.")
    directions = directions / norms[:, None]
    fraction = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    orientation = np.zeros((*fraction.shape, 3), dtype=np.float32)
    normal = np.zeros_like(orientation)

    offsets_1d = (
        np.arange(samples_per_axis, dtype=np.float64) + 0.5
    ) / samples_per_axis - 0.5
    sx, sy, sz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
    offsets = np.column_stack((sx.ravel(), sy.ravel(), sz.ravel()))
    if _NUMBA_AVAILABLE:
        return _rasterize_capsules_composite_numba(
            np.ascontiguousarray(centers),
            np.ascontiguousarray(directions),
            float(fiber_length),
            float(fiber_radius),
            int(grid_size),
            np.ascontiguousarray(offsets),
        )

    for center, direction in zip(centers, directions):
        p0 = center - 0.5 * fiber_length * direction
        p1 = center + 0.5 * fiber_length * direction
        lower = np.floor(np.minimum(p0, p1) - (fiber_radius + 0.5)).astype(int)
        upper = np.ceil(np.maximum(p0, p1) + (fiber_radius + 0.5)).astype(int)
        axes = [np.arange(lower[i], upper[i] + 1, dtype=np.int64) for i in range(3)]
        ix, iy, iz = np.meshgrid(*axes, indexing="ij")
        cells = np.column_stack((ix.ravel(), iy.ravel(), iz.ravel()))
        if not len(cells):
            continue

        # Work in modest chunks: very long fibres should not allocate all
        # candidate cells times all quadrature points at once.
        for start in range(0, len(cells), 4096):
            cell_chunk = cells[start : start + 4096]
            sample_points = (
                cell_chunk[:, None, :].astype(np.float64) + offsets[None, :, :]
            ).reshape(-1, 3)
            distances, _ = _capsule_distance_and_normal(sample_points, p0, p1)
            local_fraction = (
                (distances <= fiber_radius).reshape(len(cell_chunk), -1).mean(axis=1)
            )
            active = local_fraction > 0.0
            if not np.any(active):
                continue
            active_cells = cell_chunk[active]
            active_fraction = local_fraction[active]
            cell_centres = active_cells.astype(np.float64)
            _, cell_normal = _capsule_distance_and_normal(cell_centres, p0, p1)
            fallback = np.linalg.norm(cell_normal, axis=1) <= 1.0e-14
            if np.any(fallback):
                # On a capsule axis, use the surface sample nearest to it.
                for row in np.flatnonzero(fallback):
                    points = active_cells[row][None, :].astype(np.float64) + offsets
                    distance, candidate_normal = _capsule_distance_and_normal(
                        points, p0, p1
                    )
                    cell_normal[row] = candidate_normal[np.argmax(distance)]

            wrapped = np.mod(active_cells, grid_size)
            # Generated short-fibre RVEs are non-overlapping.  ``maximum`` is
            # nevertheless safe for a rare coarse-grid collision and prevents
            # an unphysical volume fraction above one.
            old = fraction[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]]
            replace = active_fraction > old
            if np.any(replace):
                target = wrapped[replace]
                fraction[target[:, 0], target[:, 1], target[:, 2]] = active_fraction[
                    replace
                ]
                orientation[target[:, 0], target[:, 1], target[:, 2]] = (
                    direction.astype(np.float32)
                )
                normal[target[:, 0], target[:, 1], target[:, 2]] = cell_normal[
                    replace
                ].astype(np.float32)
    return fraction, orientation, normal


def _rotate_stiffness_mandel(stiffness: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Rotate one TI stiffness per axis, retaining a compact local dependency."""
    from cufhom.materials.elasticity import rotate_C_mandel, rotation_matrix_from_vector

    rounded = np.round(np.asarray(axes, dtype=np.float64) * 1.0e8).astype(np.int64)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    out = np.empty((len(axes), 6, 6), dtype=np.float64)
    for group in range(int(inverse.max()) + 1):
        members = np.flatnonzero(inverse == group)
        axis = np.asarray(axes[members[0]], dtype=np.float64)
        out[members] = rotate_C_mandel(stiffness, rotation_matrix_from_vector(axis))
    return out


def _strain_jump_map(normals: np.ndarray) -> np.ndarray:
    """Mandel map ``a -> sym(a outer n)`` for a batch of normals."""
    n0, n1, n2 = normals.T
    result = np.zeros((len(normals), 6, 3), dtype=np.float64)
    result[:, 0, 0] = n0
    result[:, 1, 1] = n1
    result[:, 2, 2] = n2
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    result[:, 3, 1] = n2 * inv_sqrt2
    result[:, 3, 2] = n1 * inv_sqrt2
    result[:, 4, 0] = n2 * inv_sqrt2
    result[:, 4, 2] = n0 * inv_sqrt2
    result[:, 5, 0] = n1 * inv_sqrt2
    result[:, 5, 1] = n0 * inv_sqrt2
    return result


def laminate_voxel_stiffness_mandel(
    matrix_stiffness: np.ndarray,
    fiber_stiffness_local: np.ndarray,
    fibre_fraction: np.ndarray,
    fibre_orientation: np.ndarray,
    interface_normal: np.ndarray,
    *,
    storage: str = "sym21",
    dtype: np.dtype | type = np.float32,
    pure_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Build a local stiffness field using exact two-phase laminate voxels.

    The interface normal is only used in cut cells.  For each such cell this
    solves traction continuity and displacement compatibility of a rank-one
    laminate exactly.  It is physically directional, unlike scalar Voigt
    interpolation, and supports anisotropic fibres.
    """
    cm = np.asarray(matrix_stiffness, dtype=np.float64)
    cf_local = np.asarray(fiber_stiffness_local, dtype=np.float64)
    fraction = np.asarray(fibre_fraction, dtype=np.float64)
    orientation = np.asarray(fibre_orientation, dtype=np.float64)
    normal = np.asarray(interface_normal, dtype=np.float64)
    if cm.shape != (6, 6) or cf_local.shape != (6, 6):
        raise ValueError(
            "matrix_stiffness and fiber_stiffness_local must be 6x6 Mandel tensors."
        )
    if (
        fraction.ndim != 3
        or orientation.shape != (*fraction.shape, 3)
        or normal.shape != orientation.shape
    ):
        raise ValueError(
            "fraction must be (Nx,Ny,Nz), orientation/normal must be (Nx,Ny,Nz,3)."
        )
    if (
        np.any(~np.isfinite(fraction))
        or np.any(fraction < -pure_tolerance)
        or np.any(fraction > 1.0 + pure_tolerance)
    ):
        raise ValueError("fibre_fraction must lie in [0, 1].")
    fraction = np.clip(fraction, 0.0, 1.0)

    shape = fraction.shape
    field = np.broadcast_to(cm, (*shape, 6, 6)).copy()
    fibre_mask = fraction >= 1.0 - pure_tolerance
    cut_mask = (fraction > pure_tolerance) & ~fibre_mask
    if np.any(fibre_mask):
        axes = orientation[fibre_mask]
        axis_norm = np.linalg.norm(axes, axis=1)
        if np.any(axis_norm <= 1.0e-14):
            raise ValueError("A fibre-containing voxel has a zero orientation.")
        field[fibre_mask] = _rotate_stiffness_mandel(
            cf_local, axes / axis_norm[:, None]
        )

    if np.any(cut_mask):
        f1 = fraction[cut_mask]
        f0 = 1.0 - f1
        axes = orientation[cut_mask]
        axis_norm = np.linalg.norm(axes, axis=1)
        if np.any(axis_norm <= 1.0e-14):
            raise ValueError("A composite voxel has a zero fibre orientation.")
        c1 = _rotate_stiffness_mandel(cf_local, axes / axis_norm[:, None])
        normals = normal[cut_mask]
        normal_norm = np.linalg.norm(normals, axis=1)
        # A normal may be unavailable for image-derived data.  The fibre axis
        # is a deterministic, mechanically admissible fallback (end-cap case).
        invalid = normal_norm <= 1.0e-14
        if np.any(invalid):
            normals[invalid] = axes[invalid] / axis_norm[invalid, None]
            normal_norm = np.linalg.norm(normals, axis=1)
        normals /= normal_norm[:, None]

        jump = _strain_jump_map(normals)
        # Virtual work in the orthonormal Mandel basis gives traction = jump.T.
        traction = np.swapaxes(jump, -1, -2)
        c0 = np.broadcast_to(cm, c1.shape)
        t0 = np.einsum("...ik,...kl,...lj->...ij", traction, c0, jump)
        t1 = np.einsum("...ik,...kl,...lj->...ij", traction, c1, jump)
        t_mix = f0[:, None, None] * t1 + f1[:, None, None] * t0
        rhs = np.einsum("...ik,...kl->...il", traction, c0 - c1)
        correction = np.linalg.solve(t_mix, rhs)
        local = (
            f0[:, None, None] * c0
            + f1[:, None, None] * c1
            + (f0 * f1)[:, None, None]
            * np.einsum("...ik,...kl,...lm->...im", c1 - c0, jump, correction)
        )
        # Round-off only: retain the reciprocal stiffness expected by the CG
        # formulation without changing the laminate solution.
        field[cut_mask] = 0.5 * (local + np.swapaxes(local, -1, -2))

    output_dtype = np.dtype(dtype)
    storage = str(storage).lower()
    if storage in {"full", "mandel"}:
        return np.moveaxis(field.astype(output_dtype, copy=False), (-2, -1), (0, 1))
    if storage not in {"sym21", "packed21", "symmetric21"}:
        raise ValueError("storage must be 'full' or 'sym21'.")
    packed = np.stack([field[..., i, j] for i, j in _SYM21_PAIRS], axis=0)
    return packed.astype(output_dtype, copy=False)
