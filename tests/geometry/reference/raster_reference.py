"""Independent point sampling and exhaustive capsule raster for tests."""
import itertools
import numpy as np
from typing import Tuple

def flat_cylinder_interior(points, fiber, box):
    """Diagnostic only: strict membership in any periodic flat-cylinder image.

    All dimensions are physical. Does not modify SAM's capsule model. Testing
    finitely many sample points can prove intersection, never its absence.
    """
    import numpy as np
    c = fiber[:3] % box
    axis = fiber[3:6]
    half, radius = .5*fiber[6], .5*fiber[7]
    extent = half*np.abs(axis)+radius*np.sqrt(np.maximum(0., 1-axis**2))
    lo = np.ceil((-c-extent)/box).astype(int)
    hi = np.floor((box-c+extent)/box).astype(int)
    inside = np.zeros(len(points), dtype=bool)
    margin = 1e-10*max(1., box, fiber[6], fiber[7])
    for shift in itertools.product(*(range(a, b+1) for a, b in zip(lo, hi))):
        delta = points-c-box*np.array(shift)
        axial = delta@axis
        radial = delta-axial[:, None]*axis
        inside |= ((np.abs(axial) < half-margin)
                   & (np.einsum('ij,ij->i', radial, radial) < max(0., radius-margin)**2))
    return inside

def _empty_index_triplet() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.int64),
    )

def _fiber_voxel_indices_python(
    center: np.ndarray,
    direction: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    P0 = center - 0.5 * fiber_length * direction
    P1 = center + 0.5 * fiber_length * direction
    u_arr = P1 - P0
    uu = float(u_arr @ u_arr)
    if uu <= 1e-12:
        return _empty_index_triplet()

    lo = np.floor(np.minimum(P0, P1) - (fiber_radius + 1.0)).astype(int)
    hi = np.ceil(np.maximum(P0, P1) + (fiber_radius + 1.0)).astype(int)
    xs = np.arange(lo[0], hi[0] + 1, dtype=np.float64)
    ys = np.arange(lo[1], hi[1] + 1, dtype=np.float64)
    zs = np.arange(lo[2], hi[2] + 1, dtype=np.float64)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    t_r = np.clip(
        ((gx - P0[0]) * u_arr[0] + (gy - P0[1]) * u_arr[1] + (gz - P0[2]) * u_arr[2]) / uu,
        0.0,
        1.0,
    )
    dist2_r = (
        (gx - (P0[0] + t_r * u_arr[0])) ** 2
        + (gy - (P0[1] + t_r * u_arr[1])) ** 2
        + (gz - (P0[2] + t_r * u_arr[2])) ** 2
    )
    mask_r = dist2_r <= fiber_radius * fiber_radius
    if not np.any(mask_r):
        return _empty_index_triplet()

    ix_v = gx[mask_r].astype(int) % N
    iy_v = gy[mask_r].astype(int) % N
    iz_v = gz[mask_r].astype(int) % N
    return ix_v, iy_v, iz_v

def _unique_periodic_indices(indices, direction, length, radius, N):
    """One periodic voxel per fiber, including fibers longer than the cell."""
    if np.any(np.abs(direction) * length + 2.0 * radius >= N):
        ix, iy, iz = indices
        flat = np.unique((ix * N + iy) * N + iz)
        return flat // (N * N), (flat // N) % N, flat % N
    return indices

def _rasterize_fibers_python(
    centers: np.ndarray,
    directions: np.ndarray,
    fiber_length: float,
    fiber_radius: float,
    N: int,
    voxel_target: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    phase = np.zeros((N, N, N), dtype=np.uint8)
    ori = np.zeros((N, N, N, 3), dtype=np.float32)
    voxel_count = 0
    used_fibers = 0

    for c, a in zip(centers, directions):
        ix_v, iy_v, iz_v = _fiber_voxel_indices_python(
            c,
            a,
            fiber_length=fiber_length,
            fiber_radius=fiber_radius,
            N=N,
        )
        ix_v, iy_v, iz_v = _unique_periodic_indices(
            (ix_v, iy_v, iz_v), a, fiber_length, fiber_radius, N)
        if ix_v.size == 0:
            used_fibers += 1  # Retained prefix, not just nonempty rasterizations.
            continue

        new_voxels_mask = phase[ix_v, iy_v, iz_v] == 0
        candidate_count = voxel_count + int(np.count_nonzero(new_voxels_mask))
        if (
            used_fibers > 0
            and voxel_count < voxel_target <= candidate_count
            and voxel_target - voxel_count
            <= candidate_count - voxel_target
        ):
            break

        voxel_count = candidate_count
        phase[ix_v, iy_v, iz_v] = 1
        ori[ix_v, iy_v, iz_v] = a.astype(np.float32)
        used_fibers += 1

        if voxel_count >= voxel_target:
            break

    return phase, ori, voxel_count, used_fibers


def voxel_diagnostic(fibers, box, grid):
    import numpy as np
    scale = grid/box
    occupancy = np.zeros(grid**3, dtype=np.uint32)
    flat_occupancy = np.zeros(grid**3, dtype=np.uint32)
    owner = np.full(grid**3, -1, dtype=np.int32)
    for i, f in enumerate(fibers):
        ix, iy, iz = _fiber_voxel_indices_python(
            np.ascontiguousarray((f[:3] % box)*scale),
            np.ascontiguousarray(f[3:6]), f[6]*scale, .5*f[7]*scale, grid)
        flat = np.unique((ix*grid+iy)*grid+iz)
        occupancy[flat] += 1
        owner[flat] = i
        points = np.column_stack((flat//(grid*grid), (flat//grid) % grid, flat % grid))/scale
        interior = flat_cylinder_interior(points, f, box)
        flat_occupancy[flat[interior]] += 1
    used = occupancy > 0
    union = int(used.sum())
    owner = owner.reshape((grid,)*3)
    faces = 0
    for axis in range(3):
        other = np.roll(owner, 1, axis=axis)
        faces += int(((owner >= 0) & (other >= 0) & (owner != other)).sum())
    counts = np.bincount(owner[owner >= 0], minlength=len(fibers)).astype(float)
    a2 = np.einsum('n,ni,nj->ij', counts, fibers[:, 3:6], fibers[:, 3:6])/max(1, union)
    return dict(grid=grid, h_um=box/grid, vf=union/grid**3,
                multiply_occupied_sample_points=int((occupancy > 1).sum()),
                excess_memberships=int(occupancy.sum()-union),
                flat_cylinder_strict_shared_sample_points=int((flat_occupancy > 1).sum()),
                flat_cylinder_interior_sample_fraction=float((flat_occupancy > 0).sum()/grid**3),
                unlike_fiber_neighbor_faces=faces, A2_last_owner=a2.tolist(),
                note='Multiple capsule owners imply intersection (possibly boundary-only). '
                     'Strict flat-cylinder shared points prove interior intersection, '
                     'but zero points do not certify separation. Neighbor voxels do not prove overlap.')
