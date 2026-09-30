"""SAM contact kernels enumerating all possibly touching periodic images.

Image ranges follow from segment bounding boxes. The exact same narrow-phase
segment distance and force model as SAMLite is evaluated on each survivor.
Unlike sam_acceleration_kernels, this corrects the nearest-center approximation.
"""
import math
import numpy as np
from numba import njit, prange
from .sam_acceleration import separated
Array = np.ndarray

@njit(inline='always')
def _contact_normal(diffx, diffy, diffz, dist, directions, i, j):
    if dist > 1e-10:
        return -diffx/dist, -diffy/dist, -diffz/dist
    # The closest-point direction is undefined for intersecting axes. Select
    # one normal in canonical pair order so the two CSR rows are antisymmetric.
    ii, jj = np.int64(i), np.int64(j)
    lo, hi = min(ii, jj), max(ii, jj)
    x, y, z = directions[lo, 0], directions[lo, 1], directions[lo, 2]
    vx, vy, vz = directions[hi, 0], directions[hi, 1], directions[hi, 2]
    nx, ny, nz = y*vz-z*vy, z*vx-x*vz, x*vy-y*vx
    norm = math.sqrt(nx*nx+ny*ny+nz*nz)
    if norm < 1e-10:
        if abs(x) <= abs(y) and abs(x) <= abs(z):
            nx, ny, nz = 0., -z, y
        elif abs(y) <= abs(z):
            nx, ny, nz = -z, 0., x
        else:
            nx, ny, nz = -y, x, 0.
        norm = math.sqrt(nx*nx+ny*ny+nz*nz)
    if i > j:
        norm = -norm
    return nx/norm, ny/norm, nz/norm

@njit(cache=True)
def _check_candidate_serial_numba(
    cand_cx: float, cand_cy: float, cand_cz: float,
    cand_dx: float, cand_dy: float, cand_dz: float,
    centers: Array, directions: Array, n: int,
    Lx: float, Ly: float, Lz: float,
    L: float, hard_d: float, cull_r_sq: float,
    skip_idx: int = -1,
) -> bool:
    SMALL = 1e-12
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz
    ux = cand_dx * L
    uy = cand_dy * L
    uz = cand_dz * L

    for i in range(n):
        if i == skip_idx:
            continue
        ddx = centers[i, 0] - cand_cx
        ddy = centers[i, 1] - cand_cy
        ddz = centers[i, 2] - cand_cz
        ddx -= round(ddx * inv_Lx) * Lx
        ddy -= round(ddy * inv_Ly) * Ly
        ddz -= round(ddz * inv_Lz) * Lz
        bx, by, bz = ddx, ddy, ddz
        margin = 1e-12 * max(1., L, hard_d, Lx, Ly, Lz)
        ex = .5*L*(abs(cand_dx)+abs(directions[i, 0]))+hard_d+margin
        ey = .5*L*(abs(cand_dy)+abs(directions[i, 1]))+hard_d+margin
        ez = .5*L*(abs(cand_dz)+abs(directions[i, 2]))+hard_d+margin
        for sx in range(int(math.ceil((-ex-bx)/Lx)), int(math.floor((ex-bx)/Lx))+1):
            for sy in range(int(math.ceil((-ey-by)/Ly)), int(math.floor((ey-by)/Ly))+1):
                for sz in range(int(math.ceil((-ez-bz)/Lz)), int(math.floor((ez-bz)/Lz))+1):
                    ddx, ddy, ddz = bx+sx*Lx, by+sy*Ly, bz+sz*Lz
                    if ddx*ddx + ddy*ddy + ddz*ddz > cull_r_sq:
                        continue

                    if separated(ddx, ddy, ddz, cand_dx, cand_dy, cand_dz, directions[i, 0], directions[i, 1], directions[i, 2], L, hard_d):
                        continue

                    vx = directions[i, 0] * L
                    vy = directions[i, 1] * L
                    vz = directions[i, 2] * L

                    ox = cand_cx + ddx
                    oy = cand_cy + ddy
                    oz = cand_cz + ddz

                    w0x = (cand_cx - 0.5*ux) - (ox - 0.5*vx)
                    w0y = (cand_cy - 0.5*uy) - (oy - 0.5*vy)
                    w0z = (cand_cz - 0.5*uz) - (oz - 0.5*vz)

                    a = ux*ux + uy*uy + uz*uz
                    b = ux*vx + uy*vy + uz*vz
                    c = vx*vx + vy*vy + vz*vz
                    d = ux*w0x + uy*w0y + uz*w0z
                    e = vx*w0x + vy*w0y + vz*w0z
                    Det = a*c - b*b

                    if Det < SMALL:
                        sN = 0.0
                        sD = 1.0
                        tN = e
                        tD = c
                    else:
                        sN = b*e - c*d
                        tN = a*e - b*d
                        sD = Det
                        tD = Det
                        if sN < 0.0:
                            sN = 0.0
                            tN = e
                            tD = c
                        elif sN > sD:
                            sN = sD
                            tN = e + b
                            tD = c

                    if tN < 0.0:
                        tN = 0.0
                        sN = max(0.0, min(-d, a))
                        sD = a
                    elif tN > tD:
                        tN = tD
                        sN = max(0.0, min(-d + b, a))
                        sD = a

                    sc = 0.0 if abs(sN) < SMALL else sN / sD
                    tc = 0.0 if abs(tN) < SMALL else tN / tD

                    dfx = w0x + sc*ux - tc*vx
                    dfy = w0y + sc*uy - tc*vy
                    dfz = w0z + sc*uz - tc*vz
                    dist = math.sqrt(dfx*dfx + dfy*dfy + dfz*dfz)

                    if dist < hard_d:
                        return True

    return False

@njit(parallel=True, cache=True)
def _check_candidates_against_existing_numba(
    candidate_centers: Array,
    candidate_directions: Array,
    centers: Array,
    directions: Array,
    n_existing: int,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    hard_d: float,
    cull_r_sq: float,
) -> Array:
    n_candidates = candidate_centers.shape[0]
    overlaps = np.zeros(n_candidates, dtype=np.uint8)
    for k in prange(n_candidates):
        cc = candidate_centers[k]
        dd = candidate_directions[k]
        overlap = _check_candidate_serial_numba(
            cc[0], cc[1], cc[2],
            dd[0], dd[1], dd[2],
            centers, directions, n_existing,
            Lx, Ly, Lz,
            L, hard_d, cull_r_sq, -1,
        )
        overlaps[k] = 1 if overlap else 0
    return overlaps

@njit(cache=True)
def _apply_pair_forces_numba(
    pairs: Array, centers: Array, directions: Array,
    center_updates: Array, rot_updates: Array,
    Lx: float, Ly: float, Lz: float,
    L: float, d_eff: float,
    pair_cull_sq: float,
    effective_tau: float, tol_overlap: float,
    tau_rotate: float, eps_inv: float
) -> float:
    M = len(pairs)
    max_overlap = 0.0
    SMALL = 1e-12
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    for k in range(M):
        i = pairs[k, 0]
        j = pairs[k, 1]

        c1x, c1y, c1z = centers[i, 0], centers[i, 1], centers[i, 2]
        c2x, c2y, c2z = centers[j, 0], centers[j, 1], centers[j, 2]

        dx = c2x - c1x
        dy = c2y - c1y
        dz = c2z - c1z

        # Periodic wrap
        dx -= round(dx * inv_Lx) * Lx
        dy -= round(dy * inv_Ly) * Ly
        dz -= round(dz * inv_Lz) * Lz
        bx, by, bz = dx, dy, dz
        margin = 1e-12 * max(1., L, d_eff, Lx, Ly, Lz)
        ex = .5*L*(abs(directions[i, 0])+abs(directions[j, 0]))+d_eff+margin
        ey = .5*L*(abs(directions[i, 1])+abs(directions[j, 1]))+d_eff+margin
        ez = .5*L*(abs(directions[i, 2])+abs(directions[j, 2]))+d_eff+margin
        for sx in range(int(math.ceil((-ex-bx)/Lx)), int(math.floor((ex-bx)/Lx))+1):
            for sy in range(int(math.ceil((-ey-by)/Ly)), int(math.floor((ey-by)/Ly))+1):
                for sz in range(int(math.ceil((-ez-bz)/Lz)), int(math.floor((ez-bz)/Lz))+1):
                    dx, dy, dz = bx+sx*Lx, by+sy*Ly, bz+sz*Lz
                    if dx*dx + dy*dy + dz*dz > pair_cull_sq:
                        continue

                    c2img_x = c1x + dx
                    c2img_y = c1y + dy
                    c2img_z = c1z + dz

                    d1x, d1y, d1z = directions[i, 0], directions[i, 1], directions[i, 2]
                    d2x, d2y, d2z = directions[j, 0], directions[j, 1], directions[j, 2]

                    if separated(dx, dy, dz, d1x, d1y, d1z, d2x, d2y, d2z, L, d_eff):
                        continue

                    ux, uy, uz = d1x * L, d1y * L, d1z * L
                    vx, vy, vz = d2x * L, d2y * L, d2z * L

                    p0x = c1x - 0.5 * ux
                    p0y = c1y - 0.5 * uy
                    p0z = c1z - 0.5 * uz

                    q0x = c2img_x - 0.5 * vx
                    q0y = c2img_y - 0.5 * vy
                    q0z = c2img_z - 0.5 * vz

                    w0x = p0x - q0x
                    w0y = p0y - q0y
                    w0z = p0z - q0z

                    a = ux*ux + uy*uy + uz*uz
                    b = ux*vx + uy*vy + uz*vz
                    c = vx*vx + vy*vy + vz*vz
                    d = ux*w0x + uy*w0y + uz*w0z
                    e = vx*w0x + vy*w0y + vz*w0z

                    Det = a*c - b*b
                    if Det < SMALL:
                        sN = 0.0
                        sD = 1.0
                        tN = e
                        tD = c
                    else:
                        sN = b*e - c*d
                        tN = a*e - b*d
                        sD = Det
                        tD = Det
                        if sN < 0.0:
                            sN = 0.0
                            tN = e
                            tD = c
                        elif sN > sD:
                            sN = sD
                            tN = e + b
                            tD = c

                    if tN < 0.0:
                        tN = 0.0
                        sN = max(0.0, min(-d, a))
                        sD = a
                    elif tN > tD:
                        tN = tD
                        md = -d + b
                        sN = max(0.0, min(md, a))
                        sD = a

                    sc = 0.0 if abs(sN) < SMALL else sN / sD
                    tc = 0.0 if abs(tN) < SMALL else tN / tD

                    diffx = w0x + sc*ux - tc*vx
                    diffy = w0y + sc*uy - tc*vy
                    diffz = w0z + sc*uz - tc*vz

                    dist = math.sqrt(diffx*diffx + diffy*diffy + diffz*diffz)
                    delta = d_eff - dist

                    if delta > 0.0:
                        if delta > max_overlap:
                            max_overlap = delta

                        uvec_x, uvec_y, uvec_z = _contact_normal(
                            diffx, diffy, diffz, dist, directions, i, j)

                        pmag = max(
                            min(max(effective_tau, 1e-6), 0.501) * delta,
                            min(0.5 * tol_overlap, 0.45 * delta),
                        )

                        pushx = pmag * uvec_x
                        pushy = pmag * uvec_y
                        pushz = pmag * uvec_z

                        center_updates[i, 0] -= pushx
                        center_updates[i, 1] -= pushy
                        center_updates[i, 2] -= pushz

                        center_updates[j, 0] += pushx
                        center_updates[j, 1] += pushy
                        center_updates[j, 2] += pushz

                        si = 2.0 * sc - 1.0
                        sj = 2.0 * tc - 1.0

                        raw_ix = (L * si / 2.0) * delta * (-uvec_x)
                        raw_iy = (L * si / 2.0) * delta * (-uvec_y)
                        raw_iz = (L * si / 2.0) * delta * (-uvec_z)

                        dot_i = d1x*raw_ix + d1y*raw_iy + d1z*raw_iz
                        proj_ix = raw_ix - dot_i * d1x
                        proj_iy = raw_iy - dot_i * d1y
                        proj_iz = raw_iz - dot_i * d1z

                        rot_updates[i, 0] += tau_rotate * proj_ix * eps_inv
                        rot_updates[i, 1] += tau_rotate * proj_iy * eps_inv
                        rot_updates[i, 2] += tau_rotate * proj_iz * eps_inv

                        raw_jx = (L * sj / 2.0) * delta * uvec_x
                        raw_jy = (L * sj / 2.0) * delta * uvec_y
                        raw_jz = (L * sj / 2.0) * delta * uvec_z

                        dot_j = d2x*raw_jx + d2y*raw_jy + d2z*raw_jz
                        proj_jx = raw_jx - dot_j * d2x
                        proj_jy = raw_jy - dot_j * d2y
                        proj_jz = raw_jz - dot_j * d2z

                        rot_updates[j, 0] += tau_rotate * proj_jx * eps_inv
                        rot_updates[j, 1] += tau_rotate * proj_jy * eps_inv
                        rot_updates[j, 2] += tau_rotate * proj_jz * eps_inv

    return max_overlap

@njit(parallel=True, cache=True)
def _apply_forces_per_fiber_numba(
    centers: Array,
    directions: Array,
    neighbor_offsets: Array,
    neighbor_indices: Array,
    center_updates: Array,
    rot_updates: Array,
    max_overlap_per_fiber: Array,
    Lx: float,
    Ly: float,
    Lz: float,
    L: float,
    d_eff: float,
    pair_cull_sq: float,
    effective_tau: float,
    tol_overlap: float,
    tau_rotate: float,
    eps_inv: float,
) -> None:
    """Jacobi contact migration with race-free parallel accumulation by fiber."""
    n = centers.shape[0]
    SMALL = 1e-12
    move_gain = min(max(effective_tau, 1e-6), 0.501)
    inv_Lx = 1.0 / Lx
    inv_Ly = 1.0 / Ly
    inv_Lz = 1.0 / Lz

    for i in prange(n):
        c1x = centers[i, 0]
        c1y = centers[i, 1]
        c1z = centers[i, 2]
        d1x = directions[i, 0]
        d1y = directions[i, 1]
        d1z = directions[i, 2]
        ux = d1x * L
        uy = d1y * L
        uz = d1z * L
        p0x = c1x - 0.5 * ux
        p0y = c1y - 0.5 * uy
        p0z = c1z - 0.5 * uz

        update_x = 0.0
        update_y = 0.0
        update_z = 0.0
        rotate_x = 0.0
        rotate_y = 0.0
        rotate_z = 0.0
        local_max_overlap = 0.0

        for neighbor_pos in range(neighbor_offsets[i], neighbor_offsets[i + 1]):
            j = neighbor_indices[neighbor_pos]

            dx = centers[j, 0] - c1x
            dy = centers[j, 1] - c1y
            dz = centers[j, 2] - c1z
            dx -= round(dx * inv_Lx) * Lx
            dy -= round(dy * inv_Ly) * Ly
            dz -= round(dz * inv_Lz) * Lz
            bx, by, bz = dx, dy, dz
            margin = 1e-12 * max(1., L, d_eff, Lx, Ly, Lz)
            ex = .5*L*(abs(d1x)+abs(directions[j, 0]))+d_eff+margin
            ey = .5*L*(abs(d1y)+abs(directions[j, 1]))+d_eff+margin
            ez = .5*L*(abs(d1z)+abs(directions[j, 2]))+d_eff+margin
            for sx in range(int(math.ceil((-ex-bx)/Lx)), int(math.floor((ex-bx)/Lx))+1):
                for sy in range(int(math.ceil((-ey-by)/Ly)), int(math.floor((ey-by)/Ly))+1):
                    for sz in range(int(math.ceil((-ez-bz)/Lz)), int(math.floor((ez-bz)/Lz))+1):
                        dx, dy, dz = bx+sx*Lx, by+sy*Ly, bz+sz*Lz
                        if dx * dx + dy * dy + dz * dz > pair_cull_sq:
                            continue

                        d2x = directions[j, 0]
                        d2y = directions[j, 1]
                        d2z = directions[j, 2]
                        if separated(dx, dy, dz, d1x, d1y, d1z, d2x, d2y, d2z, L, d_eff):
                            continue

                        vx = d2x * L
                        vy = d2y * L
                        vz = d2z * L
                        q0x = c1x + dx - 0.5 * vx
                        q0y = c1y + dy - 0.5 * vy
                        q0z = c1z + dz - 0.5 * vz

                        w0x = p0x - q0x
                        w0y = p0y - q0y
                        w0z = p0z - q0z
                        a = ux * ux + uy * uy + uz * uz
                        b = ux * vx + uy * vy + uz * vz
                        c = vx * vx + vy * vy + vz * vz
                        dd = ux * w0x + uy * w0y + uz * w0z
                        e = vx * w0x + vy * w0y + vz * w0z
                        det = a * c - b * b

                        if det < SMALL:
                            sN = 0.0
                            sD = 1.0
                            tN = e
                            tD = c
                        else:
                            sN = b * e - c * dd
                            tN = a * e - b * dd
                            sD = det
                            tD = det
                            if sN < 0.0:
                                sN = 0.0
                                tN = e
                                tD = c
                            elif sN > sD:
                                sN = sD
                                tN = e + b
                                tD = c

                        if tN < 0.0:
                            tN = 0.0
                            sN = max(0.0, min(-dd, a))
                            sD = a
                        elif tN > tD:
                            tN = tD
                            sN = max(0.0, min(-dd + b, a))
                            sD = a

                        sc = 0.0 if abs(sN) < SMALL else sN / sD
                        tc = 0.0 if abs(tN) < SMALL else tN / tD
                        diffx = w0x + sc * ux - tc * vx
                        diffy = w0y + sc * uy - tc * vy
                        diffz = w0z + sc * uz - tc * vz
                        dist = math.sqrt(diffx * diffx + diffy * diffy + diffz * diffz)
                        delta = d_eff - dist
                        if delta <= 0.0:
                            continue

                        if delta > local_max_overlap:
                            local_max_overlap = delta

                        uvec_x, uvec_y, uvec_z = _contact_normal(
                            diffx, diffy, diffz, dist, directions, i, j)

                        pmag = max(
                            move_gain * delta,
                            min(0.5 * tol_overlap, 0.45 * delta),
                        )
                        update_x -= pmag * uvec_x
                        update_y -= pmag * uvec_y
                        update_z -= pmag * uvec_z

                        si = 2.0 * sc - 1.0
                        raw_x = (L * si / 2.0) * delta * (-uvec_x)
                        raw_y = (L * si / 2.0) * delta * (-uvec_y)
                        raw_z = (L * si / 2.0) * delta * (-uvec_z)
                        dot_i = d1x * raw_x + d1y * raw_y + d1z * raw_z
                        rotate_x += tau_rotate * (raw_x - dot_i * d1x) * eps_inv
                        rotate_y += tau_rotate * (raw_y - dot_i * d1y) * eps_inv
                        rotate_z += tau_rotate * (raw_z - dot_i * d1z) * eps_inv

        center_updates[i, 0] = update_x
        center_updates[i, 1] = update_y
        center_updates[i, 2] = update_z
        rot_updates[i, 0] = rotate_x
        rot_updates[i, 1] = rotate_y
        rot_updates[i, 2] = rotate_z
        max_overlap_per_fiber[i] = local_max_overlap
