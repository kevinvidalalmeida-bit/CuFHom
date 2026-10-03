"""Continuous capsule contacts and A2 relaxation.

Exact periodic pair/self-image energy, analytic position/orientation gradients,
and L-BFGS line search. No voxels, fiber removal, diameter changes or FFT solves.
The independent geometric audit, not optimizer.success, decides feasibility.
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize
from .sam_periodic_audit import audit_periodic_fibers
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
def _closest_components(dx, dy, dz, ux, uy, uz, vx, vy, vz):
    """Segment minimizer with scalar temporaries; same five active-set cases."""
    wx, wy, wz = -.5*ux+.5*vx-dx, -.5*uy+.5*vy-dy, -.5*uz+.5*vz-dz
    a, b, c = ux*ux+uy*uy+uz*uz, ux*vx+uy*vy+uz*vz, vx*vx+vy*vy+vz*vz
    d, e = ux*wx+uy*wy+uz*wz, vx*wx+vy*wy+vz*wz
    det = a*c-b*b
    best, s, t = np.inf, 0., 0.
    for k in range(5):
        if k == 0:
            sk, tk = 0., min(1., max(0., e/c))
        elif k == 1:
            sk, tk = 1., min(1., max(0., (e+b)/c))
        elif k == 2:
            sk, tk = min(1., max(0., -d/a)), 0.
        elif k == 3:
            sk, tk = min(1., max(0., (b-d)/a)), 1.
        elif det > 1e-12*a*c:
            sk, tk = (b*e-c*d)/det, (a*e-b*d)/det
        else:
            continue
        if 0 <= sk <= 1 and 0 <= tk <= 1:
            rx, ry, rz = wx+sk*ux-tk*vx, wy+sk*uy-tk*vy, wz+sk*uz-tk*vz
            val = rx*rx+ry*ry+rz*rz
            if val < best:
                best, s, t = val, sk, tk
    return math.sqrt(max(0., best)), s, t, wx+s*ux-t*vx, wy+s*uy-t*vy, wz+s*uz-t*vz


@njit(cache=True)
def _closest(delta, u, v):
    dist, s, t, x, y, z = _closest_components(delta[0],delta[1],delta[2],
                                             u[0],u[1],u[2],v[0],v[1],v[2])
    return dist, s, t, np.array([x,y,z])


@njit(cache=True)
def contact_energy(centers, directions, length, diameter, box, first_new=0, segments_per_fiber=1,
                   conservative_cull=True, neighbor_offsets=None, neighbor_indices=None):
    """Unnormalized energy and gradients wrt centers and unit directions.

Each unordered inter-fiber image contact occurs once. Self images use one
lexicographic half of the lattice, so +/- shifts are not counted twice.
    At intersecting axes the distance is nonsmooth; use a separating subgradient.
"""
    n = len(centers)
    gc = np.zeros_like(centers); gu = np.zeros_like(directions)
    energy, maximum = 0., 0.
    for i in range(n):
        start = max(i, first_new)
        entries = n-start if neighbor_offsets is None else neighbor_offsets[i+1]-neighbor_offsets[i]
        for entry in range(entries):
            j = start+entry if neighbor_offsets is None else neighbor_indices[neighbor_offsets[i]+entry]
            if j < start:
                continue
            if conservative_cull:
                # Reject impossible contacts BEFORE creating temporary arrays.
                # Nearest-center displacement is a componentwise lower bound
                # over all periodic images; narrow phase still enumerates them.
                distance_lower_sq = 0.
                for axis in range(3):
                    dx = centers[j, axis]-centers[i, axis]
                    dx -= np.rint(dx/box[axis])*box[axis]
                    gap = max(0., abs(dx)-.5*length*(abs(directions[i, axis])+abs(directions[j, axis])))
                    distance_lower_sq += gap*gap
                if distance_lower_sq > (diameter+1e-12*max(1., length, diameter))**2:
                    continue
            dx,dy,dz = centers[j,0]-centers[i,0],centers[j,1]-centers[i,1],centers[j,2]-centers[i,2]
            dx-=np.rint(dx/box[0])*box[0]
            dy-=np.rint(dy/box[1])*box[1]
            dz-=np.rint(dz/box[2])*box[2]
            ux,uy,uz = length*directions[i,0],length*directions[i,1],length*directions[i,2]
            vx,vy,vz = length*directions[j,0],length*directions[j,1],length*directions[j,2]
            ex,ey,ez = .5*(abs(ux)+abs(vx)),.5*(abs(uy)+abs(vy)),.5*(abs(uz)+abs(vz))
            reach=diameter+1e-12
            for sx in range(int(np.ceil((-ex-reach-dx)/box[0])),int(np.floor((ex+reach-dx)/box[0]))+1):
                for sy in range(int(np.ceil((-ey-reach-dy)/box[1])),int(np.floor((ey+reach-dy)/box[1]))+1):
                    for sz in range(int(np.ceil((-ez-reach-dz)/box[2])),int(np.floor((ez+reach-dz)/box[2]))+1):
                        if i == j and not (sx > 0 or (sx == 0 and sy > 0)
                                           or (sx == 0 and sy == 0 and sz > 0)):
                            continue
                        ox,oy,oz = dx+box[0]*sx,dy+box[1]*sy,dz+box[2]*sz
                        # Connected neighbors naturally share their end cap in
                        # a polygonal fiber. Exempt ONLY the original image;
                        # periodic copies and nonadjacent self contacts remain.
                        if (segments_per_fiber > 1 and i//segments_per_fiber == j//segments_per_fiber
                                and j-i <= 1
                                and max(abs(ox-(centers[j,0]-centers[i,0])),
                                        abs(oy-(centers[j,1]-centers[i,1])),
                                        abs(oz-(centers[j,2]-centers[i,2]))) < 1e-9):
                            continue
                        gx,gy,gz = max(0.,abs(ox)-ex),max(0.,abs(oy)-ey),max(0.,abs(oz)-ez)
                        if gx*gx+gy*gy+gz*gz > diameter**2:
                            continue
                        dist,s,t,rx,ry,rz = _closest_components(ox,oy,oz,ux,uy,uz,vx,vy,vz)
                        penetration = diameter-dist
                        if penetration <= 0:
                            continue
                        maximum = max(maximum, penetration/diameter)
                        energy += .5*(penetration/diameter)**2
                        if dist > 1e-10:
                            scale=-penetration/diameter**2/dist
                            gx,gy,gz=scale*rx,scale*ry,scale*rz
                        else:
                            nx,ny,nz = _contact_normal(rx,ry,rz,dist,directions,i,j)
                            scale=penetration/diameter**2
                            gx,gy,gz=scale*nx,scale*ny,scale*nz
                        for axis in range(3):
                            g=gx if axis==0 else gy if axis==1 else gz
                            gc[i,axis]+=g
                            gc[j,axis]-=g
                            gu[i,axis]+=(s-.5)*length*g
                            gu[j,axis]-=(t-.5)*length*g
    return energy, gc, gu, maximum


@njit(cache=True)
def candidate_contact_energies(centers, directions, candidate_centers, candidate_directions,
                               length, diameter, box):
    """Score independent straight-fiber additions with one reusable geometry.

    Reuse the exact contact kernel, including all periodic and self images.
    Candidates are alternatives, so they must never interact with one another.
    Existing fibers are copied once rather than once per candidate.
    """
    n = len(centers)
    trial_c = np.empty((n+1, 3), dtype=np.float64)
    trial_u = np.empty((n+1, 3), dtype=np.float64)
    trial_c[:n] = centers
    trial_u[:n] = directions
    energies = np.empty(len(candidate_centers), dtype=np.float64)
    for at in range(len(candidate_centers)):
        trial_c[n] = candidate_centers[at]
        trial_u[n] = candidate_directions[at]
        energies[at] = contact_energy(trial_c, trial_u, length, diameter, box,
                                      first_new=n)[0]
    return energies


def objective(x, length, box, target, weight):
    """Diameter-scaled coordinates, orientation metric set by half-length."""
    values = np.asarray(x).reshape(-1, 6)
    raw = values[:, 3:]
    norms = np.maximum(np.linalg.norm(raw, axis=1), 1e-12)
    directions = np.ascontiguousarray(raw/norms[:, None])
    energy, gc, gu, maximum = contact_energy(
        np.ascontiguousarray(values[:, :3]), directions, length, 1., box)
    n = len(values)
    error = directions.T @ directions/n-target
    denom = float(np.sum(target*target))
    a2_error = float(np.linalg.norm(error)/np.sqrt(denom))
    energy = energy/n + .5*weight*a2_error**2
    gu = gu/n + (2*weight/(n*denom))*(directions @ error)
    gr = (gu-directions*np.sum(gu*directions, axis=1)[:, None])/norms[:, None]
    return energy, np.column_stack((gc/n, gr)).ravel(), maximum, a2_error


def relax_contact_energy(fibers, box, target, *, maxiter=300,
                         orientation_weight=1., overlap_tolerance=1/120,
                         a2_tolerance=.035, progress=None):
    """Return a NEW table and diagnostics; inputs and historical data untouched."""
    f = np.array(fibers, dtype=float, copy=True)
    # Validation, including unit orientations, also checks the initial state.
    before = audit_periodic_fibers(f, box, overlap_tolerance)
    if not (np.allclose(f[:, 6], f[0, 6]) and np.allclose(f[:, 7], f[0, 7])):
        raise ValueError('Equal lengths and diameters required')
    target = np.asarray(target, dtype=float)
    if (target.shape != (3, 3) or not np.isfinite(target).all()
            or not np.allclose(target, target.T) or not np.isclose(np.trace(target), 1.)
            or np.linalg.eigvalsh(target).min() < -1e-12):
        raise ValueError('Expected symmetric PSD, trace-one A2')
    if maxiter < 1 or orientation_weight < 0 or overlap_tolerance <= 0 or a2_tolerance <= 0:
        raise ValueError('Invalid optimization settings')
    diameter = f[0, 7]; length = f[0, 6]/diameter
    domain = np.broadcast_to(np.asarray(box, dtype=float)/diameter, (3,)).copy()
    x0 = np.column_stack((f[:, :3]/diameter, .5*length*f[:, 3:6])).ravel()
    history = []
    best = [np.inf, x0.copy()]
    last_x = last_value = None
    objective_evaluations = 0

    def evaluate(x):
        nonlocal last_x, last_value, objective_evaluations
        if last_x is None or not np.array_equal(x, last_x):
            last_value = objective(x, length, domain, target, orientation_weight)
            last_x = x.copy()
            objective_evaluations += 1
        return last_value

    def record(x):
        energy, _, overlap, a2 = evaluate(x)
        score = max(overlap/overlap_tolerance, a2/a2_tolerance)
        if score < best[0]:
            best[:] = [score, x.copy()]
        history.append(dict(iteration=len(history), energy=float(energy),
                            overlap_relative=float(overlap), A2_error=float(a2)))
        if progress and (len(history) == 1 or (len(history)-1) % 25 == 0):
            progress(history[-1])

    record(x0)
    result = minimize(lambda x: evaluate(x)[:2], x0, jac=True, method='L-BFGS-B',
                      callback=record, options=dict(maxiter=maxiter, maxls=30,
                                                   ftol=1e-14, gtol=1e-9))
    record(result.x)
    chosen = best[1].reshape(-1, 6)
    f[:, :3] = np.mod(chosen[:, :3]*diameter, np.asarray(box))
    f[:, 3:6] = chosen[:, 3:]/np.linalg.norm(chosen[:, 3:], axis=1)[:, None]
    after = audit_periodic_fibers(f, box, overlap_tolerance)
    a2 = float(np.linalg.norm(f[:, 3:6].T@f[:, 3:6]/len(f)-target)/np.linalg.norm(target))
    return f, dict(before=before, after=after, A2_error=a2,
                   contact_and_A2_accepted=bool(after['accepted'] and a2 <= a2_tolerance),
                   vf_not_certified=True, optimizer_success=bool(result.success),
                   optimizer_message=str(result.message), iterations=int(result.nit),
                   evaluations=int(result.nfev), objective_evaluations=objective_evaluations,
                   history=history)
