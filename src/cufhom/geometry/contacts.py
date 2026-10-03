"""Exact periodic contacts and collective A2/A4 relaxation."""
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize
from cufhom.geometry.audit import audit_periodic_fibers
from cufhom.geometry.orientation import resolve_orientation

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


def a2_objective(x, length, box, target, weight):
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
            last_value = a2_objective(x, length, domain, target, orientation_weight)
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


def rotational_inertia(length, diameter):
    """Length-squared scale, eq. 3.14; denominator must NOT be multiplied."""
    if not np.isfinite([length, diameter]).all() or min(length, diameter) <= 0:
        raise ValueError('Positive finite length and diameter required')
    a = length / diameter
    return diameter**2 * (a**3/12 + a*a/6 + 3*a/16 + 1/15) / (a + 2/3)


def isotropic_a4():
    eye = np.eye(3)
    return (np.einsum('ij,kl->ijkl', eye, eye)
            + np.einsum('ik,jl->ijkl', eye, eye)
            + np.einsum('il,jk->ijkl', eye, eye))/15


def fourth_moment(u):
    outer = np.einsum('ni,nj->nij', u, u)
    return np.einsum('nij,nkl->ijkl', outer, outer)/len(u)


def a4_error(u, target=None):
    target = isotropic_a4() if target is None else target
    return float(np.linalg.norm(fourth_moment(u)-target)/np.linalg.norm(target))


def sphere_exp(u, tangent):
    """Great-circle update; normalization only removes round-off."""
    tangent = tangent-u*np.sum(u*tangent, axis=1)[:, None]
    theta = np.linalg.norm(tangent, axis=1)
    out = np.cos(theta)[:, None]*u + np.sinc(theta/np.pi)[:, None]*tangent
    return out/np.linalg.norm(out, axis=1)[:, None]


@njit(cache=True)
def _near_segment_pairs(pairs, centers, u, length, box, reach):
    keep = np.zeros(len(pairs), dtype=np.bool_)
    for k in range(len(pairs)):
        i, j = pairs[k]
        dx,dy,dz=centers[j,0]-centers[i,0],centers[j,1]-centers[i,1],centers[j,2]-centers[i,2]
        dx-=np.rint(dx/box[0])*box[0]
        dy-=np.rint(dy/box[1])*box[1]
        dz-=np.rint(dz/box[2])*box[2]
        ax,ay,az=length*u[i,0],length*u[i,1],length*u[i,2]
        bx,by,bz=length*u[j,0],length*u[j,1],length*u[j,2]
        ex,ey,ez=.5*(abs(ax)+abs(bx))+reach+1e-12,.5*(abs(ay)+abs(by))+reach+1e-12,.5*(abs(az)+abs(bz))+reach+1e-12
        for sx in range(int(np.ceil((-ex-dx)/box[0])),int(np.floor((ex-dx)/box[0]))+1):
            if keep[k]: break
            for sy in range(int(np.ceil((-ey-dy)/box[1])),int(np.floor((ey-dy)/box[1]))+1):
                if keep[k]: break
                for sz in range(int(np.ceil((-ez-dz)/box[2])),int(np.floor((ez-dz)/box[2]))+1):
                    if _closest_components(dx+sx*box[0],dy+sy*box[1],dz+sz*box[2],ax,ay,az,bx,by,bz)[0] <= reach+1e-12:
                        keep[k] = True
                        break
    return pairs[keep]


class ContactNeighbors:
    """Single-level AABB/segment Verlet list, checked on EVERY a4_objective evaluation.

    Hausdorff endpoint-motion bound includes rotations and rejected line-search
    trials. No assumption about a fixed rebuild interval. Self images are always
    kept. Reference positions are copied, never views into mutable optimizer data.
    """
    def __init__(self, length, domain, skin=.5, contact_diameter=1.):
        if not np.isfinite(skin) or skin <= 0:
            raise ValueError('Positive finite skin required')
        self.length, self.domain, self.skin = length, domain.copy(), skin
        self.contact_diameter = contact_diameter
        self.centers = None
        self.rebuilds = 0

    def get(self, centers, u):
        rebuild = self.centers is None or len(centers) != len(self.centers)
        if not rebuild:
            movement = endpoint_motion_bound(centers, u, self.centers, self.u,
                                              self.domain, self.length)
            rebuild = 2*movement >= self.skin*(1-1e-12)
        if rebuild:
            pairs = build_aabb_pairs(centers, u, self.domain, self.length, self.contact_diameter, self.skin)
            pairs = _near_segment_pairs(pairs, centers, u, self.length, self.domain,
                                        self.contact_diameter+self.skin)
            n = len(centers)
            counts = np.bincount(pairs[:, 0], minlength=n)+1
            offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
            indices = np.empty(offsets[-1], dtype=np.int64)
            indices[offsets[:-1]] = np.arange(n)  # retain periodic self contacts
            if len(pairs):
                slots = np.arange(len(pairs))+pairs[:, 0]+1
                indices[slots] = pairs[:, 1]
            self.offsets, self.indices = offsets, indices
            self.centers, self.u = centers.copy(), u.copy()
            self.rebuilds += 1
        return self.offsets, self.indices


def a4_objective(centers, u, length, domain, weight=1., neighbors=None, target_a4=None,
              contact_diameter=1.):
    """Diameter=1 coordinates. Returns energy and Euclidean derivatives.

    E = E_contact + N*epsilon*weight/8 * ||A4-A4_target||_F**2.
    Dividing the orientation gradient by epsilon recovers the torque/A4
    balance in eq. (9) for equal lengths. The printed eq. (8) lacks N;
    its literal derivative with averaged A4 would be smaller by N. This
    a4_objective consistently matches the eq. (9) vector field, not eq. (8).
    """
    kwargs = {}
    if neighbors is not None:
        kwargs['neighbor_offsets'], kwargs['neighbor_indices'] = neighbors.get(centers, u)
    energy, gc, gu, maximum = contact_energy(centers, u, length, contact_diameter, domain, **kwargs)
    # contact_energy normalizes by collision diameter; restore physical-diameter
    # units so adding a gap does not also weaken the contact penalty.
    energy *= contact_diameter**2
    gc *= contact_diameter**2
    gu *= contact_diameter**2
    maximum *= contact_diameter  # violation relative to physical diameter=1
    eps = rotational_inertia(length, 1.)
    target_a4 = isotropic_a4() if target_a4 is None else target_a4
    residual = fourth_moment(u)-target_a4
    energy += len(u)*eps*weight/8*np.sum(residual*residual)
    gu += eps*weight*np.einsum('ijkl,nj,nk,nl->ni', residual, u, u, u)
    return float(energy), gc, gu, float(maximum), float(np.linalg.norm(residual)/np.linalg.norm(target_a4))


@njit(cache=True)
def build_aabb_pairs(centers, directions, box, length, diameter, skin):
    # Two passes avoid allocating an n(n-1)/2 buffer when the graph is sparse.
    n=len(centers); count=0
    pairs=np.empty((0,2),dtype=np.int32)
    reach=diameter+skin+1e-12*max(1.,length,diameter,np.max(box))
    for phase in range(2):
        cursor=0
        for i in range(n):
            for j in range(i+1,n):
                lower2=0.
                for axis in range(3):
                    delta=centers[j,axis]-centers[i,axis]
                    delta-=round(delta/box[axis])*box[axis]
                    gap=max(0.,abs(delta)-.5*length*(abs(directions[i,axis])+abs(directions[j,axis])))
                    lower2+=gap*gap
                if lower2<=reach*reach:
                    if phase==1:
                        pairs[cursor,0]=i; pairs[cursor,1]=j
                    cursor+=1
        if phase==0:
            count=cursor
            pairs=np.empty((count,2),dtype=np.int32)
    return pairs


@njit(cache=True)
def endpoint_motion_bound(centers, directions, ref_centers, ref_directions, box, length):
    """Hausdorff-motion upper bound of each rigid segment on a periodic torus."""
    maximum=0.
    for i in range(len(centers)):
        translation2=0.; rotation2=0.
        for axis in range(3):
            delta=centers[i,axis]-ref_centers[i,axis]
            delta-=round(delta/box[axis])*box[axis]
            translation2+=delta*delta
            rotation2+=(directions[i,axis]-ref_directions[i,axis])**2
        maximum=max(maximum,np.sqrt(translation2)+.5*length*np.sqrt(rotation2))
    return maximum


def normalized_coordinates(x):
    values = np.asarray(x).reshape(-1, 6)
    raw = values[:, 3:]
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(values).all() or not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError('Degenerate orientation coordinate')
    return np.ascontiguousarray(values[:, :3]), np.ascontiguousarray(raw/norms[:, None]), norms


def coordinate_gradient(gc, gu, u, norms):
    tangent = gu-u*np.sum(gu*u, axis=1)[:, None]
    return np.column_stack((gc, tangent/norms[:, None])).ravel()


def minimize_a4(c, u, evaluate, epsilon, target, *, maxiter, overlap_tolerance,
                a2_tolerance, a4_tolerance, progress=None):
    x0 = np.column_stack((c, np.sqrt(epsilon)*u)).ravel()
    last_x = None
    last_value = None
    evaluations = 0
    history = []
    best_score, best_c, best_u = np.inf, c.copy(), u.copy()
    lowest_energy, energy_c, energy_u = np.inf, c.copy(), u.copy()
    energy_x = x0.copy()
    iterations = 0

    def fun(x):
        nonlocal last_x, last_value, evaluations
        if last_x is not None and np.array_equal(x, last_x):
            return last_value[0], last_value[1]
        try:
            cc, uu, norms = normalized_coordinates(x)
        except ValueError:
            return np.inf, np.zeros_like(x)
        energy, gc, gu, overlap, a4 = evaluate(cc, uu)
        evaluations += 1
        g = coordinate_gradient(gc, gu, uu, norms)
        last_x, last_value = x.copy(), (energy, g, overlap, a4)
        return energy, g

    def record(x, final=False):
        nonlocal best_score, best_c, best_u, lowest_energy, energy_c, energy_u, energy_x
        fun(x)
        cc, uu, norms = normalized_coordinates(x)
        energy, _, overlap, a4 = last_value
        a2 = float(np.linalg.norm(uu.T@uu/len(uu)-target)/np.linalg.norm(target))
        score = max(overlap/overlap_tolerance, a2/a2_tolerance, a4/a4_tolerance)
        if score < best_score:
            best_score, best_c, best_u = score, cc.copy(), uu.copy()
        if energy < lowest_energy:
            lowest_energy, energy_c, energy_u = energy, cc.copy(), uu.copy()
            energy_x = x.copy()
        if iterations % 25 == 0 or score <= 1 or final:
            row = dict(iteration=iterations, energy=energy, overlap_relative=overlap,
                       A2_error=a2, A4_error=a4, evaluations=evaluations,
                       axis_coordinate_radius_min=float(norms.min()),
                       axis_coordinate_radius_max=float(norms.max()))
            if not history or row != history[-1]:
                history.append(row)
                if progress:
                    progress(row.copy())
        return score

    def callback(x):
        nonlocal iterations
        iterations += 1
        if record(x) <= 1:
            raise StopIteration

    score = record(x0)
    message = 'constraints_satisfied'
    gradient_inf_norm = float(np.max(np.abs(last_value[1])))
    recoverable = False
    fallback_step = None
    if score > 1:
        result = minimize(fun, x0, jac=True, method='L-BFGS-B', callback=callback,
                          options=dict(maxiter=maxiter, maxfun=32*(maxiter+1),
                                       maxls=30, maxcor=12, ftol=0., gtol=1e-12))
        record(result.x, final=True)
        message = 'constraints_satisfied' if best_score <= 1 else str(result.message)
        gradient_inf_norm = float(np.max(np.abs(last_value[1])))
        # Floating-point relative-reduction and line-search stops are not
        # stationary-point certificates. Clear the quasi-Newton history on a
        # subsequent solve when a significant gradient and budget remain.
        numerical_stop = ('RELATIVE REDUCTION' in message or 'ABNORMAL' in message)
        if best_score > 1 and numerical_stop and gradient_inf_norm > 1e-10 and iterations < maxiter:
            # Verify actual descent before requesting another chunk. For almost
            # parallel rods a full angular step may cross a sharp contact ridge;
            # a center-only step leaves A2/A4 unchanged and can still remove
            # contacts. Both alternatives must decrease the same a4_objective.
            base = energy_x.copy()
            base_energy, base_gradient = fun(base)
            minimum_change = 8*np.finfo(float).eps*max(1.,abs(base_energy))
            for kind in ('full_gradient','centers'):
                direction = -base_gradient.copy()
                if kind == 'centers':
                    direction.reshape(-1,6)[:,3:] = 0.
                slope = float(base_gradient@direction)
                if slope >= -1e-30:
                    continue
                step = .3
                for _ in range(32):
                    trial = base+step*direction
                    value,_ = fun(trial)
                    if (value < lowest_energy-minimum_change
                            and value <= base_energy+1e-4*step*slope):
                        iterations += 1
                        record(trial,final=True)
                        recoverable = best_score > 1
                        fallback_step = kind
                        break
                    step *= .5
                if fallback_step:
                    break
            message = 'constraints_satisfied' if best_score <= 1 else message
            gradient_inf_norm = float(np.max(np.abs(fun(energy_x)[1])))
    returned_state = 'best_feasible' if best_score <= 1 else 'lowest_energy_infeasible'
    if best_score > 1:
        # A migration path can reduce total energy while its largest overlap
        # temporarily grows. Returning the smallest-max-overlap state here
        # would undo that progress before the caller's continuation chunk.
        best_c, best_u = energy_c, energy_u
    return best_c, best_u, dict(history=history, iterations=iterations,
                                optimizer_message=message,
                                returned_state=returned_state, lowest_energy=float(lowest_energy),
                                final_gradient_inf_norm=gradient_inf_norm,
                                recoverable_stop=bool(recoverable),fallback_step=fallback_step)


def relax_a4(fibers, box, target, *, maxiter=300, orientation_weight=1.,
             overlap_tolerance=1e-6, a2_tolerance=.035, a4_tolerance=1e-4,
             progress=None, use_neighbors=True, target_a4=None, min_gap=0.,
             neighbor_skin=.5):
    """Return feasible geometry or lowest-energy progress without changing input."""
    f = np.array(fibers, dtype=float, copy=True)
    before = audit_periodic_fibers(f, box, overlap_tolerance)
    target, target_a4, orientation = resolve_orientation(target, target_a4)
    if not np.allclose(f[:, 6:], f[0, 6:]):
        raise ValueError('Equal lengths and diameters required')
    if (not np.isfinite([orientation_weight, overlap_tolerance, a2_tolerance, a4_tolerance]).all()
            or min(orientation_weight, overlap_tolerance, a2_tolerance, a4_tolerance) <= 0
            or maxiter < 1):
        raise ValueError('Positive finite solver settings required')
    if not np.isfinite(min_gap) or min_gap < 0:
        raise ValueError('Minimum gap must be finite and nonnegative')
    diameter = f[0, 7]
    contact_diameter = 1+min_gap/diameter
    length = f[0, 6]/diameter
    domain = np.broadcast_to(np.asarray(box, dtype=float)/diameter, (3,)).copy()
    c = np.ascontiguousarray(f[:, :3]/diameter)
    u = np.ascontiguousarray(f[:, 3:6])
    neighbors = (ContactNeighbors(length, domain, skin=neighbor_skin,
                                 contact_diameter=contact_diameter) if use_neighbors else None)
    evaluations = 0

    def evaluate(c, u):
        nonlocal evaluations
        evaluations += 1
        return a4_objective(c, u, length, domain, orientation_weight, neighbors,
                         target_a4, contact_diameter)

    best_c, best_u, result = minimize_a4(
        c, u, evaluate, rotational_inertia(length, 1.), target, maxiter=maxiter,
        overlap_tolerance=overlap_tolerance, a2_tolerance=a2_tolerance,
        a4_tolerance=a4_tolerance, progress=progress)
    f[:, :3] = np.mod(best_c*diameter, np.asarray(box))
    f[:, 3:6] = best_u
    after = audit_periodic_fibers(f, box, overlap_tolerance)
    inflated = f.copy()
    inflated[:, 7] += min_gap
    gap_audit = (audit_periodic_fibers(inflated, box, overlap_tolerance/contact_diameter)
                 if min_gap else after)
    a2 = float(np.linalg.norm(best_u.T@best_u/len(f)-target)/np.linalg.norm(target))
    a4 = a4_error(best_u, target_a4)
    accepted = bool(after['accepted'] and gap_audit['accepted']
                    and a2 <= a2_tolerance and a4 <= a4_tolerance)
    return f, dict(before=before, after=after, A2_error=a2, A4_error=a4,
                   contact_and_A2_accepted=bool(after['accepted'] and a2 <= a2_tolerance),
                   contact_A2_A4_accepted=accepted, vf_not_certified=True,
                   optimizer_success=accepted, optimizer_message=result['optimizer_message'],
                   iterations=result['iterations'], evaluations=evaluations, history=result['history'],
                   neighbor_rebuilds=neighbors.rebuilds if neighbors is not None else 0,
                   orientation_norm='full_tensor_frobenius', orientation_target=orientation,
                   gap_audit=gap_audit, min_gap=min_gap, neighbor_skin=neighbor_skin,
                   returned_state=result['returned_state'], recoverable_stop=result['recoverable_stop'],
                   final_gradient_inf_norm=result['final_gradient_inf_norm'],
                   fallback_step=result['fallback_step'], solver='inertia_A4_LBFGS')
