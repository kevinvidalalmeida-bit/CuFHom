"""Physical volumes and periodic point sampling of straight cylinders and capsules."""
import numpy as np
from numba import njit


def fiber_volume(length, diameter, shape='capsule'):
    if shape not in ('capsule', 'cylinder'):
        raise ValueError('Shape must be capsule or cylinder')
    if not np.isfinite([length, diameter]).all() or min(length, diameter) <= 0:
        raise ValueError('Positive finite fiber dimensions required')
    volume = np.pi*diameter**2*length/4
    return volume + (np.pi*diameter**3/6 if shape == 'capsule' else 0.)


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
        i, j = (k+1) % 3, (k+2) % 3
        half = length/2
        extent = half*abs(u[k])+radius*np.sqrt(max(0., 1-u[k]*u[k]))
        if capsule:
            extent = half*abs(u[k])+radius
        ri = radius*np.sqrt(1+(u[i]/u[k])**2)
        rj = radius*np.sqrt(1+(u[j]/u[k])**2)
        for xk in range(int(np.floor(c[k]-extent)), int(np.ceil(c[k]+extent))+1):
            dk = xk-c[k]
            ci, cj = c[i]+dk*u[i]/u[k], c[j]+dk*u[j]/u[k]
            for xi in range(int(np.floor(ci-ri)), int(np.ceil(ci+ri))+1):
                di = xi-c[i]
                for xj in range(int(np.floor(cj-rj)), int(np.ceil(cj+rj))+1):
                    dj = xj-c[j]
                    axial = dk*u[k]+di*u[i]+dj*u[j]
                    if not capsule and abs(axial) >= half:
                        continue
                    if capsule:
                        axial = max(-half, min(half, axial))
                    radial = (dk-axial*u[k])**2+(di-axial*u[i])**2+(dj-axial*u[j])**2
                    if (radial > radius*radius if capsule else radial >= radius*radius):
                        continue
                    if k == 0:
                        x, y, z = xk % n, xi % n, xj % n
                    elif k == 1:
                        x, y, z = xj % n, xk % n, xi % n
                    else:
                        x, y, z = xi % n, xj % n, xk % n
                    if phase[x,y,z] == 0:
                        count += 1
                    phase[x,y,z] = 1
                    ori[x,y,z] = u
    return count


def rasterize_fibers(centers, directions, length, radius, grid, shape='capsule'):
    """Rasterize every fiber; coordinates and dimensions are in voxel units."""
    c = np.ascontiguousarray(centers, dtype=float)
    u = np.ascontiguousarray(directions, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3 or u.shape != c.shape:
        raise ValueError('Centers and directions must have shape (n, 3)')
    if (not np.isfinite(c).all() or not np.isfinite(u).all()
            or not np.allclose(np.linalg.norm(u, axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError('Finite centers and unit directions required')
    fiber_volume(length, 2*radius, shape)
    if isinstance(grid, bool) or not isinstance(grid, (int, np.integer)) or grid < 2:
        raise ValueError('Integer grid >= 2 required')
    phase = np.zeros((grid, grid, grid), dtype=np.uint8)
    ori = np.zeros((grid, grid, grid, 3), dtype=np.float32)
    count = _paint_fibers(c, u, float(length), float(radius), phase, ori, shape == 'capsule')
    return phase, ori, count, len(c)


def rasterize_cylinders(centers, directions, length, radius, grid):
    return rasterize_fibers(centers, directions, length, radius, grid, 'cylinder')


def rasterize_capsules(centers, directions, length, radius, grid):
    return rasterize_fibers(centers, directions, length, radius, grid, 'capsule')
