"""Small independent Green-operator constructions for numerical tests."""

import numpy as np
from cufhom.fft.operators import IsotropicReference, MixedProjector
from cufhom_core.tensors import fft


def mandel_gradient(k):
    """Real sym(k tensor u), in production order 11,22,33,23,13,12."""
    x, y, z = k
    r = np.sqrt(2.0)
    return np.array(
        [
            [x, 0, 0],
            [0, y, 0],
            [0, 0, z],
            [0, z / r, y / r],
            [z / r, 0, x / r],
            [y / r, x / r, 0],
        ]
    )


def isotropic(K, mu):
    C = np.eye(6) * 2 * mu
    C[:3, :3] += K - 2 * mu / 3
    return C


def green(shape, xp=np, K=2.5, mu=1.2):
    P = MixedProjector(shape, (1.0, 1.0, 1.0), np.ones(6, bool), xp)
    G = IsotropicReference(P, K, mu)
    G.set_shift(0.0, xp.float64)
    return P, G


def convolution_kernel(n):
    """Only for small diagnostic grids: 36 FP64 scalars per voxel."""
    fft.set_fft_backend("scipy")
    _, G = green((n,) * 3)
    kernel = np.empty((6, 6, n, n, n))
    impulse = np.zeros((6, n, n, n))
    for j in range(6):
        impulse.fill(0)
        impulse[j, 0, 0, 0] = 1
        kernel[:, j] = G(impulse)
    return kernel, G


def interaction(kernel, target, source):
    n = kernel.shape[-1]
    d = (target[:, None, :] - source[None, :, :]) % n
    return (
        kernel[:, :, d[..., 0], d[..., 1], d[..., 2]]
        .transpose(2, 0, 3, 1)
        .reshape(6 * len(target), 6 * len(source))
    )
