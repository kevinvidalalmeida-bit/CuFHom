"""Newton in isometric displacement coordinates.

Q*Q = I and QQ* = P, so the residual Q*sigma has the norm of P sigma and is
the Newton right-hand side; corrections Q z are compatible because the strain
map makes the k2 = 0 plane Hermitian, as C2R transforms assume.
"""

import numpy as np
import pytest
from cufhom.fft.linear.krylov import SymmetricCoordinates
from cufhom.fft.operators import (
    DisplacementCoordinates,
    MixedProjector,
    Timings,
    hermitian_plane,
)
from cufhom_core.tensors import fft as core_fft

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
except Exception:
    cp = None


def _coordinates(shape, xp, control=(1, 0, 0, 0, 0, 0)):
    projector = MixedProjector(
        shape, (1.0, 1.3, 0.8), np.array(control, bool), xp, Timings(xp, False)
    )
    return projector, DisplacementCoordinates(projector)


def _mirror(plane):
    return np.roll(np.flip(plane, axis=(1, 2)), 1, axis=(1, 2)).conj()


@pytest.mark.parametrize("n2", [9, 8])
def test_hermitian_plane_numpy(n2):
    """k2 = 0 (and the Nyquist plane of even grids) become Hermitian; packed splits Q."""
    rng = np.random.default_rng(0)
    h = rng.normal(size=(6, 6, 5, 5)) + 1j * rng.normal(size=(6, 6, 5, 5))
    planes = (0, 4) if n2 % 2 == 0 else (0,)
    rest = h[..., 1:4].copy()
    hermitian_plane(h, n2)
    for p in planes:
        np.testing.assert_allclose(h[..., p], _mirror(h[..., p]))
    assert np.array_equal(h[..., 1:4], rest)
    assert np.allclose(hermitian_plane(h.copy(), n2), h)
    if n2 % 2 == 0:
        q = rng.normal(size=(6, 6, 5)) + 1j * rng.normal(size=(6, 6, 5))
        g = np.zeros((6, 6, 5, 5), complex)
        g[..., 0] = q
        hermitian_plane(g, n2, packed=True)
        np.testing.assert_allclose(g[..., 0] + 1j * g[..., 4], q)
        for p in planes:
            np.testing.assert_allclose(g[..., p], _mirror(g[..., p]))


@pytest.mark.gpu
@pytest.mark.parametrize("n2", [9, 8])
def test_hermitian_plane_cuda_matches_numpy(n2):
    rng = np.random.default_rng(0)
    for packed in (False, True):
        g = rng.normal(size=(6, 6, 5, 5)) + 1j * rng.normal(size=(6, 6, 5, 5))
        np.testing.assert_allclose(
            cp.asnumpy(hermitian_plane(cp.asarray(g), n2, packed)),
            hermitian_plane(g.copy(), n2, packed),
            atol=1e-14,
        )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 128, 128), (96, 128, 130)])
@pytest.mark.parametrize("tiled", [False, True])
def test_strain_of_any_coordinates_is_compatible(shape, tiled):
    """C2R of a non-Hermitian k2 = 0 plane was off by ~9% from 128^3 on."""
    core_fft.set_fft_backend("cupy")
    P = MixedProjector(shape, (1.0, 1.3, 0.8), np.ones(6, bool), cp, Timings(cp, False))
    Q = SymmetricCoordinates(P, 2.0, 1.0)
    if tiled:
        Q = Q.tiled(1)
    rng = np.random.default_rng(1)
    z = cp.asarray(rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size)).astype(
        cp.complex64
    )
    strain = Q.strain(z)
    assert float(cp.linalg.norm(P(strain) - strain) / cp.linalg.norm(strain)) < 2e-6
    if tiled:  # the fused CG direction update uses production vectors only
        return
    p = z.copy()
    direction = Q.strain_updated_direction(p, z, 0.5)
    assert (
        float(cp.linalg.norm(P(direction) - direction) / cp.linalg.norm(direction))
        < 2e-6
    )


@pytest.mark.gpu
@pytest.mark.skipif(cp is None, reason="CuPy unavailable")
@pytest.mark.parametrize(
    "control", [(1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 1, 0), (1,) * 6, (0,) * 6]
)
def test_coordinate_residual_is_isometric(control):
    core_fft.set_fft_backend("cupy")
    P, Q = _coordinates((24, 25, 26), cp, control)
    stress = cp.asarray(np.random.default_rng(2).normal(size=(6, 24, 25, 26)))
    residual = Q.adjoint(stress)
    assert float(cp.sum(residual.view(cp.float64) ** 2)) == pytest.approx(
        float(cp.sum(P(stress) ** 2)), rel=1e-12
    )
    assert float(cp.abs(Q.strain(residual) - P(stress)).max()) < 1e-12 * float(
        cp.abs(stress).max()
    )
    # A uniform prescribed stress moves only the free mean slots, by sqrt(N).
    uniform = cp.broadcast_to(cp.arange(1.0, 7.0).reshape(6, 1, 1, 1), stress.shape)
    shifted = Q.adjoint(cp.ascontiguousarray(uniform))
    assert float(cp.abs(shifted[: 3 * Q.count]).max()) < 1e-12
    free = P.free
    assert np.allclose(
        cp.asnumpy(shifted[3 * Q.count :].real),
        np.sqrt(24 * 25 * 26) * (np.arange(1.0, 7.0)[free]),
    )
