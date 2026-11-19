"""The experimental restricted Green must match the full periodic operator."""

import numpy as np
import pytest
from cufhom_core.tensors import fft

from tests.fft.linear.green_reference import (
    convolution_kernel,
    interaction,
    isotropic,
    mandel_gradient,
)


@pytest.fixture(autouse=True)
def restore_fft():
    old = fft.get_fft_backend_status()
    yield
    fft.set_fft_backend(old["requested"])
    fft.set_cupy_plan_mode(old["cupy_plan_mode"])


@pytest.mark.parametrize("n", [5, 6])
def test_restricted_convolution_matches_full_field_and_reciprocity(n):
    kernel, G = convolution_kernel(n)
    # Include the periodic boundary and a nonsymmetric ordering of sites.
    sites = np.array([[0, 0, 0], [n - 1, 1, 0], [2, 0, n - 1], [1, 2, 3]])
    matrix = interaction(kernel, sites, sites)
    rng = np.random.default_rng(n)
    source = rng.normal(size=(len(sites), 6))
    full = np.zeros((6, n, n, n))
    full[:, sites[:, 0], sites[:, 1], sites[:, 2]] = source.T
    expected = G(full)[:, sites[:, 0], sites[:, 1], sites[:, 2]].T.ravel()
    np.testing.assert_allclose(
        matrix @ source.ravel(), expected, rtol=2e-13, atol=1e-14
    )
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-13, atol=1e-14)
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-13


@pytest.mark.parametrize(
    "direction", [(1.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-2.0, 1.0, 0.2)]
)
def test_symbol_satisfies_compatible_equilibrium(direction):
    B = mandel_gradient(np.array(direction) / np.linalg.norm(direction))
    C = isotropic(2.5, 1.2)
    gamma = B @ np.linalg.solve(B.T @ C @ B, B.T)
    # A compatible strain's stress is mapped back to that strain; orthogonal
    # stresses with no divergence contribute exactly zero compatible strain.
    np.testing.assert_allclose(gamma @ C @ B, B, rtol=2e-13, atol=2e-14)
    assert np.linalg.matrix_rank(gamma, tol=1e-12) == 3
    _, _, Vh = np.linalg.svd(B.T, full_matrices=True)
    np.testing.assert_allclose(gamma @ Vh[3:].T, 0.0, atol=2e-14)
