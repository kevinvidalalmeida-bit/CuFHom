"""Odd grids on the fused pipeline: line-pair KC against the library route."""

import numpy as np
import pytest
from cufhom.fft.linear import evaluation, fused
from cufhom.fft.linear.framework import homogenize

from tests.fft.linear.test_linear_coordinates import cp
from tests.fft.linear.test_linear_coordinates import gpu as gpu  # autouse fixture

pytestmark = pytest.mark.gpu


def material(shape, seed=5):
    """Three random isotropic phases in blobs, so every Fourier mode is excited."""
    rng = np.random.default_rng(seed)
    index = (rng.random(shape) > 0.55).astype(np.int32) + (
        rng.random(shape) > 0.8
    ).astype(np.int32)
    table = np.zeros((3, 6, 6))
    for p, (K, G) in enumerate([(3.0, 1.2), (40.0, 25.0), (12.0, 6.0)]):
        table[p, :3, :3] = K - 2 * G / 3
        table[p] += np.diag([2 * G] * 6)
    return index, table


SUPPORTED = fused.grid_supported


def solve(index, table, fused_route, monkeypatch, **options):
    support = SUPPORTED if fused_route else (lambda *args, **kwargs: False)
    for module in (fused, evaluation):
        monkeypatch.setattr(module, "grid_supported", support)
    return homogenize(index, table, discretization="gani", rtol=1e-10, **options)


@pytest.mark.parametrize("shape", [(9, 7, 15), (15, 9, 21), (8, 6, 9)])
@pytest.mark.parametrize("working", [None, "float64"])
def test_odd_grids_match_the_library_route(shape, working, monkeypatch):
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    index, table = material(shape)
    a = solve(index, table, True, monkeypatch, working=working)
    b = solve(index, table, False, monkeypatch, working=working)
    assert (
        a["implementation"] == "CuFHom-GaNi-fused"
        and b["implementation"] == "CuFHom-GaNi-library"
    )
    np.testing.assert_allclose(
        a["C"], b["C"], rtol=1e-12, atol=1e-12 * np.abs(b["C"]).max()
    )


def test_odd_line_count_with_several_slabs():
    # n0 n1 odd: the last KC group holds one line; small slabs give several chunks.
    from types import SimpleNamespace

    from cufhom.fft.linear.common import IndexedMaterial
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom.fft.operators import MixedProjector

    shape = (15, 9, 21)
    index, table = material(shape, 7)
    data = IndexedMaterial(index, table, dtype="float32")
    P = MixedProjector(shape, (1.0, 1.0, 1.0), np.ones(6, bool), cp)
    cfg = fused.default_config(shape)
    Q = SymmetricCoordinates(P, 10.0, 5.0).tiled(cfg["ta"])
    descriptor = SimpleNamespace(
        val=cp.asarray(index), material_table=cp.asarray(data.packed, cp.float32)
    )
    pipelines = [
        fused.FusedPipeline(Q, descriptor, dict(cfg, slabs=s)) for s in (15, 8, 4)
    ]
    z = cp.asarray(
        np.random.default_rng(3).normal(size=Q.size * 2).astype(np.float32)
    ).view(cp.complex64)
    out = [cp.asnumpy(p.matvec(z)) for p in pipelines]
    for o in out[1:]:
        np.testing.assert_array_equal(o, out[0])
    assert np.isfinite(out[0]).all() and np.abs(out[0]).max() > 0
