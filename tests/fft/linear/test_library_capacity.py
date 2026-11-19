"""Library routes retain their equations and audits under capacity pressure."""

import json

import numpy as np
import pytest
from cufhom_core.general import solver
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("chunk", [17, 250, 4096])
@pytest.mark.parametrize("initial", [False, True])
def test_host_strain_cg_preserves_rhs_and_resident_recurrence(
    monkeypatch, chunk, initial
):
    if cp is None:
        pytest.skip("CuPy unavailable")
    rng = np.random.default_rng(1953)
    shape = (4, 5, 6)
    raw = rng.normal(size=(6,) + shape).astype(np.float32)
    diagonal = cp.asarray(rng.uniform(1, 9, size=raw.shape).astype(np.float32))
    B = Tensor(val=cp.asarray(raw), order=1, N=shape, Fourier=False, fft_form="r")

    def action(x):
        return x.copy(val=diagonal * x.val)

    par = dict(rtol=1e-5, maxiter=200, check_true_residual=True)
    initial_value = (
        cp.asarray(rng.normal(size=raw.shape).astype(np.float32)) if initial else None
    )
    x0 = None if initial_value is None else B.copy(val=initial_value.copy())
    reference, old = solver.CG(action, B, x0=x0, par=par)
    B.cufhom_host_cg = True
    monkeypatch.setattr(solver, "_HOST_CG_CHUNK", chunk)
    x0 = None if initial_value is None else B.copy(val=initial_value.copy())
    result, info = solver.CG(action, B, x0=x0, par=par)
    assert info["host_cg_iterate"] and info["true_residual_converged"]
    assert info["kit"] == old["kit"]
    np.testing.assert_array_equal(info["norm_res_log"], old["norm_res_log"])
    cp.testing.assert_array_equal(result.val, reference.val)
    cp.testing.assert_array_equal(B.val, cp.asarray(raw))


def test_host_cg_restores_rhs_when_operator_raises(monkeypatch):
    if cp is None:
        pytest.skip("CuPy unavailable")
    shape = (4, 5, 6)
    raw = np.random.default_rng(973).normal(size=(6,) + shape).astype(np.float32)
    B = Tensor(val=cp.asarray(raw), order=1, N=shape, Fourier=False, fft_form="r")
    B.cufhom_host_cg = True
    weights = cp.linspace(1, 10, B.val.size, dtype=cp.float32).reshape(B.val.shape)
    calls = [0]

    def action(x):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("test operator failure after an update")
        return x.copy(val=weights * x.val)

    with pytest.raises(RuntimeError, match="test operator failure"):
        solver.CG(action, B, par=dict(rtol=1e-8, maxiter=100))
    assert calls[0] == 2
    cp.testing.assert_array_equal(B.val, cp.asarray(raw))


@pytest.mark.parametrize("actor", ["B", "Clib"])
def test_capacity_six_load_routes_match_cpu_reference(monkeypatch, tmp_path, actor):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import cufhom_core.general.cupy_capacity as capacity
    import cufhom_core.tensors.operators as operators
    from cufhom.fft.linear import coordinates
    from cufhom_core import applications

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    p.update(require_convergence=True, solver_maxiter=1000)
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    monkeypatch.setattr(
        operators, "library_capacity_required", lambda shape, dtype=np.float32: True
    )
    monkeypatch.setattr(
        capacity, "library_capacity_required", lambda shape, dtype=np.float32: True
    )
    if actor == "B":
        import cufhom_core.general.cupy_strain_capacity as resident

        monkeypatch.setattr(resident, "supported", lambda *a: False)
        monkeypatch.setattr(
            coordinates, "make_linear_solver", lambda **kw: kw["fallback"]
        )
        monkeypatch.setattr(
            applications, "host_iterate_required", lambda shape, dtype=np.float32: True
        )
    else:
        monkeypatch.setattr(coordinates, "fused_supported", lambda *a: False)
        monkeypatch.setattr(
            coordinates,
            "library_capacity_required",
            lambda shape, dtype=np.float32: True,
        )
    path = tmp_path / "capacity.json"
    q = dict(p, solver_timing_path=str(path), solution_field_return_in_memory=True)
    result = solve_homogenization(q)
    np.testing.assert_allclose(result, reference, rtol=3e-6, atol=3e-6)
    assert len(q["_solution_fields_result"]) == 6
    infos = json.loads(path.read_text())["cufhom_core_application_timing"][
        "primaldual"
    ]["primal"]["load_solver_infos"]
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )
    assert all(not i["linear_fused_fft"] for i in infos) if actor == "Clib" else True
    if actor == "B":
        assert all(i["host_cg_iterate"] for i in infos)


def test_component_library_ffts_preserve_normalization_and_owned_input(monkeypatch):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import cufhom_core.general.cupy_capacity as capacity
    from cufhom_core.tensors import fft

    old_backend = fft.get_fft_backend()
    old_mode = fft.CUPY_PLAN_MODE
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual")
    shape = (12, 10, 8)
    raw = np.random.default_rng(423).normal(size=(6,) + shape).astype(np.float32)
    device = cp.asarray(raw)
    try:
        full = fft.rfftn_cached(device, shape, (1, 2, 3))
        monkeypatch.setattr(
            capacity, "library_capacity_required", lambda n, dtype=np.float32: True
        )
        split = fft.rfftn_cached(device, shape, (1, 2, 3))
        cp.testing.assert_allclose(split, full, rtol=4e-6, atol=3e-5)
        saved = split.copy()
        inverse = fft.irfftn_cached(split, shape, (1, 2, 3))
        cp.testing.assert_array_equal(split, saved)
        cp.testing.assert_allclose(inverse, device, rtol=3e-6, atol=1e-6)
        raw_inverse = fft.irfftn_unscaled_cached(split.copy(), shape, (1, 2, 3))
        cp.testing.assert_allclose(
            raw_inverse / np.prod(shape), device, rtol=3e-6, atol=1e-6
        )
    finally:
        fft.set_fft_backend(old_backend)
        fft.set_cupy_plan_mode(old_mode)


def test_capacity_pool_separates_tiny_and_field_allocations_and_releases_cache():
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core.general.cupy_capacity import (
        capacity_memory_pool,
        current_memory_pool,
    )

    with capacity_memory_pool() as pool:
        assert current_memory_pool() is pool
        large = cp.zeros(1_048_576, dtype=cp.float32)
        tiny = cp.asarray([7.0], dtype=cp.float32)
        cp.cuda.get_current_stream().synchronize()
        assert len(pool.pools) >= 2
        assert pool.used_bytes() >= large.nbytes + tiny.nbytes
        del large
        pool.free_all_blocks()
        assert pool.total_bytes() < 1_048_576
        cp.testing.assert_array_equal(tiny, cp.asarray([7.0], dtype=cp.float32))
    assert current_memory_pool() is cp.get_default_memory_pool()
    cp.testing.assert_array_equal(tiny, cp.asarray([7.0], dtype=cp.float32))


def test_capacity_threshold_cache_is_device_specific_and_not_free_memory(monkeypatch):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import cufhom_core.general.cupy_capacity as capacity

    active = [0]
    queried = []

    def props(device):
        queried.append(device)
        return {"totalGlobalMem": (16 if device == 0 else 4) * 2**30}

    capacity._device_capacity.cache_clear()
    monkeypatch.setattr(cp.cuda.runtime, "getDevice", lambda: active[0])
    monkeypatch.setattr(cp.cuda.runtime, "getDeviceProperties", props)
    try:
        assert not capacity.library_capacity_required((320,) * 3)
        assert capacity.library_capacity_required((512,) * 3)
        active[0] = 1
        assert capacity.library_capacity_required((320,) * 3)
        active[0] = 0
        assert not capacity.host_iterate_required((320,) * 3)
        assert queried == [0, 1]
    finally:
        capacity._device_capacity.cache_clear()


def test_capacity_pool_outlives_process_cached_arrays():
    if cp is None:
        pytest.skip("CuPy unavailable")
    import gc

    from cufhom_core.general.cupy_capacity import capacity_memory_pool

    with capacity_memory_pool() as pool:
        identity = id(pool)
        cached = cp.arange(1000, dtype=cp.float32)
    del pool
    gc.collect()
    with capacity_memory_pool() as other:
        assert id(other) == identity
        scratch = cp.ones(2000, dtype=cp.float32)
        cp.testing.assert_array_equal(cached, cp.arange(1000, dtype=cp.float32))
        assert float(scratch.sum()) == 2000


def test_host_cg_zero_rhs_keeps_finite_solution_and_rhs():
    if cp is None:
        pytest.skip("CuPy unavailable")
    B = Tensor(
        val=cp.zeros((6, 4, 4, 4), cp.float32),
        order=1,
        N=(4, 4, 4),
        Fourier=False,
        fft_form="r",
    )
    B.cufhom_host_cg = True
    x, info = solver.CG(
        lambda x: x.copy(),
        B,
        par=dict(rtol=1e-5, maxiter=100, check_true_residual=True),
    )
    assert info["kit"] == 0 and info["true_residual_converged"]
    cp.testing.assert_array_equal(x.val, cp.zeros_like(B.val))
    cp.testing.assert_array_equal(B.val, cp.zeros_like(B.val))
