"""FP64 capacity workflows preserve the discrete problem and audit tolerance."""

import json

import numpy as np
import pytest
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")


@pytest.mark.parametrize("shape", [(12, 10, 8), (9, 11, 13)])
def test_fp64_shared_fft_buffer_matches_ordinary_operator(shape):
    from cufhom_core.general.cupy_strain_capacity import supported
    from cufhom_core.tensors import fft

    from tests.fft.linear.test_resident_strain_capacity import operator

    old = fft.get_fft_backend()
    fft.set_fft_backend("cupy")
    try:
        A, B = operator(shape)
        B.val = B.val.astype(cp.float64)
        A.mat_rev[0][0].material_table = A.mat_rev[0][0].material_table.astype(
            cp.float64
        )
        saved = B.val.copy()
        reference = A(B).val
        assert supported(A, B)
        A._capacity_inplace_fft = True
        with cp.cuda.Stream(non_blocking=True) as stream:
            # The inputs above are produced on the default stream.
            cp.cuda.Stream.null.synchronize()
            value = A(B).val
            assert value.dtype == cp.float64 and value.flags.c_contiguous
            cp.testing.assert_allclose(value, reference, rtol=2e-13, atol=2e-12)
            cp.testing.assert_array_equal(B.val, saved)
        stream.synchronize()
    finally:
        fft.set_fft_backend(old)


@pytest.mark.parametrize("host_loads", [1, 6])
def test_fp64_streamed_energy_keeps_all_entries_and_deferred_means(host_loads):
    from cufhom_core.general.cupy_cfield_assembly import indexed_assembly_streamed
    from cufhom_core.postprocess import assembly_matrix_fused

    from tests.fft.linear.test_streamed_assembly import sample

    A, fields, arrays = sample()
    A.material_table = A.material_table.astype(cp.float64)
    arrays = arrays.astype(np.float64)
    for field, array in zip(fields, arrays):
        field.val = cp.asarray(array)
    means = np.arange(36, dtype=np.float64).reshape(6, 6) / 17
    for field, mean in zip(fields, means):
        field.val += cp.asarray(mean[:, None, None, None])
    reference = assembly_matrix_fused(A, fields)
    for k, (field, array, mean) in enumerate(zip(fields, arrays, means)):
        field.val = array.copy() if k < host_loads else cp.asarray(array)
        field.cufhom_deferred_mean = mean
    result = cp.asnumpy(indexed_assembly_streamed(A, fields, 37)) / np.prod(A.N)
    np.testing.assert_allclose(result, reference, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("actor", ["C", "Clib", "B"])
def test_fp64_capacity_six_load_audits_and_cpu_reference(monkeypatch, tmp_path, actor):
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    import cufhom_core.tensors.operators as operators
    from cufhom.fft.linear import coordinates
    from cufhom_core import applications
    from cufhom_core.general import cupy_capacity

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    p.update(
        solver_real_dtype="float64",
        solver_fp64_coordinates=True,
        solver_rtol=1e-9,
        require_convergence=True,
        solver_maxiter=1000,
    )
    reference = solve_homogenization(dict(p, fft_backend="scipy", solver_rtol=1e-12))
    monkeypatch.setattr(cupy_capacity, "library_capacity_required", lambda *args: True)
    monkeypatch.setattr(operators, "library_capacity_required", lambda *args: True)
    monkeypatch.setattr(coordinates, "library_capacity_required", lambda *args: True)
    monkeypatch.setattr(coordinates, "_host_audit_required", lambda *args: True)
    monkeypatch.setattr(applications, "host_iterate_required", lambda *args: True)
    monkeypatch.setattr(applications, "_needs_bounded_assembly", lambda *args: True)
    if actor == "B":
        monkeypatch.setattr(
            coordinates, "make_linear_solver", lambda **kw: kw["fallback"]
        )
    elif actor == "Clib":
        monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)
    path = tmp_path / "capacity.json"
    value = solve_homogenization(dict(p, solver_timing_path=str(path)))
    np.testing.assert_allclose(value, reference, rtol=5e-12, atol=5e-12)
    app = json.loads(path.read_text())["cufhom_core_application_timing"]
    assert app["solution_assembly"]["strategy"] == "host_fields_chunked_energy"
    infos = app["primaldual"]["primal"]["load_solver_infos"]
    assert len(infos) == 6 and all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-9 for i in infos
    )
    if actor == "B":
        assert all(i["resident_cg_iterate"] for i in infos)
    else:
        assert all(i["linear_fused_fft"] == (actor == "C") for i in infos)


def test_capacity_selectors_account_for_precision(monkeypatch):
    from cufhom_core.general import cupy_capacity as capacity

    monkeypatch.setattr(capacity, "_device_capacity", lambda device: 16 * 2**30)
    assert not capacity.library_capacity_required((384,) * 3, np.float32)
    assert capacity.library_capacity_required((384,) * 3, np.float64)
    assert not capacity.host_iterate_required((384,) * 3, np.float32)
    assert not capacity.host_iterate_required((448,) * 3, np.float32)
    assert capacity.host_iterate_required((448,) * 3, np.float64)


@pytest.mark.parametrize("initial", [False, True])
def test_fp64_host_iterate_preserves_recurrence_and_rhs(monkeypatch, initial):
    from cufhom_core.general import solver
    from cufhom_core.tensors import Tensor

    rng = np.random.default_rng(5741)
    shape = (4, 5, 6)
    raw = cp.asarray(rng.normal(size=(6,) + shape))
    diagonal = cp.asarray(rng.uniform(1, 9, size=raw.shape))
    B = Tensor(val=raw.copy(), order=1, N=shape, Fourier=False, fft_form="r")

    def action(x):
        return x.copy(val=diagonal * x.val)

    start = raw * 0.15 if initial else None
    par = dict(rtol=1e-10, maxiter=200, check_true_residual=True)
    reference, old = solver.CG(
        action, B, x0=B.copy(val=start.copy()) if initial else None, par=par
    )
    B.cufhom_host_cg = True
    monkeypatch.setattr(solver, "_HOST_CG_CHUNK", 37)
    value, info = solver.CG(
        action, B, x0=B.copy(val=start.copy()) if initial else None, par=par
    )
    assert info["host_cg_iterate"] and info["true_residual_converged"]
    assert info["kit"] == old["kit"]
    np.testing.assert_array_equal(info["norm_res_log"], old["norm_res_log"])
    cp.testing.assert_array_equal(value.val, reference.val)
    cp.testing.assert_array_equal(B.val, raw)
