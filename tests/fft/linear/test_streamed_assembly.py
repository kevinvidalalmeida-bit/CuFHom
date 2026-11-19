"""Bounded assembly must retain all 36 energy entries and the physical solve."""

from types import SimpleNamespace

import numpy as np
import pytest
from cufhom_core.general.cupy_cfield_assembly import indexed_assembly_streamed
from cufhom_core.postprocess import assembly_matrix_fused
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def fixed_solve_route(monkeypatch):
    # These paired solves isolate retention/assembly to FP64 roundoff. A
    # background compile can switch the second solve from library to fused
    # FP32 iterations, changing the fields before assembly is even compared.
    # Deferred compilation has its own tests in test_linear_fused_fft.py.
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")


def sample():
    rng = np.random.default_rng(5231)
    shape = (7, 5, 6)
    raw = rng.normal(size=(3, 6, 6))
    stiffness = np.einsum("mki,mkj->mij", raw, raw) + np.eye(6)
    i, j = np.triu_indices(6)
    table = stiffness[:, i, j].astype(np.float32)
    ids = rng.integers(0, 3, size=shape, dtype=np.int32)
    A = Tensor(
        name="material",
        val=cp.asarray(ids)[None],
        order=1,
        N=shape,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    A.material_table = cp.ascontiguousarray(cp.asarray(table))
    arrays = rng.normal(size=(6, 6) + shape).astype(np.float32)
    fields = [
        Tensor(
            name=str(k),
            val=cp.asarray(a),
            order=1,
            N=shape,
            Fourier=False,
            fft_form="r",
        )
        for k, a in enumerate(arrays)
    ]
    return A, fields, arrays


@pytest.mark.parametrize("chunk", [1, 37, 210, 400])
@pytest.mark.parametrize("host_loads", [1, 6])
def test_chunked_energy_matches_full_gpu_and_fp64_reference(chunk, host_loads):
    if cp is None:
        pytest.skip("CuPy unavailable")
    A, fields, arrays = sample()
    reference = assembly_matrix_fused(A, fields)
    for k in range(host_loads):
        fields[k].val = arrays[k].copy()
    # Includes a partial last chunk, mixed host/device fields and six host fields.
    result = cp.asnumpy(indexed_assembly_streamed(A, fields, chunk)) / np.prod(A.N)
    np.testing.assert_allclose(result, reference, rtol=2e-13, atol=2e-13)
    automatic = assembly_matrix_fused(A, fields)
    np.testing.assert_allclose(automatic, reference, rtol=2e-13, atol=2e-13)
    for k in range(host_loads):
        np.testing.assert_array_equal(fields[k].val, arrays[k])


def test_chunked_rejects_invalid_material_index():
    if cp is None:
        pytest.skip("CuPy unavailable")
    A, fields, arrays = sample()
    fields[0].val = arrays[0]
    A.val.ravel()[57] = -1
    with pytest.raises(ValueError, match="Nonfinite indexed assembly"):
        assembly_matrix_fused(A, fields)


def test_bounded_restore_reconstructs_each_load_exactly():
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core.applications import _compress_solution, _prepare_bounded_assembly

    A, fields, arrays = sample()

    class Reconstruction:
        def strain(self, z):
            return z * 2

    for field, value in zip(fields, arrays):
        field.cufhom_coordinates = (Reconstruction(), cp.asarray(value / 2))
        assert _compress_solution(field, np.zeros(6))
    info = _prepare_bounded_assembly(fields)
    assert info["retained_host_bytes"] == arrays.nbytes
    assert all(isinstance(f.val, np.ndarray) for f in fields)
    for field, value in zip(fields, arrays):
        np.testing.assert_array_equal(field.val, value)
        assert not hasattr(field, "cufhom_coordinates")


def test_energy_only_selector_protects_requested_field_outputs(monkeypatch):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core import applications as apps

    A, fields, _ = sample()
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (0, 16 * 2**30))
    # Force an unavailable pool budget without making an actual large allocation.
    monkeypatch.setattr(
        cp,
        "get_default_memory_pool",
        lambda: SimpleNamespace(total_bytes=lambda: 0, used_bytes=lambda: 0),
    )
    conf = dict(store_solution_fields=False, postprocess_assembly="fused")
    assert apps._needs_bounded_assembly(SimpleNamespace(solve=conf), A, fields)
    conf["store_solution_fields"] = True
    assert not apps._needs_bounded_assembly(SimpleNamespace(solve=conf), A, fields)


def test_six_load_streaming_matches_resident_fields_and_cpu(monkeypatch, tmp_path):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import json

    from cufhom_core import applications as apps

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    p.update(solver_rtol=1e-5, require_convergence=True)
    resident = solve_homogenization(dict(p))
    monkeypatch.setattr(apps, "_needs_bounded_assembly", lambda *a: True)
    timing = tmp_path / "stream.json"
    streamed = solve_homogenization(dict(p, solver_timing_path=str(timing)))
    record = json.loads(timing.read_text())["cufhom_core_application_timing"]
    assert record["solution_assembly"]["strategy"] == "host_fields_chunked_energy"
    infos = record["primaldual"]["primal"]["load_solver_infos"]
    assert len(infos) == 6
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )
    np.testing.assert_allclose(streamed, resident, rtol=2e-13, atol=2e-13)
    # CPU reference uses the same physical problem, different arithmetic/solver.
    monkeypatch.undo()
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    np.testing.assert_allclose(streamed, reference, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("n3", [650, 651])
def test_host_spectral_residual_crosses_chunk_and_frequency_boundaries(n3):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom.fft.linear.coordinates import _spectral_distance

    shape = (13, 170, n3)
    hshape = (6, 13, 170, n3 // 2 + 1)
    rng = np.random.default_rng(5710)
    a = (rng.normal(size=hshape) + 1j * rng.normal(size=hshape)).astype(np.complex64)
    b = (rng.normal(size=hshape) + 1j * rng.normal(size=hshape)).astype(np.complex64)
    weights = np.full(hshape[-1], 2.0)
    weights[0] = 1
    if n3 % 2 == 0:
        weights[-1] = 1
    diff = (a - b).astype(np.complex128)
    expected = np.sum((diff.real**2 + diff.imag**2) * weights)
    device_b = cp.asarray(b)
    host_result = _spectral_distance(a, shape, device_b)
    device_result = _spectral_distance(cp.asarray(a), shape, device_b)
    np.testing.assert_allclose([host_result, device_result], expected, rtol=2e-14)


@pytest.mark.parametrize("shape", [(6, 8, 10), (5, 7, 9)])
def test_inplace_independent_projector_preserves_all_frequency_values(shape):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core.projections import elasticity_direct
    from cufhom_core.tensors.fft import get_fft_backend, set_fft_backend
    from cufhom_core.tensors.objects import elasticity_direct_array

    previous = get_fft_backend()
    set_fft_backend("cupy")
    try:
        projector = elasticity_direct(
            shape, (1.0, 1.3, 0.9), fft_form="r", dtype=np.float32
        )
    finally:
        set_fft_backend(previous)
    rng = np.random.default_rng(347)
    hshape = (6,) + shape[:-1] + (shape[-1] // 2 + 1,)
    values = cp.asarray(
        (rng.normal(size=hshape) + 1j * rng.normal(size=hshape)).astype(np.complex64)
    )
    expected = elasticity_direct_array(projector, values)
    result = elasticity_direct_array(projector, values, out=values)
    assert result.data.ptr == values.data.ptr
    cp.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("return_fields", [False, True])
def test_host_audit_retains_six_load_accuracy_and_field_output(
    monkeypatch, tmp_path, return_fields
):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import json

    from cufhom.fft.linear import coordinates

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    resident = solve_homogenization(dict(p))
    monkeypatch.setattr(
        coordinates, "_host_audit_required", lambda shape, dtype=np.float32: True
    )
    timing = tmp_path / "host-audit.json"
    q = dict(
        p, solver_timing_path=str(timing), solution_field_return_in_memory=return_fields
    )
    result = solve_homogenization(q)
    report = json.loads(timing.read_text())
    infos = report["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    assert all(
        i["physical_rhs_host"]
        and i["true_residual_converged"]
        and i["true_norm_res_rel"] <= 1e-5
        for i in infos
    )
    np.testing.assert_allclose(result, resident, atol=2e-13, rtol=2e-13)
    if return_fields:
        assert len(q["_solution_fields_result"]) == 6
        assert all(np.isfinite(f).all() for f in q["_solution_fields_result"])


@pytest.mark.parametrize("return_fields", [False, True])
def test_direct_host_retention_skips_restoration_and_preserves_outputs(
    monkeypatch, tmp_path, return_fields
):
    if cp is None:
        pytest.skip("CuPy unavailable")
    import json

    from cufhom.fft.linear import coordinates
    from cufhom_core import applications

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    resident = solve_homogenization(dict(p))
    policy = coordinates._retain_host_solution
    monkeypatch.setattr(
        coordinates,
        "_retain_host_solution",
        lambda pb, shape, dtype=np.float32: policy(pb, (512, 512, 512), dtype),
    )
    monkeypatch.setattr(
        coordinates, "_host_audit_required", lambda shape, dtype=np.float32: True
    )
    if not return_fields:

        def no_restoration(*args):
            raise AssertionError("Accepted host fields should never be restored")

        monkeypatch.setattr(
            applications, "_restore_offloaded_solutions", no_restoration
        )
    timing = tmp_path / "retained.json"
    q = dict(
        p, solver_timing_path=str(timing), solution_field_return_in_memory=return_fields
    )
    result = solve_homogenization(q)
    record = json.loads(timing.read_text())["cufhom_core_application_timing"]
    infos = record["primaldual"]["primal"]["load_solver_infos"]
    assert all(
        i["retained_solution_host"] == (not return_fields)
        and i["true_residual_converged"]
        and i["true_norm_res_rel"] <= 1e-5
        for i in infos
    )
    if not return_fields:
        assert record["solution_assembly"]["strategy"] == "host_fields_chunked_energy"
    else:
        assert len(q["_solution_fields_result"]) == 6
    np.testing.assert_allclose(result, resident, atol=2e-13, rtol=2e-13)


@pytest.mark.parametrize(
    "blocked",
    [
        "store_solution_fields",
        "partial_load_output",
        "stress_field_consumer",
        "solution_field_consumer",
        "affine_sensitivity_cfields",
    ],
)
def test_direct_host_policy_preserves_consumer_contracts(blocked):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom.fft.linear.coordinates import _retain_host_solution

    conf = dict(store_solution_fields=False, postprocess_assembly="fused")
    assert _retain_host_solution(SimpleNamespace(solve=conf), (512,) * 3)
    conf[blocked] = (lambda *a: None) if blocked.endswith("consumer") else True
    assert not _retain_host_solution(SimpleNamespace(solve=conf), (512,) * 3)


def test_double_buffered_assembly_honors_nondefault_source_stream():
    if cp is None:
        pytest.skip("CuPy unavailable")
    A, fields, arrays = sample()
    reference = assembly_matrix_fused(A, fields)
    stream = cp.cuda.Stream(non_blocking=True)
    ready = cp.cuda.Event()
    ready.record()
    stream.wait_event(ready)
    with stream:
        for i in range(3):
            fields[i].val = arrays[i].copy()
        for i in range(3, 6):
            fields[i].val = fields[i].val.copy()
        result = indexed_assembly_streamed(A, fields, chunk_voxels=19)
        actual = cp.asnumpy(result) / np.prod(A.N)
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("chunk", [1, 37, 400])
def test_deferred_macroscopic_strain_is_added_in_fp32_on_staging(chunk):
    if cp is None:
        pytest.skip("CuPy unavailable")
    A, fields, raw = sample()
    means = np.random.default_rng(175).normal(size=(6, 6)).astype(np.float32)
    total = raw + means[:, :, None, None, None]
    for field, value in zip(fields, total):
        field.val = cp.asarray(value)
    reference = assembly_matrix_fused(A, fields)
    for i, field in enumerate(fields):
        field.val = raw[i].copy()
        field.cufhom_deferred_mean = means[i]
    actual = cp.asnumpy(indexed_assembly_streamed(A, fields, chunk)) / np.prod(A.N)
    np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13)
    for field, original in zip(fields, raw):
        np.testing.assert_array_equal(field.val, original)
