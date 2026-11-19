"""The large strain route keeps x/r/p resident without changing its equations."""

import json

import numpy as np
import pytest
from cufhom_core.general import solver
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp

pytestmark = pytest.mark.gpu


def operator(shape):
    from cufhom_core.projections import elasticity_direct
    from cufhom_core.tensors.objects import SYM21_PAIRS
    from cufhom_core.tensors.operators import DFT, Operator

    rng = np.random.default_rng(413)
    raw = cp.asarray(rng.normal(size=(6,) + shape).astype(np.float32))
    B = Tensor(val=raw, order=1, N=shape, Fourier=False, fft_form="r")
    ids = cp.asarray(rng.integers(0, 3, shape, dtype=np.int32))
    M = rng.normal(size=(3, 6, 6))
    C = np.einsum("pji,pjk->pik", M, M) + np.eye(6)[None, :, :]
    A = Tensor(
        val=ids, order=0, N=shape, Fourier=False, fft_form="r", multype="sym21_indexed"
    )
    A.material_table = cp.asarray(
        [[c[i, j] for i, j in SYM21_PAIRS] for c in C], dtype=cp.float32
    )
    G = elasticity_direct(shape, Y=np.array([1.0, 1.3, 0.9]), fft_form="r")
    F = DFT(N=shape, fft_form="r")
    MANDEL_IDENTITY = DFT(N=shape, inverse=True, fft_form="r")
    return Operator(mat_rev=[[A, Operator(mat_rev=[[F, G, MANDEL_IDENTITY]])]]), B


@pytest.mark.parametrize("shape", [(12, 10, 8), (9, 11, 13), (32, 32, 32)])
@pytest.mark.parametrize("side_stream", [False, True])
def test_inplace_library_operator_matches_existing_and_preserves_inputs(
    shape, side_stream
):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core.general.cupy_strain_capacity import supported
    from cufhom_core.tensors import fft

    previous = fft.get_fft_backend()
    fft.set_fft_backend("cupy")
    stream = (
        cp.cuda.Stream(non_blocking=True)
        if side_stream
        else cp.cuda.get_current_stream()
    )
    try:
        with stream:
            A, B = operator(shape)
            saved = B.val.copy()
            assert supported(A, B)
            reference = A(B).val
            A._capacity_inplace_fft = True
            for _ in range(2):
                result = A(B).val
                assert result.flags.c_contiguous
                assert result.data.ptr != B.val.data.ptr
                cp.testing.assert_allclose(result, reference, rtol=3e-6, atol=3e-4)
                cp.testing.assert_array_equal(B.val, saved)
        stream.synchronize()
    finally:
        fft.set_fft_backend(previous)


@pytest.mark.parametrize("initial", [False, True])
def test_resident_rhs_reuse_reproduces_cg_and_restores_rhs(initial):
    if cp is None:
        pytest.skip("CuPy unavailable")
    rng = np.random.default_rng(741)
    shape = (5, 4, 6)
    raw = cp.asarray(rng.normal(size=(6,) + shape).astype(np.float32))
    diagonal = cp.asarray(rng.uniform(1, 9, size=raw.shape).astype(np.float32))
    B = Tensor(val=raw.copy(), order=1, N=shape, Fourier=False, fft_form="r")

    def action(x):
        return x.copy(val=diagonal * x.val)

    start = raw * 0.15 if initial else None
    p = dict(rtol=1e-5, maxiter=200, check_true_residual=True)
    reference, old = solver.CG(
        action, B, x0=B.copy(val=start.copy()) if initial else None, par=p
    )
    B.cufhom_resident_cg = True
    result, new = solver.CG(
        action, B, x0=B.copy(val=start.copy()) if initial else None, par=p
    )
    assert (
        new["resident_cg_iterate"]
        and not new["host_cg_iterate"]
        and new["true_residual_converged"]
    )
    assert not hasattr(action, "_capacity_inplace_fft")
    assert new["kit"] == old["kit"]
    np.testing.assert_array_equal(new["norm_res_log"], old["norm_res_log"])
    cp.testing.assert_array_equal(result.val, reference.val)
    cp.testing.assert_array_equal(B.val, raw)


def test_resident_cg_restores_rhs_and_operator_after_exception():
    if cp is None:
        pytest.skip("CuPy unavailable")
    shape = (4, 5, 6)
    raw = cp.arange(720, dtype=cp.float32).reshape((6,) + shape) + 1
    B = Tensor(val=raw.copy(), order=1, N=shape, Fourier=False, fft_form="r")
    B.cufhom_resident_cg = True
    diagonal = cp.linspace(1, 9, 720, dtype=cp.float32).reshape(raw.shape)
    calls = []

    def action(x):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("resident operator failure")
        return x.copy(val=diagonal * x.val)

    action._capacity_inplace_fft = False
    with pytest.raises(RuntimeError, match="resident operator failure"):
        solver.CG(action, B, par=dict(rtol=1e-8, maxiter=100))
    cp.testing.assert_array_equal(B.val, raw)
    assert action._capacity_inplace_fft is False


@pytest.mark.parametrize("fields", [False, True])
def test_resident_capacity_six_loads_against_independent_cpu_fp64(
    monkeypatch, tmp_path, fields
):
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom.fft.linear import coordinates
    from cufhom_core import applications

    from tests.fft.linear.test_linear_coordinates import case, solve_homogenization

    p = case((12, 10, 8))
    p.update(require_convergence=True, solver_maxiter=1000)
    reference = solve_homogenization(
        dict(p, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    monkeypatch.setattr(coordinates, "make_linear_solver", lambda **kw: kw["fallback"])
    monkeypatch.setattr(
        applications, "host_iterate_required", lambda shape, dtype=np.float32: True
    )
    path = tmp_path / "resident.json"
    q = dict(p, solver_timing_path=str(path))
    if fields:
        q["solution_field_return_in_memory"] = True
    result = solve_homogenization(q)
    np.testing.assert_allclose(result, reference, rtol=3e-6, atol=3e-6)
    info = json.loads(path.read_text())["cufhom_core_application_timing"]["primaldual"][
        "primal"
    ]["load_solver_infos"]
    assert all(i["resident_cg_iterate"] and not i["host_cg_iterate"] for i in info)
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in info
    )
    if fields:
        assert len(q["_solution_fields_result"]) == 6


def test_resident_zero_rhs_is_finite_and_restored():
    if cp is None:
        pytest.skip("CuPy unavailable")
    B = Tensor(
        val=cp.zeros((6, 4, 4, 4), cp.float32),
        order=1,
        N=(4, 4, 4),
        Fourier=False,
        fft_form="r",
    )
    B.cufhom_resident_cg = True
    result, info = solver.CG(
        lambda x: x.copy(),
        B,
        par=dict(rtol=1e-5, maxiter=100, check_true_residual=True),
    )
    assert info["kit"] == 0 and info["true_residual_converged"]
    cp.testing.assert_array_equal(result.val, cp.zeros_like(B.val))
    cp.testing.assert_array_equal(B.val, cp.zeros_like(B.val))


def test_resident_selector_rejects_other_layouts():
    if cp is None:
        pytest.skip("CuPy unavailable")
    from cufhom_core.general.cupy_strain_capacity import supported
    from cufhom_core.tensors import fft

    previous = fft.get_fft_backend()
    fft.set_fft_backend("cupy")
    try:
        A, B = operator((12, 10, 8))
        assert supported(A, B)
        assert not supported(lambda x: x, B)
        assert not supported(A, B.copy(val=B.val.astype(cp.float64)))
        assert not supported(A, B.copy(val=B.val[..., ::-1]))
        A.mat_rev[0][0].material_table = A.mat_rev[0][0].material_table[:, :20].copy()
        assert not supported(A, B)
    finally:
        fft.set_fft_backend(previous)
