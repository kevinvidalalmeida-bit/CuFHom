"""Independent material/coordinate references, including inadmissible slots."""

from types import SimpleNamespace

import numpy as np
import pytest
from cufhom.fft.linear.energy import (
    energy_action,
    energy_update_device,
    material_energy,
)
from cufhom.fft.linear.krylov import SymmetricCoordinates
from cufhom.fft.nonlinear.solver import MixedProjector
from cufhom_core.tensors import Tensor, fft

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_gpu():
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")


def test_default_pipeline_reports_energy_and_material_profile(tmp_path, monkeypatch):
    import json
    import os
    from contextlib import redirect_stdout

    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    phase = np.zeros((8, 8, 8), np.uint8)
    phase[2:5, 2:5, 2:5] = 1
    ori = np.zeros(phase.shape + (3,), np.float32)
    ori[..., 2] = 1
    p = dict(
        input_dir=".",
        phase_array=phase,
        ori_array=ori,
        fft_backend="cupy",
        Em=3.0,
        nu_m=0.3,
        Ef_L=90.0,
        Ef_T=15.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
        solver_rtol=1e-5,
        solver_maxiter=1000,
        require_convergence=True,
        solver_verbose=False,
        profile_timing=True,
        solver_timing_path=str(tmp_path / "profile.json"),
    )
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        C = solve_homogenization(p)
    detail = json.loads((tmp_path / "profile.json").read_text())
    primal = detail["cufhom_core_application_timing"]["primaldual"]["primal"]
    infos = primal["load_solver_infos"]
    assert len(infos) == 6 and detail["linear_formulation"] == "displacement"
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5 for i in infos
    )
    assert np.linalg.eigvalsh((C + C.T) / 2)[0] > 0
    # The material law is fused into the CG kernels; profiling times the
    # right-hand sides and the physical audits, without changing the path.
    loads = detail["cufhom_core_solver_timing"]["primal"]["loads"]
    assert sum(load["components"]["A"]["calls"] for load in loads) >= 6


def test_energy_rejects_custom_operator_and_coordinate_contracts():
    from cufhom.fft.linear.energy import supports_energy

    shape = (3, 3, 3)
    Q = SymmetricCoordinates(
        MixedProjector(shape, (1.0,) * 3, np.ones(6, bool), cp), 3.0, 2.0
    )
    A = Tensor(
        name="indexed",
        val=cp.zeros((1,) + shape, cp.int32),
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    A.material_table = cp.ones((1, 21), cp.float32)
    assert supports_energy(Q, A)

    class Custom(Tensor):
        def __call__(self, field):
            return 2 * field

    other = Custom(
        name="custom",
        val=A.val,
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    other.material_table = A.material_table
    assert not supports_energy(Q, other)
    A.multype = "scalar"
    assert not supports_energy(Q, A)
    A.multype = "sym21_indexed"
    A.material_table = cp.empty((0, 21), cp.float32)
    assert not supports_energy(Q, A)

    class CustomCoordinates(SymmetricCoordinates):
        pass

    A.material_table = cp.ones((1, 21), cp.float32)
    assert supports_energy(Q, A)
    assert not supports_energy(CustomCoordinates(Q.P, 3.0, 2.0), A)


@pytest.mark.parametrize("bad_id", [-1, 1])
def test_invalid_material_id_poisoned_without_out_of_bounds(bad_id):
    shape = (3, 3, 3)
    A = Tensor(
        name="indexed",
        val=cp.zeros((1,) + shape, cp.int32),
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    A.material_table = cp.ones((1, 21), cp.float32)
    A.val.ravel()[7] = bad_id
    strain = cp.ones((6,) + shape, cp.float32)
    stress, pap = material_energy(A, strain)
    assert np.isnan(float(pap)) and bool(cp.all(cp.isnan(stress.reshape(6, -1)[:, 7])))
    field = Tensor(
        name="field", val=strain, N=shape, order=1, Fourier=False, fft_form="r"
    )
    assert bool(cp.all(cp.isnan(A(field).val.reshape(6, -1)[:, 7])))


@pytest.mark.parametrize("shape", [(8, 8, 8), (9, 7, 5), (10, 8, 12)])
@pytest.mark.parametrize("smooth", [False, True])
def test_energy_and_fused_update(shape, smooth):
    rng = np.random.default_rng(758)
    old = fft.get_fft_backend_status()
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual")
    try:
        matrices = rng.normal(size=(7, 6, 6))
        matrices = np.einsum("mki,mkj->mij", matrices, matrices) + np.eye(6)
        ids = rng.integers(0, 7, shape, dtype=np.int32)
        table = matrices[:, np.triu_indices(6)[0], np.triu_indices(6)[1]]
        descriptor = SimpleNamespace(
            val=cp.asarray(ids),
            material_table=cp.ascontiguousarray(cp.asarray(table, cp.float32)),
        )
        Q = SymmetricCoordinates(
            MixedProjector(shape, (2.0, 3.0, 4.0), np.ones(6, bool), cp), 3.0, 2.0
        )
        cpu = SymmetricCoordinates(
            MixedProjector(shape, (2.0, 3.0, 4.0), np.ones(6, bool), np), 3.0, 2.0
        )
        source = rng.normal(size=(6,) + shape)
        fft.set_fft_backend("scipy")
        p64 = cpu.adjoint(source)
        p = cp.asarray(p64, cp.complex64)
        eps64 = cpu.strain(p64)
        sig64 = np.einsum("...ij,j...->i...", matrices[ids], eps64)
        expected_pap = np.vdot(eps64, sig64).real
        ap = cpu.adjoint(sig64)
        np.testing.assert_allclose(np.vdot(p64, ap).real, expected_pap, rtol=2e-14)
        fft.set_fft_backend("cupy")
        eps = Q.strain(p)
        stress, pap = material_energy(descriptor, eps)
        np.testing.assert_allclose(cp.asnumpy(stress), sig64, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(float(pap), expected_pap, rtol=8e-7)
        deferred, pap2 = energy_action(Q, descriptor, p)
        assert (
            isinstance(pap2, cp.ndarray)
            and pap2.dtype == cp.float64
            and pap2.shape == ()
        )
        np.testing.assert_allclose(float(pap2), float(pap), rtol=1e-14)
        # All coordinate slots, including mean/Nyquist, exercise the update.
        x = cp.asarray(
            rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size), cp.complex64
        )
        r = x.copy()
        y = 2 * x
        old_y = y.copy()
        old_x = x.copy()
        old_r = r.copy()
        alpha, blend = 0.07, 0.35
        new_rr = energy_update_device(
            p, deferred, alpha * float(pap2), pap2, x, r, (y, blend) if smooth else None
        )
        assert deferred.spectrum is None
        np.testing.assert_allclose(
            cp.asnumpy(x),
            cp.asnumpy(old_x) + np.float32(alpha) * p64,
            atol=1e-6,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            cp.asnumpy(r),
            cp.asnumpy(old_r) - np.float32(alpha) * ap,
            atol=2e-5,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            new_rr, np.vdot(cp.asnumpy(r), cp.asnumpy(r)).real, rtol=2e-7
        )
        if smooth:
            np.testing.assert_allclose(
                cp.asnumpy(y),
                cp.asnumpy(old_y) + np.float32(blend) * cp.asnumpy(old_x - old_y),
                atol=1e-6,
            )
    finally:
        fft.set_fft_backend(old["requested"])
        fft.set_cupy_plan_mode(old["cupy_plan_mode"])


@pytest.mark.parametrize(
    "denominator,numerator",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (np.nan, 1.0),
        (np.inf, 1.0),
        (1.0, np.nan),
        (1.0, np.inf),
        (1e-300, 1e300),
    ],
)
def test_device_energy_invalid_step_cannot_report_convergence(denominator, numerator):
    from cufhom.fft.linear.energy import DeferredHP

    # All slots are masked on this grid. An invalid scalar must still poison
    # the reduction instead of becoming a zero residual after projection.
    Q = SymmetricCoordinates(
        MixedProjector((1, 1, 1), (1.0, 1.0, 1.0), np.ones(6, bool), cp), 3.0, 2.0
    )
    step = DeferredHP(Q, cp.zeros((6,) + Q.hshape, cp.complex64))
    v = cp.zeros(Q.size, cp.complex64)
    norm = energy_update_device(
        v, step, numerator, cp.asarray(denominator, cp.float64), v.copy(), v.copy()
    )
    assert not np.isfinite(norm)


def test_device_energy_false_recursive_exit_restarts(monkeypatch, tmp_path):
    """Exercise the device callback through CG and the physical acceptance."""
    import cufhom.fft.linear.coordinates as module
    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    # The library route's host CG drives this callback; odd grids are fused now.
    monkeypatch.setattr(module, "fused_supported", lambda *args: False)
    shape = (5, 7, 5)
    phase = np.zeros(shape, np.uint8)
    phase[1:3, 1:3, :] = 1
    ori = np.zeros(shape + (3,), np.float32)
    ori[..., 0] = 1
    base = dict(
        input_dir=".",
        phase_array=phase,
        ori_array=ori,
        Em=3.0,
        nu_m=0.3,
        Ef_L=70.0,
        Ef_T=8.0,
        nu_LT=0.25,
        nu_TT=0.3,
        G_LT=5.0,
        solver_rtol=1e-5,
        require_convergence=True,
        solver_verbose=False,
    )
    reference = solve_homogenization(
        dict(base, fft_backend="scipy", solver_real_dtype="float64", solver_rtol=1e-10)
    )
    original, calls = module.energy_update_device, []

    def false_once(*args):
        norm = original(*args)
        calls.append(1)
        return 0.0 if len(calls) == 1 else norm

    monkeypatch.setattr(module, "energy_update_device", false_once)
    actual = solve_homogenization(dict(base, fft_backend="cupy"))
    assert len(calls) > 6
    np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=3e-6)
