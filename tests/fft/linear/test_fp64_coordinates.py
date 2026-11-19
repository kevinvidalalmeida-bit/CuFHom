"""FP64 coordinates versus independent dense material/CPU GaNi references."""

import json

import numpy as np
import pytest
from cufhom.fft.linear import coordinates, fused
from cufhom.fft.linear.energy import energy_action, energy_update_device
from cufhom.fft.linear.krylov import SymmetricCoordinates
from cufhom.fft.operators import MixedProjector
from cufhom_core.tensors import Tensor, fft

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    old = fft.get_fft_backend_status()
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    yield
    fft.set_fft_backend(old["requested"])
    fft.set_cupy_plan_mode(old["cupy_plan_mode"])


@pytest.mark.parametrize("shape", [(12, 10, 14), (13, 11, 26)])
@pytest.mark.parametrize("lookup", [False, True])
def test_fp64_operator_and_updates_against_dense_voxel_law(shape, lookup, monkeypatch):
    monkeypatch.setattr(fused, "_PIPELINE_PARTS", {})
    monkeypatch.setattr(fused, "_GEOMETRY_CACHE", fused.OrderedDict())
    if not lookup:
        monkeypatch.setattr(fused, "_FREQUENCY_NORM_MAX_BYTES", 0)
    rng = np.random.default_rng(81)
    lengths = (1.0, 1.3, 0.9)
    Q = SymmetricCoordinates(
        MixedProjector(shape, lengths, np.ones(6, bool), cp), 3.0, 2.0
    )
    ids = rng.integers(0, 3, shape, dtype=np.int32)
    m = rng.normal(size=(3, 6, 6))
    m = m @ m.transpose(0, 2, 1) + 5 * np.eye(6)
    A = Tensor(
        name="indexed",
        val=cp.asarray(ids[None]),
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    iu = np.triu_indices(6)
    A.material_table = cp.ascontiguousarray(cp.asarray(m[:, iu[0], iu[1]]))
    field = rng.normal(size=(6,) + shape)
    cpu = SymmetricCoordinates(
        MixedProjector(shape, lengths, np.ones(6, bool), np), 3.0, 2.0
    )
    fft.set_fft_backend("scipy")
    p64 = cpu.adjoint(field)
    strain = cpu.strain(p64)
    stress = np.einsum("...ij,j...->i...", m[ids], strain)
    ap64 = cpu.adjoint(stress)
    expected_energy = np.vdot(strain, stress).real
    fft.set_fft_backend("cupy")
    p = cp.asarray(p64)
    np.testing.assert_allclose(cp.asnumpy(Q.strain(p)), strain, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(
        cp.asnumpy(
            A(
                Tensor(
                    name="strain",
                    val=cp.asarray(strain),
                    N=shape,
                    order=1,
                    Fourier=False,
                    fft_form="r",
                )
            ).val
        ),
        stress,
        atol=2e-13,
        rtol=2e-13,
    )
    for is_fused in [False, True]:
        pipeline = fused.FusedPipeline(Q, A) if is_fused else None
        if is_fused:
            assert (pipeline._parts.get("frequency_norm") is not None) == lookup
        v = pipeline.to_fused(p) if is_fused else p.copy()
        step, energy = (
            (fused.FusedStep(pipeline), pipeline.apply(v))
            if is_fused
            else energy_action(Q, A, v)
        )
        assert energy.dtype == cp.float64
        np.testing.assert_allclose(float(energy), expected_energy, rtol=3e-14)
        x = cp.zeros_like(v)
        r = v.copy()
        y = cp.ones_like(v)
        energy_update_device(v, step, 0.07 * float(energy), energy, x, r, (y, 0.3))
        if is_fused:
            pipeline.settle()
            x, r = pipeline.to_production(x), pipeline.to_production(r)
            assert (
                pipeline.S.dtype == cp.complex128 and pipeline.table.dtype == cp.float64
            )
        assert x.dtype == cp.complex128
        np.testing.assert_allclose(cp.asnumpy(x), 0.07 * p64, atol=3e-14, rtol=3e-13)
        np.testing.assert_allclose(
            cp.asnumpy(r), p64 - 0.07 * ap64, atol=5e-13, rtol=5e-13
        )


@pytest.mark.parametrize(
    "backend", ["unfused", "fused_host", "fused_graph", "odd_auto"]
)
def test_fp64_six_load_solve_and_physical_acceptance(backend, tmp_path, monkeypatch):
    from cufhom.fft.linear.solver import solve_homogenization

    shape = (9, 7, 5) if backend == "odd_auto" else (12, 10, 14)
    phase = np.zeros(shape, np.uint8)
    phase[2:8, 3:7, 1:10] = 1
    ori = np.zeros(shape + (3,))
    ori[:] = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    p = dict(
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
        solver_real_dtype="float64",
        solver_rtol=1e-10,
        solver_verbose=False,
    )
    reference = solve_homogenization(dict(p, fft_backend="scipy"))
    if backend == "unfused":
        monkeypatch.setattr(coordinates, "fused_supported", lambda *args: False)
    monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", "0" if backend == "fused_host" else "1")
    path = tmp_path / "timing.json"
    result = solve_homogenization(
        dict(
            p,
            fft_backend="cupy",
            solver_fp64_coordinates=True,
            solver_timing_path=str(path),
        )
    )
    report = json.loads(path.read_text())
    infos = report["cufhom_core_application_timing"]["primaldual"]["primal"][
        "load_solver_infos"
    ]
    assert report["linear_formulation"] == "displacement"
    assert len(infos) == 6
    # Odd grids take the fused pipeline too (line-pair KC).
    assert all(i["linear_fused_fft"] == (backend != "unfused") for i in infos)
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-10 for i in infos
    )
    np.testing.assert_allclose(result, reference, atol=1e-11, rtol=2e-11)


def test_geometry_cache_evicts_without_invalidating_live_pipelines(monkeypatch):
    monkeypatch.setattr(fused, "_GEOMETRY_CACHE", fused.OrderedDict())
    monkeypatch.setattr(fused, "_FREQUENCY_NORM_MAX_BYTES", 8 * 8 * 4 * 8)
    first, second = {}, {}
    live = fused._geometry_for_pipeline("first", first, (8, 8, 8), (1.0, 1.0, 1.0))
    saved = live.copy()
    other = fused._geometry_for_pipeline("second", second, (8, 8, 8), (1.0, 1.3, 0.9))
    assert list(fused._GEOMETRY_CACHE) == ["second"]
    assert "frequency_norm" not in first
    cp.testing.assert_array_equal(live, saved)
    assert other.nbytes <= fused._FREQUENCY_NORM_MAX_BYTES
    rebuilt = fused._geometry_for_pipeline("first", first, (8, 8, 8), (1.0, 1.0, 1.0))
    cp.testing.assert_array_equal(rebuilt, saved)
    assert list(fused._GEOMETRY_CACHE) == ["first"]
