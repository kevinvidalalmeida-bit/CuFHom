"""Fused axis-split FFT pipeline: line FFTs, operator, layout and full solves."""

import contextlib
import io

import numpy as np
import pytest
from cufhom.fft.linear import fused
from cufhom.fft.linear.energy import energy_action, energy_update_device
from cufhom.fft.linear.krylov import SymmetricCoordinates, fused_cg
from cufhom.fft.operators import MixedProjector
from cufhom_core.tensors import Tensor, fft

from tests.fft.linear.test_linear_fusions import matrix_protocol, real_dot

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_gpu():
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual", preserve_cache=True)


@pytest.fixture(autouse=True)
def compile_before_solving(monkeypatch):
    # Deterministic path choice; the deferred background compile has its own test.
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")


@pytest.mark.parametrize(
    "L", [4, 6, 12, 26, 44, 60, 64, 90, 98, 120, 128, 160, 216, 256]
)
@pytest.mark.parametrize("sign", [-1, 1])
def test_line_fft_matches_numpy(L, sign):
    W, NT = 6, 128
    source = f"""
extern "C" __global__ void k(float2* x,const float2* tw_g) {{
  __shared__ float2 buf[{L * W}]; __shared__ float2 tw[{L}];
  const int tid=threadIdx.x; float2* base=x+(long long)blockIdx.x*{L * W};
  for(int i=tid;i<{L};i+={NT})tw[i]=tw_g[i];
  for(int i=tid;i<{L * W};i+={NT})buf[i]=base[i];
  __syncthreads();
  {fused._fft(L, W, NT, sign)}
  for(int i=tid;i<{L * W};i+={NT})base[i]=buf[i];
}}"""
    kernel = cp.RawKernel(source, "k")
    rng = np.random.default_rng(L)
    a = (rng.standard_normal((3, L, W)) + 1j * rng.standard_normal((3, L, W))).astype(
        np.complex64
    )
    out = cp.asarray(a)
    kernel(
        (3,),
        (NT,),
        (out, cp.asarray(np.exp(-2j * np.pi * np.arange(L) / L).astype(np.complex64))),
    )
    reference = (
        np.fft.fft(a.astype(np.complex128), axis=1)
        if sign < 0
        else np.fft.ifft(a.astype(np.complex128), axis=1) * L
    )
    assert np.linalg.norm(cp.asnumpy(out) - reference) <= 3e-7 * np.linalg.norm(
        reference
    )


def _problem(shape, seed=0):
    rng = np.random.default_rng(seed)
    P = MixedProjector(shape, (1.0, 1.3, 0.9), np.ones(6, bool), cp)
    Q = SymmetricCoordinates(P, 2.5, 1.2)
    ids = cp.asarray(rng.integers(0, 4, size=(1,) + shape).astype(np.int32))
    A = Tensor(
        name="indexed",
        val=ids,
        N=shape,
        order=1,
        Fourier=False,
        fft_form="r",
        multype="sym21_indexed",
    )
    table = np.zeros((4, 21), np.float32)
    iu = np.triu_indices(6)
    for row in range(4):
        G = rng.standard_normal((6, 6))
        table[row] = (G @ G.T + 6 * (row + 1) * np.eye(6))[iu]
    A.material_table = cp.asarray(table)
    return Q, A, rng


@pytest.mark.parametrize(
    "shape", [(8, 8, 8), (12, 10, 14), (15, 9, 10), (13, 11, 26), (14, 21, 18)]
)
@pytest.mark.parametrize("direction", [False, True])
def test_fused_operator_matches_production_energy_pipeline(shape, direction):
    Q, A, rng = _problem(shape)
    assert fused.fused_supported(Q, A)

    def field():
        return cp.asarray(rng.standard_normal((6,) + shape).astype(np.float32))

    p, r = Q.adjoint(field()), Q.adjoint(field())
    beta = 0.43 if direction else 0.0
    p0, r0 = p.copy(), r.copy()
    step, pap = energy_action(Q, A, p0, (r0, beta) if direction else None)
    x0, rr0 = cp.zeros_like(p0), r0.copy()
    norm0 = energy_update_device(p0, step, float(pap), pap, x0, rr0)

    pipeline = fused.FusedPipeline(Q, A)
    pf, rf = pipeline.to_fused(p), pipeline.to_fused(r)

    def rel(a, b):
        return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))

    # Even grids pack the k2 Nyquist plane into k2 = 0: the round trip keeps
    # the Hermitian parts of both planes (exact for real fields up to rounding).
    assert rel(pipeline.to_production(pf), p) <= 1e-6
    papf = pipeline.apply(pf, (rf, beta) if direction else None)
    xf, rrf = cp.zeros_like(pf), rf.copy()
    norm = fused.FusedStep(pipeline).update(
        pf, float(papf), xf, rrf, None, denominator=papf
    )
    pipeline.settle()

    assert rel(pipeline.to_production(pf), p0) <= 1e-6
    assert abs(float(papf) - float(pap)) <= 1e-6 * abs(float(pap))
    assert rel(pipeline.to_production(rf - rrf), r0 - rr0) <= 2e-6
    assert rel(pipeline.to_production(xf), x0) <= 1e-6
    assert abs(norm - norm0) <= 1e-5 * norm0


def test_fused_smoothing_and_invalid_energy():
    Q, A, rng = _problem((8, 10, 12))
    pipeline = fused.FusedPipeline(Q, A)
    p = pipeline.to_fused(
        Q.adjoint(cp.asarray(rng.standard_normal((6, 8, 10, 12)).astype(np.float32)))
    )
    pap = pipeline.apply(p)
    x, r = cp.ones_like(p), cp.zeros_like(p)
    y = cp.zeros_like(p)
    step = fused.FusedStep(pipeline)
    step.update(p, float(pap), x, r, (y, 0.25), denominator=pap)
    pipeline.settle()
    # y moves a quarter of the way to the previous x before x is updated.
    assert cp.allclose(y, 0.25 * cp.ones_like(p))
    bad = cp.asarray(-1.0, cp.float64)
    pipeline.apply(p)
    assert not np.isfinite(step.update(p, 1.0, x, r, None, denominator=bad))


def test_fused_contract_rejections():
    Q, A, _ = _problem((8, 8, 9))
    assert fused.fused_supported(Q, A)  # odd real-transform axis: line pairs
    Q, A, _ = _problem((8, 8, 17))
    assert not fused.fused_supported(Q, A)  # prime contiguous length above 13
    Q, A, _ = _problem((17, 8, 8))
    assert not fused.fused_supported(Q, A)  # prime factor above 13
    Q, A, _ = _problem((8, 8, 8))
    A.material_table = A.material_table.astype(cp.float64)
    assert fused.fused_supported(Q, A)  # native FP64 kernels now exist


def test_fused_cg_accept_tightens_and_continues():
    rng = np.random.default_rng(3)
    n = 4096
    d = cp.asarray(np.linspace(1, 50, n).astype(np.float32))
    rhs = cp.asarray(
        (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    )

    calls = []

    def accept(x):
        calls.append(np.sqrt(real_dot(rhs - d * x, rhs - d * x)))
        return 1e-4 if len(calls) == 1 else None

    x, iterations = fused_cg(
        lambda z: d * z,
        rhs,
        real_dot,
        1e-3,
        500,
        accept=accept,
        **matrix_protocol(lambda z: d * z),
    )
    assert len(calls) == 2
    assert calls[1] <= 1e-3 * 1e-2 * np.sqrt(real_dot(rhs, rhs)) * 1.01
    assert iterations > 0


def test_full_solve_fused_matches_production(tmp_path, monkeypatch):
    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    rng = np.random.default_rng(7)
    phase = (rng.random((16, 18, 20)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
    )

    def run(enabled):
        import json

        from cufhom.fft.linear import coordinates

        if not enabled:
            monkeypatch.setattr(coordinates, "fused_supported", lambda Q, A: False)
        p = dict(base, solver_timing_path=str(tmp_path / f"timing_{enabled}.json"))
        with contextlib.redirect_stdout(io.StringIO()):
            C = solve_homogenization(p)
        infos = json.loads((tmp_path / f"timing_{enabled}.json").read_text())[
            "cufhom_core_application_timing"
        ]["primaldual"]["primal"]["load_solver_infos"]
        monkeypatch.undo()
        monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
        return C, infos

    C_fused, infos_fused = run(True)
    C_prod, infos_prod = run(False)
    assert all(i["linear_fused_fft"] for i in infos_fused)
    assert not any(i["linear_fused_fft"] for i in infos_prod)
    assert all(
        i["true_residual_converged"] and i["true_norm_res_rel"] <= 1e-5
        for i in infos_fused
    )
    assert np.linalg.norm(C_fused - C_prod) <= 1e-6 * np.linalg.norm(C_prod)
    assert (
        abs(
            sum(i["cg_iterations"] for i in infos_fused)
            - sum(i["cg_iterations"] for i in infos_prod)
        )
        <= 6
    )


def test_forced_solution_offload_is_exact(tmp_path, monkeypatch):
    import json

    import cufhom_core.applications as applications
    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    rng = np.random.default_rng(11)
    phase = (rng.random((12, 14, 16)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
    )
    with contextlib.redirect_stdout(io.StringIO()):
        C_device = solve_homogenization(dict(base))
        monkeypatch.setattr(applications, "_resident_limit", lambda *a: 0)
        monkeypatch.setattr(applications, "_COMPRESS_SOLUTIONS", False)
        monkeypatch.setattr(applications, "_RESIDENCY", {})
        C_host = solve_homogenization(
            dict(base, solver_timing_path=str(tmp_path / "t.json"))
        )
    timing = json.loads((tmp_path / "t.json").read_text())[
        "cufhom_core_application_timing"
    ]
    assert timing["offloaded_solution_loads"] == [0, 1, 2, 3, 4]
    with contextlib.redirect_stdout(io.StringIO()):
        monkeypatch.setattr(applications, "_resident_limit", lambda *a: 2)
        monkeypatch.setattr(applications, "_RESIDENCY", {})
        C_partial = solve_homogenization(
            dict(base, solver_timing_path=str(tmp_path / "t2.json"))
        )
    timing = json.loads((tmp_path / "t2.json").read_text())[
        "cufhom_core_application_timing"
    ]
    assert timing["offloaded_solution_loads"] == [0, 1, 2]
    assert np.array_equal(C_device, C_partial)
    assert np.array_equal(C_device, C_host)


@pytest.mark.parametrize("slabs,streams", [(3, 1), (2, 2), (5, 3)])
def test_l2_slab_chunks_are_bitwise_identical(slabs, streams):
    shape = (16, 12, 20)
    Q, A, rng = _problem(shape)
    p = Q.adjoint(cp.asarray(rng.standard_normal((6,) + shape).astype(np.float32)))
    out = []
    for config in (
        dict(slabs=16, slab_streams=1),
        dict(slabs=slabs, slab_streams=streams),
    ):
        pipeline = fused.FusedPipeline(Q, A, config)
        z = pipeline.to_fused(p)
        pap = pipeline.apply(z)
        x, r = cp.zeros_like(z), cp.zeros_like(z)
        norm = pipeline.update(z, float(pap), pap, x, r)
        pipeline.settle()
        out.append((float(pap), norm, pipeline.to_production(r)))
    assert out[0][0] == out[1][0] and out[0][1] == out[1][1]
    assert cp.array_equal(out[0][2], out[1][2])


def test_gpu_orientation_groups_match_cpu_layout():
    from cufhom.fft.linear import solver

    rng = np.random.default_rng(5)
    phase = (rng.random((14, 12, 10)) < 0.4).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    ori[phase == 1] = np.round(
        ori[phase == 1], 2
    )  # repeated orientations -> shared groups
    Cm = np.diag([5.0, 5, 5, 2, 2, 2]).astype(np.float32)
    Cf = np.diag([90.0, 9, 9, 4, 5, 5]).astype(np.float32)
    (ids, table), groups, _ = solver._build_cfield_gpu(
        phase, ori, Cm, Cf, storage="sym21", indexed=True
    )
    phase.flags.writeable = ori.flags.writeable = False
    layout = solver.compile_cfield_geometry(phase, ori)
    (ids_cpu, table_cpu), groups_cpu, _ = solver._build_cfield_gpu(
        phase, ori, Cm, Cf, storage="sym21", indexed=True, geometry_layout=layout
    )
    assert groups == groups_cpu and cp.array_equal(ids, ids_cpu)
    assert float(cp.max(cp.abs(table - table_cpu))) <= 1e-6 * float(
        cp.max(cp.abs(table_cpu))
    )


@pytest.mark.parametrize("iterations", [1, 3, 8])
def test_graph_cg_matches_host_loop_bitwise(iterations):
    from cufhom.fft.linear.energy import energy_update_device

    shape = (16, 12, 20)
    Q, A, rng = _problem(shape)
    pipeline = fused.FusedPipeline(Q, A, dict(graph_iterations=iterations))
    b = pipeline.to_fused(
        Q.adjoint(cp.asarray(rng.standard_normal((6,) + shape).astype(np.float32)))
    )

    def dot(u, v):
        return float(
            cp.sum(
                u.view(cp.float32).astype(cp.float64)
                * v.view(cp.float32).astype(cp.float64)
            )
        )

    def H(z):
        z = cp.ascontiguousarray(z)
        r, x = cp.zeros_like(z), cp.zeros_like(z)
        pap = pipeline.apply(z)
        pipeline.update(z, float(pap), pap, x, r)
        pipeline.pending = None
        return -r

    step = fused.FusedStep(pipeline)
    x_host, it_host = fused_cg(
        H,
        b,
        dot,
        1e-5,
        500,
        residual_smoothing=True,
        action_and_dot=lambda z, direction=None: (step, pipeline.apply(z, direction)),
        device_update_callback=energy_update_device,
        residual_action=H,
        settle=pipeline.settle,
    )
    x_host = x_host.copy()
    x_graph, it_graph = pipeline.graph_cg(b, dot, 1e-5, 500, H)
    assert it_host == it_graph and cp.array_equal(x_host, x_graph)


def test_graph_cg_stops_at_unattainable_tolerance():
    """A tolerance below FP32 accuracy must exhaust the budget, not restart forever."""
    shape = (16, 12, 20)
    Q, A, rng = _problem(shape)
    pipeline = fused.FusedPipeline(Q, A, dict(graph_iterations=8))
    b = pipeline.to_fused(
        Q.adjoint(cp.asarray(rng.standard_normal((6,) + shape).astype(np.float32)))
    )

    def dot(u, v):
        return float(
            cp.sum(
                u.view(cp.float32).astype(cp.float64)
                * v.view(cp.float32).astype(cp.float64)
            )
        )

    def H(z):
        z = cp.ascontiguousarray(z)
        r, x = cp.zeros_like(z), cp.zeros_like(z)
        pap = pipeline.apply(z)
        pipeline.update(z, float(pap), pap, x, r)
        pipeline.pending = None
        return -r

    with pytest.raises(fused.LinearSolveError):
        pipeline.graph_cg(b, dot, 1e-12, 200, H)


def test_compressed_solution_storage_is_exact(tmp_path, monkeypatch):
    import json

    import cufhom_core.applications as applications
    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    rng = np.random.default_rng(13)
    phase = (rng.random((12, 14, 16)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
        solution_field_return_in_memory=True,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = dict(base)
        C_device = solve_homogenization(p)
        fields_device = p["_solution_fields_result"]
        monkeypatch.setattr(applications, "_resident_limit", lambda *a: 0)
        for budget in (0, 2):
            monkeypatch.setattr(
                applications, "_compressed_budget", lambda solution, b=budget: b
            )
            monkeypatch.setattr(applications, "_RESIDENCY", {})
            q = dict(base, solver_timing_path=str(tmp_path / f"t{budget}.json"))
            C = solve_homogenization(q)
            timing = json.loads((tmp_path / f"t{budget}.json").read_text())[
                "cufhom_core_application_timing"
            ]
            assert timing["offloaded_solution_loads"] == (
                [0, 1, 2, 3, 4] if budget == 0 else [0, 1, 2]
            )
            assert np.array_equal(C, C_device)
            for a, b in zip(q["_solution_fields_result"], fields_device):
                assert np.array_equal(a, b)


@pytest.mark.parametrize("graphs", ["0", "1"])
@pytest.mark.parametrize("mode", ["resident", "offload", "compressed"])
def test_coordinate_release_preserves_fields_and_compressed_ownership(
    tmp_path, monkeypatch, graphs, mode
):
    import cufhom_core.applications as applications
    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", graphs)
    rng = np.random.default_rng(113)
    phase = (rng.random((12, 14, 16)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
        solution_field_return_in_memory=True,
    )
    # Compare against retaining coordinates with exactly the same solve/storage
    # route, including the old graph snapshot and compressed reconstruction.
    monkeypatch.setattr(
        applications, "_resident_limit", lambda *a: 6 if mode == "resident" else 0
    )
    monkeypatch.setattr(applications, "_COMPRESS_SOLUTIONS", mode == "compressed")
    monkeypatch.setattr(applications, "_compressed_budget", lambda solution: 2)
    with contextlib.redirect_stdout(io.StringIO()):
        with monkeypatch.context() as patch:
            patch.setattr(
                applications,
                "_release_full_solution_coordinates",
                lambda solution: None,
            )
            patch.setattr(applications, "_RESIDENCY", {})
            reference = dict(base)
            C_reference = solve_homogenization(reference)

        restore = applications._restore_offloaded_solutions
        observations = []

        def checked_restore(solutions, moved):
            if mode == "compressed":
                assert all(
                    sol.val is None and hasattr(sol, "cufhom_coordinates")
                    for sol in solutions[:5]
                )
                assert not hasattr(solutions[-1], "cufhom_coordinates")
            else:
                assert all(not hasattr(sol, "cufhom_coordinates") for sol in solutions)
            elapsed = restore(solutions, moved)
            assert all(
                sol.val is not None and not hasattr(sol, "cufhom_coordinates")
                for sol in solutions
            )
            observations.append(True)
            return elapsed

        monkeypatch.setattr(
            applications, "_restore_offloaded_solutions", checked_restore
        )
        monkeypatch.setattr(applications, "_RESIDENCY", {})
        candidate = dict(base)
        C = solve_homogenization(candidate)
        # Exercise the cached residency decision on the following solve too.
        repeated = dict(base)
        C_repeated = solve_homogenization(repeated)
    assert len(observations) == 2
    assert np.array_equal(C, C_reference) and np.array_equal(C_repeated, C_reference)
    for params in (candidate, repeated):
        for actual, expected in zip(
            params["_solution_fields_result"], reference["_solution_fields_result"]
        ):
            assert np.array_equal(actual, expected)


def test_small_new_grid_defers_compile_then_uses_fused(tmp_path, monkeypatch):
    import json

    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "background")
    monkeypatch.setattr(fused, "_READY", set())
    monkeypatch.setattr(fused, "_BACKGROUND", {})
    rng = np.random.default_rng(17)
    phase = (rng.random((10, 14, 22)) < 0.3).astype(
        np.uint8
    )  # a grid no other test compiles
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
    )
    fused._KERNELS.clear()
    fused._PIPELINE_PARTS.clear()

    def run(tag):
        with contextlib.redirect_stdout(io.StringIO()):
            C = solve_homogenization(
                dict(base, solver_timing_path=str(tmp_path / f"{tag}.json"))
            )
        infos = json.loads((tmp_path / f"{tag}.json").read_text())[
            "cufhom_core_application_timing"
        ]["primaldual"]["primal"]["load_solver_infos"]
        return C, infos

    C_first, infos = run("first")
    assert not any(i["linear_fused_fft"] for i in infos)
    for worker in fused._BACKGROUND.values():
        worker.join(60)
    C_second, infos = run("second")
    assert all(i["linear_fused_fft"] for i in infos)
    assert np.linalg.norm(C_second - C_first) <= 1e-6 * np.linalg.norm(C_first)
    cache = json.loads((tmp_path / "kernels.json").read_text())
    assert any(k.startswith("compiled:") for k in cache)


def test_host_loop_solve_matches_graph_solve_bitwise(tmp_path, monkeypatch):
    """Large grids run the host-driven loop; it must give the graph result."""
    import json

    from cufhom.fft.linear.solver import solve_homogenization

    monkeypatch.setenv("CUFHOM_KERNEL_CACHE", str(tmp_path / "kernels.json"))
    rng = np.random.default_rng(19)
    phase = (rng.random((12, 10, 16)) < 0.3).astype(np.uint8)
    ori = rng.standard_normal(phase.shape + (3,)).astype(np.float32)
    ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    base = dict(
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
    )
    results = []
    for graphs in ("1", "0"):
        monkeypatch.setenv("CUFHOM_CUDA_GRAPHS", graphs)
        timing = tmp_path / f"graphs{graphs}.json"
        with contextlib.redirect_stdout(io.StringIO()):
            C = solve_homogenization(dict(base, solver_timing_path=str(timing)))
        infos = json.loads(timing.read_text())["cufhom_core_application_timing"][
            "primaldual"
        ]["primal"]["load_solver_infos"]
        assert all(i["linear_fused_fft"] for i in infos)
        results.append((C, [i["cg_iterations"] for i in infos]))
    assert (
        np.array_equal(results[0][0], results[1][0]) and results[0][1] == results[1][1]
    )
