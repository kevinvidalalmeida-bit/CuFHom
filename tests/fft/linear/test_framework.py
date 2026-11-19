"""Shared public contract, specialized plans, and legacy GaNi compatibility."""

import numpy as np
import pytest
from cufhom.fft.linear.common import IndexedMaterial, SolveOptions, audit_record
from cufhom.fft.linear.framework import homogenize
from cufhom.fft.linear.hex8_general import homogenize_hex8


def case(shape=(3, 4, 5), dtype="float64"):
    rng = np.random.default_rng(61)
    a = rng.normal(size=(4, 6, 6))
    c = a.transpose(0, 2, 1) @ a + 3 * np.eye(6)
    return IndexedMaterial(rng.integers(0, 4, shape), c, dtype=dtype)


def test_material_owns_identical_precision_rounded_coefficients():
    index = np.zeros((2, 3, 4), np.int32)
    table = np.eye(6)[None] * 1.234567891
    data = IndexedMaterial(index, table, dtype="float32")
    table[:] = 0
    index[:] = 7
    np.testing.assert_array_equal(data.index, 0)
    np.testing.assert_array_equal(np.diag(data.stiffness[0]), np.float32(1.234567891))
    assert not data.index.flags.writeable and not data.packed.flags.writeable
    rows, cols = np.triu_indices(6)
    np.testing.assert_array_equal(data.packed, data.stiffness[:, rows, cols])
    with pytest.raises(ValueError, match="real"):
        IndexedMaterial(index, np.eye(6, dtype=complex)[None] + 1j)
    with pytest.raises(ValueError, match="outside"):
        IndexedMaterial(index, np.eye(6)[None])


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, np.nan])
def test_invalid_budget(bad):
    with pytest.raises(ValueError):
        SolveOptions(maxiter=bad)


def test_audit_is_native_and_zero_rhs_is_explicit():
    a = audit_record(
        metric="projected_stress_RMS",
        residual=0,
        rhs_norm=0,
        threshold=0,
        iterations=0,
        accepted=True,
        native={},
    )
    assert a["relative"] is None and a["accepted"]
    with pytest.raises(RuntimeError, match="rejected"):
        audit_record(
            metric="nodal_force_Linf",
            residual=1e-4,
            rhs_norm=1,
            threshold=1e-6,
            iterations=10,
            accepted=True,
            native={},
        )


@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_homogeneous_rectangular_anisotropic_patch(kind):
    data = case()
    homogeneous = IndexedMaterial(np.zeros(data.shape, int), data.stiffness[:1])
    result = homogenize(
        homogeneous,
        discretization=kind,
        lengths=(0.8, 1.3, 0.9),
        backend="numpy",
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        result["C"], homogeneous.stiffness[0], rtol=1e-12, atol=1e-12
    )
    assert len(result["audits"]) == 6
    assert all(a["accepted"] for a in result["audits"])
    assert result["continuum_error_certified"] is False
    assert result["mandel_order"] == ("11", "22", "33", "23", "13", "12")


def test_hex8_reference_and_optimized_share_solver_and_results():
    data = case((3, 3, 3))
    optimized = homogenize(data, discretization="hex8", backend="numpy", rtol=1e-10)
    reference = homogenize(
        data, discretization="hex8", backend="numpy", rtol=1e-10, execution="reference"
    )
    native = homogenize_hex8(data.index, data.stiffness, backend="numpy", rtol=1e-10)
    np.testing.assert_array_equal(optimized["C"], native["C"])
    np.testing.assert_allclose(optimized["C"], reference["C"], atol=3e-13, rtol=3e-13)
    assert reference["execution"] == "staged_quadrature"
    assert reference["iterations"] == optimized["iterations"]


def test_gani_indexed_input_matches_legacy_dense_material_without_patches():
    from cufhom.fft.linear.solver import solve_homogenization
    from cufhom_core.tensors import fft

    data = case()
    full = np.moveaxis(data.stiffness[data.index], (-2, -1), (0, 1))
    p = dict(
        phase_array=np.zeros(data.shape, np.uint8),
        ori_array=np.zeros(data.shape + (3,)),
        Em=1.0,
        nu_m=0.3,
        Ef_L=2.0,
        Ef_T=2.0,
        G_LT=1.0,
        nu_LT=0.2,
        nu_TT=0.2,
        Cfield_array=full,
        solver_real_dtype="float64",
        solver_rtol=1e-10,
        fft_backend="scipy",
    )
    expected = solve_homogenization(p)
    previous = fft.FFT_BACKEND
    actual = homogenize(data, backend="numpy", rtol=1e-10)
    assert fft.FFT_BACKEND == previous
    np.testing.assert_array_equal(actual["C"], expected)
    assert all(a["metric"] == "projected_stress_RMS" for a in actual["audits"])


def test_unsupported_capabilities_are_not_silent_fallbacks():
    data = case()
    with pytest.raises(ValueError, match="isotropic reference"):
        homogenize(data, reference=data.stiffness[0], backend="numpy")
    with pytest.raises(ValueError, match="automatic"):
        homogenize(data, backend="numpy", storage="host")
    with pytest.raises(ValueError, match="precision"):
        homogenize(data, dtype="float32", backend="numpy")
    with pytest.raises(ValueError, match="discretization"):
        homogenize(data, discretization="unknown")


@pytest.mark.parametrize(
    "backend", ["numpy", pytest.param("cupy", marks=pytest.mark.gpu)]
)
def test_stored_fp32_reference_inverse_must_remain_positive(backend):
    from cufhom.fft.linear.hex8_general import Hex8Operator

    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    q = np.linalg.qr(np.random.default_rng(448).normal(size=(3, 3)))[0]
    c0 = np.eye(6)
    rows = [0, 5, 4]
    c0[np.ix_(rows, rows)] = q @ np.diag([1.0, 1.0, 1e-10]) @ q.T
    with pytest.raises(ValueError, match="positive definite"):
        Hex8Operator(
            np.zeros((4, 4, 4), int),
            np.eye(6)[None],
            reference=c0,
            dtype="float32",
            backend=backend,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,rtol,tolerance", [("float32", 1e-5, 3e-6), ("float64", 1e-8, 2e-11)]
)
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_cuda_unified_entry_preserves_specialized_operator(
    dtype, rtol, tolerance, kind, monkeypatch
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    data = case((12, 10, 14), dtype)
    cpu_data = IndexedMaterial(data.index, data.stiffness, dtype="float64")
    cpu = homogenize(cpu_data, discretization=kind, backend="numpy", rtol=1e-10)
    gpu = homogenize(data, discretization=kind, rtol=rtol)
    np.testing.assert_allclose(gpu["C"], cpu["C"], atol=tolerance * 10, rtol=tolerance)
    assert all(i["accepted"] and i["residual"] <= i["threshold"] for i in gpu["audits"])
    if kind == "gani":
        assert gpu["execution"] == "generated_fft"
        assert all(i["linear_fused_fft"] for i in gpu["infos"])
    else:
        reference = homogenize(
            data, discretization=kind, rtol=rtol, execution="reference"
        )
        np.testing.assert_allclose(
            reference["C"], gpu["C"], atol=tolerance * 10, rtol=tolerance
        )
        if dtype == "float64":
            assert gpu["iterations"] == reference["iterations"]


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
def test_cuda_entry_mixed_default_energy_bounds_and_hex8_energy_criterion(kind):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    x, y, z = np.indices((16, 18, 20))
    index = ((x - 8) ** 2 + (y - 9) ** 2 + (z - 10) ** 2 < 30).astype(np.int32)
    mu, bulk = 2.0, 5.0
    c0 = 2 * mu * np.eye(6)
    c0[:3, :3] += bulk - 2 * mu / 3
    data = IndexedMaterial(index, np.stack([c0, 15 * c0]), dtype="float64")
    exact = homogenize(
        data, discretization=kind, reference=c0, rtol=1e-11, working="float64"
    )["C"]
    mixed = homogenize(data, discretization=kind, reference=c0, rtol=1e-6)
    assert (
        mixed["working"] == "float32"
        and mixed["precision_contract"]["krylov"] == "float32"
    )
    error = np.diag(mixed["C"] - exact)
    bound = mixed["energy_error_bound"]
    assert np.all(
        error <= np.array(bound["upper"]) * (1 + 1e-6) + 1e-14 * np.abs(exact).max()
    )
    assert np.all(
        error >= np.array(bound["lower"]) * (1 - 1e-6) - 1e-14 * np.abs(exact).max()
    )
    for tau in (1e-5, 1e-8):
        energy = homogenize(data, discretization=kind, reference=c0, energy_rtol=tau)
        assert all(a["metric"] == "energy_error_upper" for a in energy["audits"])
        error = energy["C"] - exact
        assert np.all(np.diag(error) <= tau * np.diag(exact))
        assert np.linalg.eigvalsh(error)[0] >= -1e-13 * np.abs(exact).max()
        assert sum(energy["iterations"]) < sum(mixed["iterations"])


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["gani", "hex8"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_cuda_energy_criterion_in_fixed_precision(kind, dtype):
    """The energy criterion is not tied to the mixed route: the fixed FP64 and
    FP32 configurations accept on the same fresh bound, and the accepted
    candidate energy is the assembled diagonal."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    x, y, z = np.indices((16, 18, 20))
    index = ((x - 8) ** 2 + (y - 9) ** 2 + (z - 10) ** 2 < 30).astype(np.int32)
    mu, bulk = 2.0, 5.0
    c0 = 2 * mu * np.eye(6)
    c0[:3, :3] += bulk - 2 * mu / 3
    table = np.stack([c0, 15 * c0])
    exact = homogenize(
        IndexedMaterial(index, table, dtype="float64"),
        discretization=kind,
        reference=c0,
        rtol=1e-11,
        working="float64",
    )["C"]
    tau, rounding = (
        (1e-8, 1e-12) if dtype == "float64" else (1e-4, 1e-5)
    )  # FP32 coefficients and assembly
    result = homogenize(
        IndexedMaterial(index, table, dtype=dtype),
        discretization=kind,
        reference=c0,
        energy_rtol=tau,
        working=dtype,
    )
    assert result["working"] == dtype
    assert all(
        a["metric"] == "energy_error_upper" and a["accepted"] for a in result["audits"]
    )
    error = np.diag(result["C"] - exact)
    assert np.all(error <= (tau + rounding) * np.diag(exact))
    assert np.all(error >= -rounding * np.diag(exact))
    np.testing.assert_allclose(
        [i["energy"] for i in result["infos"]], np.diag(result["C"]), rtol=10 * rounding
    )


def test_energy_criterion_requires_cuda():
    with pytest.raises(ValueError, match="CUDA PCG"):
        homogenize(
            np.zeros((2, 2, 2), int),
            np.eye(6)[None],
            discretization="gani",
            backend="numpy",
            energy_rtol=1e-6,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["fused", "library"])
@pytest.mark.parametrize("phases", [2, 9])
def test_cuda_gani_mixed_load_evaluation_matches_fp64_route(monkeypatch, route, phases):
    """The fused load evaluation at z = 0 and, on the library route, the
    indicator-spectra load (P <= 6) or the z=0 shortcut (P > 6)."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import cufhom.fft.linear.coordinates as coordinates
    import cufhom.fft.linear.framework as framework

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    if route == "library":
        monkeypatch.setattr(coordinates, "_evaluation_pipeline", lambda *args: None)
    else:  # a compact (here host-resident) load is evaluated by the fused pipeline
        monkeypatch.setattr(
            framework,
            "_gani_residency",
            lambda material, configuration="mixed", **kwargs: (
                "host" if configuration == "mixed" else "iterate"
            ),
        )
    rng = np.random.default_rng(phases)
    index = rng.integers(0, phases, (18, 16, 20)).astype(np.int32)
    a = rng.normal(size=(phases, 6, 6))
    table = a.transpose(0, 2, 1) @ a + 4 * np.eye(6)
    data = IndexedMaterial(index, table, dtype="float64")
    mixed = homogenize(data, discretization="gani", rtol=1e-8)
    pure = homogenize(data, discretization="gani", rtol=1e-8, working="float64")
    expected = (
        "fused"
        if route == "fused"
        else "indicators"
        if phases <= 6
        else "six_components"
    )
    assert all(
        i["load_evaluation"] == expected for i in mixed["native_report"]["infos"]
    )
    assert all(
        i.get("phase_spectra_rhs") == (expected == "indicators") for i in mixed["infos"]
    )
    np.testing.assert_allclose(
        mixed["C"], pure["C"], rtol=1e-12, atol=1e-12 * np.abs(pure["C"]).max()
    )
    for x, y in zip(mixed["infos"], pure["infos"]):
        assert abs(x["rhs_norm"] / y["rhs_norm"] - 1) < 1e-13
    for result in (mixed, pure):  # one per-load schema for both GaNi routes
        assert result["iterations"] == [i["cg_iterations"] for i in result["infos"]]
        assert all(
            np.isfinite(i["final_norm_res"]) and i["energy_G0"] > 0
            for i in result["infos"]
        )
        assert "assembly" in result["memory"]


@pytest.mark.gpu
def test_cuda_device_free_bytes_counts_cached_pool_blocks():
    """Residency policies see the same memory before and after a cached call."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear.common import device_free_bytes

    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = device_free_bytes()
    block = cp.empty(256 << 20, cp.uint8)
    del block
    assert pool.total_bytes() - pool.used_bytes() >= 256 << 20
    assert abs(device_free_bytes() - before) < 32 << 20


@pytest.mark.gpu
@pytest.mark.parametrize("energy", [None, 1e-8])
def test_cuda_gani_mixed_host_solution_is_bitwise(monkeypatch, energy):
    """FP64 coordinates in pinned host memory, streamed in several slabs per
    evaluation, reproduce the device-resident route exactly."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import cufhom.fft.linear.coordinates as coordinates
    import cufhom.fft.linear.framework as framework

    monkeypatch.setenv(
        "CUFHOM_FUSED_COMPILE", "sync"
    )  # the host route needs the pipeline layout
    monkeypatch.setattr(coordinates, "_SLAB_BYTES", 1 << 14)
    rng = np.random.default_rng(3)
    index = rng.integers(0, 3, (16, 12, 20)).astype(np.int32)
    a = rng.normal(size=(3, 6, 6))
    data = IndexedMaterial(
        index, a.transpose(0, 2, 1) @ a + 4 * np.eye(6), dtype="float64"
    )
    results = {}
    for residency in ("device", "host"):
        monkeypatch.setattr(
            framework,
            "_gani_residency",
            lambda material, configuration="mixed", energy=False, library=False, residency=residency: (
                residency
            ),
        )
        results[residency] = homogenize(
            data, discretization="gani", rtol=1e-8, energy_rtol=energy
        )
    device, host = results["device"], results["host"]
    assert host["memory"]["solution"] == "host" and host["memory"]["retained"] == "host"
    assert host["iterations"] == device["iterations"]
    np.testing.assert_array_equal(host["C"], device["C"])


@pytest.mark.gpu
@pytest.mark.parametrize("configuration", ["mixed", "float32", "float64"])
@pytest.mark.parametrize("energy", [None, 1e-8])
def test_cuda_gani_host_iterate_matches_the_device_route(
    monkeypatch, configuration, energy
):
    """The Krylov iterate and its smoothed copy in pinned host memory, a third
    of their rows resident and the others streamed in several slabs per
    iteration, follow the device-resident route: same kernels per element,
    only the FP64 energy products split at the resident rows."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import cufhom.fft.linear.coordinates as coordinates
    import cufhom.fft.linear.framework as framework

    monkeypatch.setenv(
        "CUFHOM_FUSED_COMPILE", "sync"
    )  # the host route needs the pipeline layout
    monkeypatch.setattr(coordinates, "_SLAB_BYTES", 1 << 14)
    monkeypatch.setattr(
        coordinates,
        "_resident_rows",
        lambda shape, configuration, smoothing, shared=False: shape[1] // 3,
    )
    rng = np.random.default_rng(3)
    index = rng.integers(0, 3, (16, 12, 20)).astype(np.int32)
    a = rng.normal(size=(3, 6, 6))
    dtype = "float32" if configuration == "float32" else "float64"
    data = IndexedMaterial(index, a.transpose(0, 2, 1) @ a + 4 * np.eye(6), dtype=dtype)
    results = {}
    for residency in ("device", "iterate"):
        monkeypatch.setattr(
            framework,
            "_gani_residency",
            lambda material, configuration="mixed", energy=False, library=False, residency=residency: (
                residency
            ),
        )
        results[residency] = homogenize(
            data,
            discretization="gani",
            rtol=1e-5 if dtype == "float32" else 1e-8,
            working="float32" if configuration == "mixed" else dtype,
            energy_rtol=energy,
        )
    device, host = results["device"], results["iterate"]
    assert host["memory"]["iterate"] == "host" and host["memory"]["retained"] == "host"
    assert all(i["iterate_resident_rows"] == 4 for i in host["native_report"]["infos"])
    assert host["iterations"] == device["iterations"]
    np.testing.assert_allclose(host["C"], device["C"], rtol=1e-12, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("configuration", ["mixed", "float32", "float64"])
@pytest.mark.parametrize("energy", [False, True])
def test_cuda_hex8_host_residency_matches_the_device_route(
    monkeypatch, configuration, energy
):
    """The HEX8 solution in pinned host memory (mixed: the FP64 solution; FP32
    and FP64: the Krylov iterate, updated on a side stream every iteration),
    a third of its planes resident and the others in several slabs, gathered
    for every fresh evaluation in the memory of z and the spectrum, follows
    the device-resident route."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import cufhom.fft.linear.coordinates as coordinates
    import cufhom.fft.linear.hex8_general as hex8

    monkeypatch.setenv(
        "CUFHOM_FUSED_COMPILE", "sync"
    )  # the iterate level needs the fused PCG
    monkeypatch.setattr(coordinates, "_SLAB_BYTES", 1 << 14)
    monkeypatch.setattr(hex8, "_hex8_resident_rows", lambda op, pcg: op.shape[0])
    x, y, z = np.indices((16, 18, 20))
    index = ((x - 8) ** 2 + (y - 9) ** 2 + (z - 10) ** 2 < 30).astype(np.int32)
    mu, bulk = 2.0, 5.0
    c0 = 2 * mu * np.eye(6)
    c0[:3, :3] += bulk - 2 * mu / 3
    dtype = "float32" if configuration == "float32" else "float64"
    data = IndexedMaterial(index, np.stack([c0, 15 * c0]), dtype=dtype)
    options = (
        dict(energy_rtol=1e-4 if dtype == "float32" else 1e-8)
        if energy
        else dict(rtol=1e-4 if dtype == "float32" else 1e-8)
    )
    results = {}
    for level in ("device", "host" if configuration == "mixed" else "iterate"):
        monkeypatch.setattr(hex8, "hex8_residency", lambda op, level=level: level)
        results[level] = homogenize(
            data,
            discretization="hex8",
            reference=c0,
            working="float32" if configuration == "mixed" else dtype,
            **options,
        )
    device, host = results.values()
    assert (
        host["native_report"]["residency"] != "device"
        and host["memory"]["retained"] == "host"
    )
    assert host["iterations"] == device["iterations"]
    # Retained products are summed in host chunks: rounding-level differences. In the
    # FP32 configuration the device route updates the iterate with FP32 FMAs (FAST32)
    # while the host route keeps its own update order, so near-zero entries differ at
    # 1e-13-1e-11 of the matrix scale, far below the FP32 accuracy of the solve.
    fp32 = configuration == "float32"
    np.testing.assert_allclose(
        host["C"],
        device["C"],
        rtol=1e-9 if fp32 else 1e-12,
        atol=(1e-9 if fp32 else 1e-13) * np.abs(device["C"]).max(),
    )


@pytest.mark.gpu
def test_cuda_hex8_residency_policy():
    """HEX8 stays on the device while its load fits, then keeps its solution
    (mixed: FP64; fixed: the Krylov iterate) in pinned host memory; that
    level needs less device memory, and FP64 reaches 512^3 within 15 GiB."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from types import SimpleNamespace

    import cufhom.fft.linear.hex8_general as hex8

    def op(n, working, dtype):
        return SimpleNamespace(
            shape=(n,) * 3, n=n**3, working=np.dtype(working), dtype=np.dtype(dtype)
        )

    for working, dtype in (
        ("float32", "float64"),
        ("float32", "float32"),
        ("float64", "float64"),
    ):
        assert hex8.hex8_residency(op(64, working, dtype)) == "device"
        assert hex8.hex8_residency(op(4096, working, dtype)) is None
        big = op(512, working, dtype)
        assert hex8.hex8_working_bytes(
            big, "host" if working != dtype else "iterate"
        ) < hex8.hex8_working_bytes(big)
    assert hex8.hex8_working_bytes(op(512, "float64", "float64"), "iterate") < 15 << 30


@pytest.mark.gpu
def test_cuda_gani_residency_policy():
    """Each configuration stays on the device while its load fits, then moves
    its solution and Krylov iterate to pinned host memory (the mixed FP64
    coordinates first); every level needs less device memory than the one
    before, and FP64 reaches 512^3 within 15 GiB."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear.coordinates import mixed_residency, mixed_working_bytes

    for configuration in ("mixed", "float32", "float64"):
        assert mixed_residency((64,) * 3, configuration) == "device"
        assert mixed_residency((4096,) * 3, configuration) is None
        levels = (
            ("device", "host", "iterate")
            if configuration == "mixed"
            else ("device", "iterate")
        )
        for energy in (False, True):
            sizes = [
                mixed_working_bytes((512,) * 3, level, configuration, energy)
                for level in levels
            ]
            assert all(a > b for a, b in zip(sizes, sizes[1:]))
    assert mixed_working_bytes((512,) * 3, "iterate", "float64") < 15 << 30
    with pytest.raises(ValueError, match="residency"):
        mixed_working_bytes((64,) * 3, "pinned")


@pytest.mark.gpu
def test_cuda_gani_mixed_growth_guard_recovers_a_single_fp32_segment(monkeypatch):
    """One FP32 segment to 1e-9 passes the FP32 recursion floor; the growth
    guard replaces the residual and the fresh FP64 audit still accepts."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import cufhom.fft.linear.coordinates as coordinates

    rng = np.random.default_rng(5)
    index = rng.integers(0, 2, (16, 18, 20)).astype(np.int32)
    table = np.stack([np.eye(6), 30 * np.eye(6)]) + np.ones((2, 6, 6))
    data = IndexedMaterial(index, table, dtype="float64")
    reference = homogenize(data, discretization="gani", rtol=1e-9, working="float64")
    monkeypatch.setattr(coordinates, "_FP32_SEGMENT", 1e-12)
    mixed = homogenize(data, discretization="gani", rtol=1e-9)
    assert all(
        i["reliable_drop"] == 0 and i["true_residual_converged"] for i in mixed["infos"]
    )
    assert all(i["fp64_evaluations"] > 2 for i in mixed["infos"])  # guard replacements
    np.testing.assert_allclose(
        mixed["C"],
        reference["C"],
        rtol=1e-10,
        atol=1e-10 * np.abs(reference["C"]).max(),
    )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,lengths,tile",
    [
        ((16, 12, 20), (1.0, 1.0, 1.0), 2),
        ((15, 9, 11), (1.3, 0.7, 1.1), None),
        ((24, 20, 32), (0.9, 1.2, 1.0), 4),
    ],
)
def test_cuda_fused_residual_audit_equals_projected_distance(shape, lengths, tile):
    """||G h||^2 from the FP64 coordinate residual equals the projected
    spectral distance; the stored FP32 residual is that of the plain map."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear import coordinates
    from cufhom.fft.linear.active_capacity import ActiveCapacity
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom.fft.operators import MixedProjector
    from cufhom_core.tensors import Tensor

    Q = SymmetricCoordinates(
        MixedProjector(shape, lengths, np.ones(6, bool), cp), 2.0, 1.3
    )
    Q = Q.tiled(tile) if tile else Q
    descriptor = Tensor(
        name="A",
        val=cp.zeros((1,) + shape, cp.int32),
        order=1,
        N=np.array(shape),
        Y=np.array(lengths),
        multype="sym21_indexed",
        Fourier=False,
        origin=0,
        fft_form="r",
    )
    descriptor.material_table = cp.ones((1, 21))
    operator = ActiveCapacity(descriptor, shape, lengths, Q)
    _, h, v = operator.work()
    v[...] = cp.asarray(np.random.default_rng(1).normal(size=v.shape))
    operator.forward(v, h)
    r, plain = cp.empty(Q.size, cp.complex64), cp.empty(Q.size, cp.complex64)
    metric = coordinates._residual_map(Q, h, r, audit=True)
    coordinates._residual_map(Q, h, plain)
    operator.project(h)
    assert abs(metric / coordinates._spectral_distance(h, shape) - 1) < 1e-13
    assert bool((r == plain).all())


@pytest.mark.gpu
@pytest.mark.parametrize("arithmetic", ["df64", "float64"])
@pytest.mark.parametrize(
    "shape,lengths", [((24, 20, 32), (1.0, 1.0, 1.0)), ((30, 18, 28), (1.3, 0.7, 1.1))]
)
def test_cuda_fused_evaluation_matches_library_fp64(
    monkeypatch, arithmetic, shape, lengths
):
    """Total-strain residual, audit and stress sum of the fused evaluation
    pipeline against cuFFT FP64 (library route of the mixed GaNi load)."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear import coordinates, evaluation
    from cufhom.fft.linear.active_capacity import ActiveCapacity
    from cufhom.fft.linear.common import SYM21_PAIRS
    from cufhom.fft.linear.fused import default_config
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom.fft.operators import MixedProjector
    from cufhom_core.tensors import Tensor

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    rng = np.random.default_rng(4)
    tile = default_config(shape, 99 * 1024, np.float32)["ta"]
    Q = SymmetricCoordinates(
        MixedProjector(shape, lengths, np.ones(6, bool), cp), 2.0, 1.3
    ).tiled(tile)
    a = rng.normal(size=(3, 6, 6))
    c = a.transpose(0, 2, 1) @ a + 4 * np.eye(6)
    descriptor = Tensor(
        name="A",
        val=cp.asarray(rng.integers(0, 3, (1,) + shape), cp.int32),
        order=1,
        N=np.array(shape),
        Y=np.array(lengths),
        multype="sym21_indexed",
        Fourier=False,
        origin=0,
        fft_form="r",
    )
    descriptor.material_table = cp.asarray(
        np.stack([c[:, i, j] for i, j in SYM21_PAIRS], 1)
    )
    z = cp.asarray(rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size))
    macro = rng.normal(size=6)
    operator = ActiveCapacity(descriptor, shape, lengths, Q)
    w, h, v = operator.work()
    operator.map(z, h, False)
    operator.inverse(h, v)
    v += cp.asarray(macro)[:, None, None, None]
    operator.constitutive(w)
    operator.forward(v, h)
    total = cp.asnumpy(h[:, 0, 0, 0].real)
    reference = cp.empty(Q.size, cp.complex64)
    audit = coordinates._residual_map(Q, h, reference, True)
    pipeline = evaluation.EvaluationPipeline(Q, descriptor, arithmetic)
    assert pipeline.ready
    r = cp.empty(Q.size, cp.complex64)
    pipeline.inverse(z)
    rr, metric, mean = pipeline.forward(macro, r)
    tol = 1e-13 if arithmetic == "df64" else 1e-15
    assert (
        abs(metric / audit - 1) < tol
        and np.abs(mean - total).max() < tol * np.abs(total).max()
    )
    assert float(cp.abs(r - reference).max()) <= 2**-23 * float(cp.abs(reference).max())
    exact = float(cp.vdot(r.astype(cp.complex128), r.astype(cp.complex128)).real)
    assert abs(rr / exact - 1) < 1e-14


@pytest.mark.gpu
@pytest.mark.parametrize(
    "arithmetic,solution,residual",
    [
        ("df64", "float32", "float32"),
        ("float64", "float32", "float32"),
        ("float64", "float64", "float64"),
    ],
)
def test_cuda_fused_evaluation_in_the_fixed_configurations(
    monkeypatch, arithmetic, solution, residual
):
    """The FP32 and FP64 configurations evaluate their working-precision
    solution directly and store the residual in it; the FP32 library map
    widens the solution slab by slab, exactly as a full FP64 copy."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear import coordinates, evaluation
    from cufhom.fft.linear.active_capacity import ActiveCapacity
    from cufhom.fft.linear.common import SYM21_PAIRS
    from cufhom.fft.linear.fused import default_config
    from cufhom.fft.linear.krylov import SymmetricCoordinates
    from cufhom.fft.operators import MixedProjector
    from cufhom_core.tensors import Tensor

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    monkeypatch.setattr(
        coordinates, "_SLAB_BYTES", 1 << 16
    )  # several slabs on a small grid
    shape, lengths = (30, 18, 28), (1.3, 0.7, 1.1)
    rng = np.random.default_rng(7)
    tile = default_config(shape, 99 * 1024, np.float32)["ta"]
    Q = SymmetricCoordinates(
        MixedProjector(shape, lengths, np.ones(6, bool), cp), 2.0, 1.3
    ).tiled(tile)
    a = rng.normal(size=(3, 6, 6))
    c = a.transpose(0, 2, 1) @ a + 4 * np.eye(6)
    descriptor = Tensor(
        name="A",
        val=cp.asarray(rng.integers(0, 3, (1,) + shape), cp.int32),
        order=1,
        N=np.array(shape),
        Y=np.array(lengths),
        multype="sym21_indexed",
        Fourier=False,
        origin=0,
        fft_form="r",
    )
    descriptor.material_table = cp.asarray(
        np.stack([c[:, i, j] for i, j in SYM21_PAIRS], 1)
    )
    complex_type = cp.complex64 if solution == "float32" else cp.complex128
    z = cp.asarray(rng.normal(size=Q.size) + 1j * rng.normal(size=Q.size)).astype(
        complex_type
    )
    macro = rng.normal(size=6)
    operator = ActiveCapacity(descriptor, shape, lengths, Q)
    w, h, v = operator.work()
    coordinates._map_solution(Q, operator, z, h)
    widened = cp.empty_like(h)
    operator.map(z.astype(cp.complex128), widened, False)
    assert bool((h == widened).all())
    operator.inverse(h, v)
    v += cp.asarray(macro)[:, None, None, None]
    operator.constitutive(w)
    operator.forward(v, h)
    total = cp.asnumpy(h[:, 0, 0, 0].real)
    reference = cp.empty(
        Q.size, cp.complex64 if residual == "float32" else cp.complex128
    )
    audit = coordinates._residual_map(Q, h, reference, True)
    pipeline = evaluation.EvaluationPipeline(
        Q, descriptor, arithmetic, solution=solution, residual=residual
    )
    r = cp.empty_like(reference)
    pipeline.inverse(z)
    rr, metric, mean = pipeline.forward(macro, r)
    tol = 1e-13 if arithmetic == "df64" else 1e-15
    assert (
        abs(metric / audit - 1) < tol
        and np.abs(mean - total).max() < tol * np.abs(total).max()
    )
    rounding = (
        2**-23 if residual == "float32" else 1e-12
    )  # FP64: Stockham versus cuFFT order
    assert float(cp.abs(r - reference).max()) <= rounding * float(
        cp.abs(reference).max()
    )
    with pytest.raises(ValueError, match="precision"):
        pipeline.inverse(
            z.astype(cp.complex128 if solution == "float32" else cp.complex64)
        )


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_gani_fixed_configurations_match_the_core_route(dtype, monkeypatch):
    """One coordinate route for every configuration: FP64 reproduces the core
    strain-field route to rounding; FP32 differs only by its more accurate
    df64 evaluation and identity assembly."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear import framework
    from cufhom.fft.linear.common import SolveOptions

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    rng = np.random.default_rng(11)
    index = rng.integers(0, 3, (20, 18, 22)).astype(np.int32)
    a = rng.normal(size=(3, 6, 6))
    data = IndexedMaterial(index, a.transpose(0, 2, 1) @ a + 4 * np.eye(6), dtype=dtype)
    rtol = 1e-8 if dtype == "float64" else 1e-5
    unified = homogenize(data, discretization="gani", rtol=rtol, working=dtype)
    core = framework._gani(
        data,
        (1.0, 1.0, 1.0),
        SolveOptions("cupy", rtol, 0.0, 3000, "auto"),
        None,
        np.dtype(dtype),
        None,
    )
    assert unified["memory"]["assembly"] == "identity_fp32_coordinates"
    assert unified["precision_contract"]["evaluation"] == (
        ["float64"] if dtype == "float64" else ["df64"]
    )
    scale = np.abs(core["C"]).max()
    np.testing.assert_allclose(
        unified["C"],
        core["C"],
        rtol=0,
        atol=(1e-12 if dtype == "float64" else 1e-6) * scale,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("energy", [None, 1e-8])
def test_cuda_gani_mixed_evaluation_arithmetics(monkeypatch, energy):
    """df64 and float64 fused evaluations accept the same solution as an FP64 run."""
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    rng = np.random.default_rng(6)
    index = rng.integers(0, 3, (16, 12, 20)).astype(np.int32)
    a = rng.normal(size=(3, 6, 6))
    data = IndexedMaterial(
        index, a.transpose(0, 2, 1) @ a + 4 * np.eye(6), dtype="float64"
    )
    reference = homogenize(data, discretization="gani", rtol=1e-11, working="float64")
    results = {
        e: homogenize(
            data, discretization="gani", rtol=1e-8, energy_rtol=energy, evaluation=e
        )
        for e in ("df64", "float64")
    }
    for arithmetic, result in results.items():
        assert result["precision_contract"]["evaluation"] == [arithmetic]
        np.testing.assert_allclose(
            result["C"],
            reference["C"],
            rtol=0,
            atol=1e-8 * np.abs(reference["C"]).max(),
        )
    assert results["df64"]["iterations"] == results["float64"]["iterations"]
    with pytest.raises(ValueError, match="evaluation"):
        homogenize(data, discretization="gani", evaluation="float16")
