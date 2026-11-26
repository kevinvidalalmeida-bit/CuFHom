"""General HEX8: dense FE assembly, exact quadrature, all modes, CPU/CUDA."""

import numpy as np
import pytest
from cufhom.fft.linear.hex8_general import (
    _NODES,
    Hex8Operator,
    cp,
    hex8_shape_gradients,
    homogenize_hex8,
    solve_hex8_load,
)


def material(seed=17, phases=4):
    a = np.random.default_rng(seed).normal(size=(phases, 6, 6))
    return a.transpose(0, 2, 1) @ a + 2 * np.eye(6)


def dense(op):
    k = np.zeros((3 * op.n, 3 * op.n))
    rhs = np.zeros((3 * op.n, 6))
    # Independent nodal scatter with all Gauss points, not the gather table.
    for cell in np.ndindex(op.shape):
        ids = [
            np.ravel_multi_index(tuple((np.array(cell) + node) % op.shape), op.shape)
            for node in _NODES
        ]
        dofs = np.array([c * op.n + node for node in ids for c in range(3)])
        for bg in op.B.reshape(8, 6, 24):
            c = op.C[op.index[cell]]
            k[np.ix_(dofs, dofs)] += bg.T @ c @ bg / (8 * op.n)
            rhs[dofs] -= bg.T @ c / (8 * op.n)
    return k, rhs


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 3, 3)])
def test_anisotropic_action_rhs_and_six_load_energy(shape):
    rng = np.random.default_rng(31)
    labels = rng.integers(0, 4, shape)
    op = Hex8Operator(labels, material(), lengths=(0.7, 1.2, 1.5), backend="numpy")
    k, rhs = dense(op)
    u = rng.normal(size=(3, *shape))
    np.testing.assert_allclose(
        op.action(u).ravel(), k @ u.ravel(), atol=3e-13, rtol=1e-12
    )
    for a in range(6):
        np.testing.assert_allclose(op.rhs(np.eye(6)[a]).ravel(), rhs[:, a], atol=3e-15)
    r = homogenize_hex8(
        labels,
        material(),
        lengths=op.lengths,
        backend="numpy",
        rtol=1e-10,
        return_fields=True,
    )
    reference = np.linalg.pinv(k, rcond=1e-12) @ rhs
    totals = []
    for a in range(6):
        np.testing.assert_allclose(
            r["solutions"][a].ravel(), reference[:, a], atol=2e-10, rtol=1e-7
        )
        assert r["infos"][a]["residual_inf"] <= r["infos"][a]["threshold"]
        fields = []
        for cell in np.ndindex(shape):
            values = np.array(
                [
                    r["solutions"][a][
                        (slice(None), *tuple((np.array(cell) + node) % shape))
                    ]
                    for node in _NODES
                ]
            ).ravel()
            fields.append(
                np.eye(6)[a] + np.einsum("gij,j->gi", op.B.reshape(8, 6, 24), values)
            )
        totals.append(fields)
    total = np.array(totals)
    exact_energy = np.einsum(
        "angi,nij,bngj->ab", total, op.C[labels.ravel()], total
    ) / (8 * op.n)
    np.testing.assert_allclose(r["C"], exact_energy, atol=2e-12)
    assert np.linalg.norm(r["energy_correction"]) < 1e-8


def test_patch_rigid_rotation_and_anisotropic_reference_modes():
    shape = (4, 3, 6)
    lengths = (1.2, 0.8, 1.7)
    b = hex8_shape_gradients(shape, lengths).reshape(8, 6, 24)
    xyz = _NODES * np.array(lengths) / shape
    a = np.array([[0.1, 0.2, 0.3], [0.4, -0.5, 0.6], [0.7, 0.8, -0.9]])
    expected = np.r_[
        np.diag(a),
        (a[1, 2] + a[2, 1]) / np.sqrt(2),
        (a[0, 2] + a[2, 0]) / np.sqrt(2),
        (a[0, 1] + a[1, 0]) / np.sqrt(2),
    ]
    np.testing.assert_allclose(
        np.einsum("gij,j->gi", b, (xyz @ a.T).ravel()),
        np.tile(expected, (8, 1)),
        atol=2e-15,
    )
    np.testing.assert_allclose(
        np.einsum("gij,j->gi", b, (xyz @ (a - a.T).T).ravel()), 0, atol=2e-15
    )
    c = material(phases=1)
    op = Hex8Operator(np.zeros(shape, int), c, lengths=lengths, backend="numpy")
    rng = np.random.default_rng(43)
    u = rng.normal(size=(3, *shape))
    u -= u.mean((1, 2, 3), keepdims=True)
    np.testing.assert_allclose(op.precondition(op.action(u)), u, atol=3e-13)
    np.testing.assert_array_equal(op.action(np.ones_like(u)), 0)
    checker = np.broadcast_to((-1.0) ** np.arange(6), u.shape).copy()
    assert np.linalg.norm(op.action(checker)) > 1
    for macro in np.eye(6):
        np.testing.assert_array_equal(op.rhs(macro), 0)
        assert solve_hex8_load(op, macro)["iterations"] == 0


def test_table_validation_packing_and_failure():
    c = material(phases=3)
    labels = np.indices((3, 4, 5)).sum(0) % 3
    packed = np.stack([c[:, i, j] for i in range(6) for j in range(i, 6)], axis=1)
    a = Hex8Operator(labels, packed, backend="numpy")
    np.testing.assert_array_equal(a.C, c)
    with pytest.raises(ValueError, match="positive definite"):
        Hex8Operator(labels, -c, backend="numpy")
    with pytest.raises(ValueError, match="integer"):
        Hex8Operator(labels.astype(float), c, backend="numpy")
    with pytest.raises(ValueError, match="indices"):
        Hex8Operator(labels + 10, c, backend="numpy")
    with pytest.raises(ValueError, match="symmetric"):
        wrong = c.copy()
        wrong[0, 0, 1] += 1
        Hex8Operator(labels, wrong, backend="numpy")
    with pytest.raises(RuntimeError, match="failed fresh residual"):
        solve_hex8_load(a, np.eye(6)[0], rtol=1e-12, maxiter=1)
    with pytest.raises(ValueError):
        solve_hex8_load(a, np.eye(6)[0], rtol=-1)


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,rtol,agreement", [("float64", 1e-10, 3e-12), ("float32", 2e-5, 3e-6)]
)
def test_cuda_general_material_and_host_storage(dtype, rtol, agreement):
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(71)
    labels = rng.integers(0, 5, (4, 3, 5))
    c = material(phases=5)
    cpu = Hex8Operator(labels, c, lengths=(0.8, 1.3, 1.7), dtype=dtype, backend="numpy")
    gpu = Hex8Operator(labels, c, lengths=cpu.lengths, dtype=dtype, working=dtype)
    u = rng.normal(size=(3, *labels.shape)).astype(dtype)
    np.testing.assert_allclose(
        cp.asnumpy(gpu.action(cp.asarray(u))),
        cpu.action(u),
        atol=agreement * 10,
        rtol=agreement,
    )
    np.testing.assert_allclose(
        cp.asnumpy(gpu.precondition(cp.asarray(u))),
        cpu.precondition(u),
        atol=agreement * 10,
        rtol=agreement,
    )
    np.testing.assert_allclose(
        gpu.mean_stress(cp.asarray(u), np.eye(6)[3]),
        cpu.mean_stress(u, np.eye(6)[3]),
        atol=agreement,
    )
    device = homogenize_hex8(
        labels, c, lengths=cpu.lengths, dtype=dtype, rtol=rtol, storage="device"
    )
    host = homogenize_hex8(
        labels, c, lengths=cpu.lengths, dtype=dtype, rtol=rtol, storage="host"
    )
    reference = homogenize_hex8(
        labels, c, lengths=cpu.lengths, backend="numpy", rtol=1e-11
    )
    np.testing.assert_allclose(
        device["C"], reference["C"], atol=agreement * 10, rtol=agreement
    )
    np.testing.assert_allclose(host["C"], device["C"], atol=agreement)
    assert all(
        i["accepted"] and i["residual_inf"] <= i["threshold"] for i in device["infos"]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("working", ["float64", "float32"])
def test_cuda_call_releases_its_pcg_workspace_without_gc(working):
    """Repeated calls keep the same device memory (no operator/PCG cycle left to gc)."""
    if cp is None or not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA unavailable")
    import gc

    labels = np.random.default_rng(5).integers(0, 3, (16, 12, 20))
    pool = cp.get_default_memory_pool()
    gc.collect()
    gc.disable()
    try:
        used = []
        for _ in range(3):
            homogenize_hex8(
                labels, material(phases=3), dtype="float64", working=working, rtol=1e-6
            )
            used.append(pool.used_bytes())
    finally:
        gc.enable()
    assert used[2] == used[1]


@pytest.mark.gpu
def test_cuda_interior_interface_partition_and_homogeneous_anisotropy():
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    labels = np.zeros((6, 4, 5), np.int32)
    labels[3:] = 1
    c = material(phases=2)
    rng = np.random.default_rng(182)
    u = rng.normal(size=(3, *labels.shape))
    for index, stiffness in [(labels, c), (np.zeros_like(labels), c[:1])]:
        cpu = Hex8Operator(index, stiffness, backend="numpy")
        gpu = Hex8Operator(index, stiffness, working="float64")
        assert gpu.interior_count + gpu.interface_nodes.size == index.size
        np.testing.assert_allclose(
            cp.asnumpy(gpu.action(cp.asarray(u))), cpu.action(u), rtol=1e-12, atol=2e-13
        )
        np.testing.assert_array_equal(
            cp.asnumpy(gpu.action(cp.ones_like(cp.asarray(u)))), 0
        )
        if len(stiffness) == 1:
            assert gpu.interface_nodes.size == 0
            mean_zero = u - u.mean((1, 2, 3), keepdims=True)
            np.testing.assert_allclose(
                cp.asnumpy(gpu.precondition(gpu.action(cp.asarray(mean_zero)))),
                mean_zero,
                atol=2e-13,
            )


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [(37, 9, 35), (16, 17, 64), (5, 4, 6)])
def test_cuda_interior_plane_window_on_partial_blocks(shape):
    """The interior stencil marches blocks of planes along x: depths, y and z
    tiles that do not divide the grid and periodic halos match the CPU action."""
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    x, y, z = np.indices(shape)
    index = ((x // 7 + y // 5 + z // 6) % 3).astype(
        np.int32
    )  # mostly interior nodes, interfaces in all axes
    c = material(phases=3)
    u = np.random.default_rng(7).normal(size=(3, *shape))
    reference = Hex8Operator(index, c, backend="numpy").action(u)
    for working, rtol in [("float64", 1e-12), ("float32", 2e-6)]:
        gpu = Hex8Operator(index, c, working=working)
        assert gpu.interior_count > index.size // 4
        out = cp.asnumpy(gpu.action(cp.asarray(u, dtype=working)))
        np.testing.assert_allclose(
            out, reference, rtol=0, atol=rtol * np.abs(reference).max()
        )


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype,working", [("float64", None), ("float32", None), ("float64", "float32")]
)
def test_cuda_fused_partials_residual_and_solve_on_multiblock_grid(dtype, working):
    """Grids spanning several 32x8 interior blocks and interface blocks."""
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear.hex8_general import _finish

    rng = np.random.default_rng(5)
    shape = (11, 19, 70)
    x, y, z = np.indices(shape)
    labels = (((x - 5) ** 2 + (y - 9) ** 2 < 20) | (z % 23 < 4)).astype(np.int32) + (
        z > 60
    )
    c = material(phases=3)
    gpu = Hex8Operator(labels, c, lengths=(0.9, 1.1, 2.0), dtype=dtype, working=working)
    cpu = Hex8Operator(labels, c, lengths=gpu.lengths, backend="numpy", dtype=dtype)
    assert 0 < gpu.interior_count < labels.size
    u = cp.asarray(rng.normal(size=(3, *shape)), dtype=gpu.working)
    out, state = cp.empty_like(u), cp.zeros(2, cp.int32)
    partial = cp.empty(gpu._action_partial_capacity(), cp.float64)
    m = gpu._action_partials(u, out, state, partial)
    tol = 1e-12 if gpu.working == np.float64 else 2e-6
    expected = cpu.action(cp.asnumpy(u).astype(np.float64), audit=True)
    np.testing.assert_allclose(
        cp.asnumpy(out), expected, rtol=tol, atol=tol * np.abs(expected).max()
    )
    direct = float(
        np.einsum(
            "i,i->",
            cp.asnumpy(u).astype(np.float64).ravel(),
            cp.asnumpy(out).astype(np.float64).ravel(),
        )
    )
    # FP64 working: products and sums in FP64. FP32 working (FAST32): each voxel's
    # three-component product is formed in FP32 and the sum is accumulated in FP64,
    # so the partials match the FP64 dot product of the FP32 vectors to rounding.
    dot_tol = 1e-12 if gpu.working == np.float64 else 1e-9
    assert abs(_finish(partial[:m], 1)[0] - direct) <= dot_tol * abs(direct)
    xs = cp.asarray(rng.normal(size=(3, *shape)), dtype=gpu.dtype)
    r = cp.empty((3, *shape), gpu.working)
    for macro in np.eye(6)[[0, 4]]:
        value = gpu._residual_into(xs, macro, r)
        fresh = cpu.rhs(macro) - cpu.action(
            cp.asnumpy(xs).astype(np.float64), audit=True
        )
        np.testing.assert_allclose(
            cp.asnumpy(r), fresh, rtol=tol, atol=tol * np.abs(fresh).max()
        )
        assert value == pytest.approx(np.abs(fresh).max(), rel=1e-12)
    result = homogenize_hex8(
        labels,
        c,
        lengths=gpu.lengths,
        dtype=dtype,
        working=working,
        rtol=1e-8 if dtype == "float64" else 1e-5,
    )
    reference = homogenize_hex8(
        labels, c, lengths=gpu.lengths, backend="numpy", rtol=1e-11
    )
    agreement = 1e-12 if dtype == "float64" else 3e-6
    np.testing.assert_allclose(
        result["C"],
        reference["C"],
        rtol=agreement,
        atol=agreement * np.abs(reference["C"]).max(),
    )


@pytest.mark.gpu
def test_load_work_stress_equals_gauss_mean_stress():
    rng = np.random.default_rng(12)
    labels = rng.integers(0, 3, (4, 5, 3))
    op = Hex8Operator(
        labels, material(phases=3), lengths=(0.8, 1.3, 1.1), backend="numpy"
    )
    u = rng.normal(size=(3, *labels.shape))
    for macro in (np.eye(6)[2], rng.normal(size=6)):
        np.testing.assert_allclose(
            op.load_work_stress(u, macro),
            op.mean_stress(u, macro),
            rtol=1e-12,
            atol=1e-13,
        )
    if cp is None:
        return
    try:
        if not cp.cuda.runtime.getDeviceCount():
            return
    except cp.cuda.runtime.CUDARuntimeError:
        return
    gpu = Hex8Operator(labels, material(phases=3), lengths=op.lengths)
    for macro in (np.eye(6)[2], rng.normal(size=6)):
        np.testing.assert_allclose(
            gpu.load_work_stress(cp.asarray(u), macro),
            op.mean_stress(u, macro),
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("working", ["float64", "float32"])
def test_cuda_energy_criterion_bounds_the_effective_energy_error(working):
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    shape = (12, 10, 14)
    x, y, z = np.indices(shape)
    labels = ((x - 6) ** 2 + (y - 5) ** 2 + (z - 7) ** 2 < 16).astype(np.int32)
    c = material(phases=2)
    c[1] *= 30
    exact = homogenize_hex8(labels, c, rtol=1e-12)["C"]
    for tau in (1e-4, 1e-7):
        result = homogenize_hex8(labels, c, working=working, energy_rtol=tau)
        error = result["C"] - exact
        assert (
            np.linalg.eigvalsh(error)[0] > -1e-13 * np.abs(exact).max()
        )  # C~ - C* is PSD
        for a, info in enumerate(result["infos"]):
            assert info["criterion"] == "energy"
            assert (
                error[a, a]
                <= info["energy_error_upper"] * (1 + 1e-6) + 1e-14 * exact[a, a]
            )
            assert error[a, a] <= tau * exact[a, a]
    loose = homogenize_hex8(labels, c, working=working, energy_rtol=1e-4)
    strict = homogenize_hex8(labels, c, working=working, rtol=1e-6)
    assert sum(i["iterations"] for i in loose["infos"]) < sum(
        i["iterations"] for i in strict["infos"]
    )


@pytest.mark.gpu
def test_cuda_fp32_audit_drift_restarts_instead_of_keeping_the_direction():
    """A failed FP32 audit whose recursion drifted far must restart (p = z):
    keeping p there stalls CG near the threshold (regression at 192^3)."""
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 6, 6))
    c = a.transpose(0, 2, 1) @ a + 2 * np.eye(6)
    c[1] *= 200
    x, y, z = np.indices((40, 36, 44))
    labels = (((x - 20) ** 2 + (y - 18) ** 2 + (z - 22) ** 2) < 150).astype(np.int32)
    result = homogenize_hex8(labels, c, dtype="float32", rtol=1e-5)
    events = [kind for info in result["infos"] for _, kind in info["events"]]
    assert "restart" in events
    assert max(info["iterations"] for info in result["infos"]) < 200
    assert all(info["residual_inf"] <= info["threshold"] for info in result["infos"])


@pytest.mark.gpu
@pytest.mark.parametrize("working", ["float64", "float32"])
def test_cuda_fused_preconditioner_matches_cufft_path(working, monkeypatch):
    """Generated-FFT PCG (hex8_fused) against the cuFFT PCG on supported grids."""
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    from cufhom.fft.linear import hex8_fused
    from cufhom.fft.linear.hex8_general import _CudaPCG

    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    rng = np.random.default_rng(9)
    c = material(phases=3)
    c[2] *= 20
    tol = 1e-13 if working == "float64" else 2e-6
    for shape in [(24, 20, 32), (12, 18, 20), (26, 14, 22)]:
        x, y, z = np.indices(shape)
        labels = ((x * 7 + y * 3 + z) % 9 < 3).astype(np.int32) + (
            (x - shape[0] // 2) ** 2 + (y - 5) ** 2 < 16
        )
        op = Hex8Operator(labels, c, lengths=(1.1, 0.9, 1.3), working=working)
        fused, library = hex8_fused.fused_pcg(op), _CudaPCG(op)
        assert fused is not None
        r = cp.asarray(rng.normal(size=(3, *shape)), dtype=op.working)
        r -= r.mean(axis=(1, 2, 3), keepdims=True)
        for pcg in (fused, library):
            pcg.reset_history(4)
            cp.copyto(pcg.r, r)
        rho = [pcg.prepare(0, 0.0, 1) for pcg in (fused, library)]
        assert abs(rho[0] / rho[1] - 1) <= tol * 10
        np.testing.assert_allclose(
            cp.asnumpy(fused.z),
            cp.asnumpy(library.z),
            rtol=tol * 10,
            atol=tol * 10 * float(cp.abs(library.z).max()),
        )
        results = [
            homogenize_hex8(labels, c, lengths=op.lengths, working=working, rtol=1e-8)
        ]
        monkeypatch.setattr(hex8_fused, "fused_pcg", lambda op: None)
        results.append(
            homogenize_hex8(labels, c, lengths=op.lengths, working=working, rtol=1e-8)
        )
        monkeypatch.undo()
        monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
        a, b = results
        np.testing.assert_allclose(
            a["C"], b["C"], rtol=1e-12, atol=1e-12 * np.abs(b["C"]).max()
        )
        for i, j in zip(
            a["infos"], b["infos"]
        ):  # FP32 rounding of two FFTs over 4-decade segments
            assert abs(i["iterations"] - j["iterations"]) <= max(
                1, 0.03 * j["iterations"]
            )


@pytest.mark.gpu
def test_cuda_double_float_residual_matches_fp64():
    """The df64 fresh residual (interface gather and 2.5-D stencil) against
    FP64, and a six-load solve accepted with either arithmetic."""
    if cp is None:
        pytest.skip("CuPy unavailable")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    shape = (37, 9, 35)
    x, y, z = np.indices(shape)
    index = ((x // 7 + y // 5 + z // 6) % 3).astype(np.int32)
    c = material(phases=3)
    rng = np.random.default_rng(11)
    u, macro = cp.asarray(rng.normal(size=(3, *shape))), rng.normal(size=6)
    norms, residuals = {}, {}
    for evaluation in ("float64", "df64"):
        op = Hex8Operator(index, c, dtype="float64", evaluation=evaluation)
        residuals[evaluation] = cp.empty((3, *shape), cp.float64)
        norms[evaluation] = op._residual_into(u, macro, residuals[evaluation])
    assert abs(norms["df64"] / norms["float64"] - 1) < 1e-13
    scale = float(cp.abs(residuals["float64"]).max())
    assert float(cp.abs(residuals["df64"] - residuals["float64"]).max()) < 1e-13 * scale
    results = {
        e: homogenize_hex8(index, c, rtol=1e-8, evaluation=e)
        for e in ("float64", "df64")
    }
    np.testing.assert_allclose(
        results["df64"]["C"],
        results["float64"]["C"],
        rtol=1e-12,
        atol=1e-12 * np.abs(results["float64"]["C"]).max(),
    )
    with pytest.raises(ValueError, match="evaluation"):
        Hex8Operator(index, c, evaluation="float16")
