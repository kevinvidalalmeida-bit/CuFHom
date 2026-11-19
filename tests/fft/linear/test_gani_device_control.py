"""Bounded device control preserves the residual-audited GaNi algorithm."""

import numpy as np
import pytest
from cufhom.fft import homogenize
from cufhom.fft.linear import coordinates, fused
from cufhom.fft.linear.common import LinearSolveError
from cufhom.fft.linear.energy import energy_update_device
from cufhom.fft.linear.krylov import fused_cg
from cufhom_core.tensors import fft

from tests.fft.linear.test_linear_fused_fft import _problem
from tests.fft.linear.test_linear_fusions import real_dot

cp = pytest.importorskip("cupy")

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("CUFHOM_FUSED_COMPILE", "sync")
    fft.set_fft_backend("cupy")
    fft.set_cupy_plan_mode("manual", preserve_cache=True)


def _solve(shape, smoothing, device, maxiter=300, invalid=False, reject_first=False):
    Q, A, rng = _problem(shape, seed=1903)
    pipeline = fused.FusedPipeline(Q, A)
    rhs = pipeline.to_fused(
        Q.adjoint(cp.asarray(rng.standard_normal((6,) + shape).astype(np.float32)))
    )
    if invalid:
        pipeline.table *= -1
    stats, inspections = {}, []

    def accept(x):
        residual = rhs - pipeline.matvec(x)
        inspections.append(np.sqrt(real_dot(residual, residual)))
        return 1e-4 if reject_first and len(inspections) == 1 else None

    x, iterations = fused_cg(
        pipeline.matvec,
        rhs,
        real_dot,
        2e-5,
        maxiter,
        action_and_dot=lambda p, direction=None: (
            fused.FusedStep(pipeline),
            pipeline.apply(p, direction),
        ),
        device_update_callback=energy_update_device,
        settle=pipeline.settle,
        residual_smoothing=smoothing,
        accept=accept,
        stats=stats,
        device_prefix=pipeline.prefix if device else None,
    )
    return cp.asnumpy(x), iterations, stats, inspections


@pytest.mark.parametrize("shape", [(8, 8, 8), (12, 10, 14)])
@pytest.mark.parametrize("smoothing", [False, True])
def test_prefix_preserves_host_iterates_and_fresh_acceptance(shape, smoothing):
    before = _solve(shape, smoothing, False)
    after = _solve(shape, smoothing, True)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]
    np.testing.assert_array_equal(after[3], before[3])
    assert after[2]["device_control_iterations"] > 0
    assert after[2]["control_queries"] < before[2]["control_queries"]


def test_rejected_fresh_candidate_still_restarts_and_tightens():
    before = _solve((8, 8, 8), True, False, reject_first=True)
    after = _solve((8, 8, 8), True, True, reject_first=True)
    np.testing.assert_array_equal(after[0], before[0])
    assert after[1] == before[1]
    assert len(after[3]) >= 2 and after[2]["restarts"] >= 1


@pytest.mark.parametrize("device", [False, True])
def test_prefix_breakdown_stops_at_first_invalid_step(device):
    with pytest.raises(LinearSolveError) as caught:
        _solve((8, 8, 8), True, device, invalid=True)
    assert caught.value.status == "nonfinite"
    assert caught.value.iterations == 1


@pytest.mark.parametrize("device", [False, True])
def test_prefix_respects_iteration_budget(device):
    with pytest.raises(LinearSolveError) as caught:
        _solve((8, 8, 8), True, device, maxiter=1)
    assert caught.value.status == "maxiter"
    assert caught.value.iterations == 1


@pytest.mark.parametrize(
    "event", ["drop", "growth", "smoothing", "candidate", "block_cap", "budget"]
)
def test_prefix_stops_on_first_host_control_boundary(event):
    Q, A, rng = _problem((8, 10, 12), seed=1701)
    host, device = fused.FusedPipeline(Q, A), fused.FusedPipeline(Q, A)
    rhs = host.to_fused(
        Q.adjoint(cp.asarray(rng.standard_normal((6, 8, 10, 12)).astype(np.float32)))
    )
    rr = real_dot(rhs, rhs)
    ph, rh, xh = rhs.copy(), rhs.copy(), cp.zeros_like(rhs)
    denominator = host.apply(ph)
    first = host.update(ph, rr, denominator, xh, rh)
    reference = max(rr, 2 * first)
    limit, floor, drop, growth, smoothing, remaining = 0.0, rr, 0.0, None, False, 8
    if event == "drop":
        drop = np.sqrt(1.01 * first / reference)
    elif event == "growth":
        floor, growth = first * 1e-6, 100.0
    elif event == "smoothing":
        limit, smoothing = 1.01 * first / 100.0, True
    elif event == "candidate":
        limit = 1.01 * first
    elif event == "budget":
        remaining = 1
    expected = 4 if event == "block_cap" else 1
    previous, new, expected_floor = rr, first, min(floor, first)
    for _ in range(1, expected):
        denominator = host.apply(ph, (rh, new / previous))
        previous = new
        new = host.update(ph, previous, denominator, xh, rh)
        expected_floor = min(expected_floor, new)
    pd, rd, xd = rhs.copy(), rhs.copy(), cp.zeros_like(rhs)
    count, old, actual, actual_floor = device.prefix(
        pd,
        rr,
        xd,
        rd,
        None,
        limit,
        floor,
        reference,
        drop,
        growth,
        smoothing,
        remaining,
    )
    assert (count, old, actual, actual_floor) == (
        expected,
        previous,
        new,
        expected_floor,
    )
    host.settle()
    device.settle()
    for a, b in ((ph, pd), (rh, rd), (xh, xd)):
        np.testing.assert_array_equal(cp.asnumpy(a), cp.asnumpy(b))


@pytest.mark.parametrize("evaluation", ["df64", "float64"])
@pytest.mark.parametrize("rtol", [1e-6, 1e-8])
def test_mixed_six_loads_match_host_with_reliable_updates(
    monkeypatch, evaluation, rtol
):
    rng = np.random.default_rng(82026)
    labels = rng.integers(0, 2, size=(8, 10, 12))
    a = rng.normal(size=(2, 6, 6))
    table = a.transpose(0, 2, 1) @ a + 2 * np.eye(6)
    table[1] *= 10
    options = dict(
        discretization="gani",
        dtype="float64",
        working="float32",
        evaluation=evaluation,
        lengths=(0.8, 1.3, 1.1),
        rtol=rtol,
    )
    original = coordinates.fused_cg
    native_infos = []
    finalize = coordinates._finalize_cg_info

    def record(info, *args):
        finalize(info, *args)
        native_infos.append(dict(info))

    monkeypatch.setattr(coordinates, "_finalize_cg_info", record)

    def host(*args, **kwargs):
        kwargs.pop("device_prefix", None)
        return original(*args, **kwargs)

    with monkeypatch.context() as local:
        local.setattr(coordinates, "fused_cg", host)
        before = homogenize(labels, table, **options)
    after = homogenize(labels, table, **options)
    np.testing.assert_array_equal(after["C"], before["C"])
    np.testing.assert_array_equal(after["iterations"], before["iterations"])
    assert len(after["audits"]) == 6
    assert all(a["accepted"] for a in after["audits"])
    for old, new in zip(before["infos"], after["infos"]):
        for key in ("final_norm_res", "replacements", "restarts", "candidate_checks"):
            assert new[key] == old[key]
    assert len(native_infos) == 12
    for old, new in zip(native_infos[:6], native_infos[6:]):
        assert new["linear_cg_control"] == "device_prefix4"
        assert new["device_control_iterations"] > 0
        assert new["control_queries"] < old["control_queries"]
    if rtol < 1e-6:
        assert sum(i["replacements"] for i in after["infos"]) > 0


@pytest.mark.parametrize(
    "n,control",
    [(32, "device_prefix4_host_smoothing"), (64, "host_scalar_per_iteration")],
)
def test_measured_control_policy_and_public_diagnostics(n, control):
    axis = (np.arange(n) + 0.5) / n - 0.5
    labels = (
        axis[:, None, None] ** 2 + axis[None, :, None] ** 2 + axis[None, None, :] ** 2
        < 0.3**2
    ).astype(np.int32)
    c = 2 * np.eye(6)
    c[:3, :3] += 2 / 3
    result = homogenize(
        labels,
        np.stack((c, 10 * c)),
        discretization="gani",
        dtype="float64",
        working="float32",
        rtol=1e-6,
    )
    assert result["precision_contract"]["control"] == control
    assert all(a["accepted"] for a in result["audits"])
    for info in result["infos"]:
        assert info["control_queries"] > 0
        assert (info["device_control_iterations"] > 0) == (n == 32)
