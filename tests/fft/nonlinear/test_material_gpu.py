"""CUDA return maps: cancellation, apex invariants and nonsymmetric tangents."""

import numpy as np
import pytest
from cufhom.materials import drucker_prager as dp
from cufhom.materials import plasticity as j2

pytestmark = pytest.mark.gpu


@pytest.fixture
def cuda():
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("CUDA unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    return cp


def response(cp, law, material, e):
    ids = np.zeros(e.shape[1:], np.int32)
    table = np.eye(6, dtype=e.dtype)[None]
    state = j2.initial_state(ids.shape, dtype=e.dtype)
    device_state = j2.PlasticState(
        cp.asarray(state.plastic_strain), cp.asarray(state.equivalent)
    )
    implementation = j2.CuPyMaterial if law is j2 else dp.CuPyDruckerPrager
    gpu = implementation(cp, material, cp.asarray(ids), cp.asarray(table), e.dtype)
    cpu = law.update_cpu(e, state, material, ids, table)
    actual = gpu.update(cp.asarray(e), device_state)
    return gpu, cpu, actual, device_state


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1e-6, 1e-12, 1e-18])
@pytest.mark.parametrize("law", [j2, dp])
def test_cuda_tiny_positive_radius(cuda, dtype, scale, law):
    m = (j2.J2Material if law is j2 else dp.DruckerPragerMaterial)(
        yield_stress=scale, hardening=scale, **({} if law is j2 else {"friction": 0.0})
    )
    e = np.zeros((6, 2, 2, 3), dtype=dtype)
    e[3] = 0.1
    _, cpu, actual, _ = response(cuda, law, m, e)
    stress = cuda.asnumpy(actual.stress)
    q = np.sqrt(1.5 * np.sum(stress.astype(float) ** 2, axis=0))
    delta = cuda.asnumpy(actual.dp)
    assert np.all(delta > 0)
    np.testing.assert_allclose(
        q, scale * (1 + delta.astype(float)), rtol=4e-7, atol=0.0
    )
    np.testing.assert_allclose(stress, cpu.stress, rtol=4e-7, atol=0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nu", [0.3, 0.499999, 0.499999999999])
def test_cuda_apex_pressure_without_cancellation(cuda, dtype, nu):
    m = dp.DruckerPragerMaterial(nu=nu, dilatancy=0.1)
    e = np.zeros((6, 2, 2, 3), dtype=dtype)
    e[:3] = 0.01
    _, cpu, actual, _ = response(cuda, dp, m, e)
    stress = cuda.asnumpy(actual.stress)
    delta = cuda.asnumpy(actual.dp)
    np.testing.assert_array_equal(stress[0], stress[1])
    np.testing.assert_array_equal(stress[1], stress[2])
    np.testing.assert_array_equal(stress[3:], 0.0)
    np.testing.assert_allclose(
        m.friction * stress[0].astype(float),
        m.yield_stress + m.hardening * delta.astype(float),
        rtol=4e-7,
        atol=0.0,
    )
    np.testing.assert_allclose(stress, cpu.stress, rtol=4e-7, atol=0.0)


@pytest.mark.parametrize("dilatancy", [None, 0.1, 0.0])
def test_cuda_smooth_dp_tangent_and_trial_state(cuda, dilatancy):
    m = dp.DruckerPragerMaterial(dilatancy=dilatancy)
    e = np.broadcast_to(
        np.array([0.018, -0.006, -0.006, 0.003, 0.001, 0.004])[:, None, None, None],
        (6, 2, 2, 3),
    ).copy()
    gpu, cpu, actual, state = response(cuda, dp, m, e)
    tangent = []
    for c in range(6):
        v = np.zeros_like(e)
        v[c] = 1.0
        applied = cuda.asnumpy(gpu.tangent(cuda.asarray(v), actual))
        h = 1e-7
        plus = cuda.asnumpy(gpu.update(cuda.asarray(e + h * v), state).stress)
        minus = cuda.asnumpy(gpu.update(cuda.asarray(e - h * v), state).stress)
        np.testing.assert_allclose(
            applied, (plus - minus) / (2 * h), rtol=3e-7, atol=3e-9
        )
        tangent.append(applied[:, 0, 0, 0])
    tangent = np.stack(tangent, axis=1)
    if dilatancy is None:
        np.testing.assert_allclose(tangent, tangent.T, atol=3e-15, rtol=0.0)
    else:
        assert np.max(np.abs(tangent - tangent.T)) > 1e-3
    np.testing.assert_allclose(
        cuda.asnumpy(actual.stress), cpu.stress, rtol=3e-15, atol=3e-17
    )
    np.testing.assert_array_equal(cuda.asnumpy(state.plastic_strain), 0.0)
    np.testing.assert_array_equal(cuda.asnumpy(state.equivalent), 0.0)
