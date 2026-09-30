"""Independent dense/CPU oracles for the benchmark-only spatial PCG."""
from pathlib import Path
import sys
import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from spatial_preconditioner_candidate import BlockCompliance, SpatialPCG, cp, device_dot
from nonlinear_lowlevel_candidates import action_context
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.nonlinear_reference import IsotropicReference
from pipeline.nonlinear_solver import solve_nonlinear_homogenization, MixedProjector
from ffthompy.tensors import fft


@pytest.fixture(autouse=True)
def gpu_available():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('GPU unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('GPU unavailable')
    old = fft.get_fft_backend_status()
    fft.set_fft_backend('cupy')
    fft.set_cupy_plan_mode('manual', preserve_cache=True)
    try:
        yield
    finally:
        fft.set_fft_backend(old['requested'])
        fft.set_cupy_plan_mode(old['cupy_plan_mode'], preserve_cache=True)
        fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode'])


def sample():
    shape = (5, 6, 7)
    rng = np.random.default_rng(735)
    law = J2Material()
    ids = cp.asarray(rng.integers(0, 2, size=shape), dtype=cp.int32)
    table = np.zeros((2, 6, 6)); table[0] = np.eye(6)*2*law.mu
    table[0, :3, :3] += law.K-2*law.mu/3
    a = rng.normal(size=(6, 6)); table[1] = a@a.T+np.eye(6)
    gpu = CuPyMaterial(cp, law, ids, cp.asarray(table, dtype=cp.float32), np.float32)
    strain = cp.asarray(.004*rng.normal(size=(6,)+shape), dtype=cp.float32)
    strain[0] += .006
    state = initial_state(shape, cp, np.float32)
    response = gpu.update(strain, state)
    Q = DisplacementCoordinates(MixedProjector(shape, (1., 2., 3.),
        [True, False, False, False, False, False], cp))
    return rng, gpu, response, Q, state


@pytest.mark.parametrize('edge', [2, 4])
@pytest.mark.parametrize('support', ['all', 'matrix', 'plastic'])
@pytest.mark.parametrize('averaging', ['voigt', 'reuss'])
@pytest.mark.parametrize('shift', [0., .7])
def test_blocks_restriction_adjoint_spd_and_immutability(edge, support, averaging, shift):
    rng, gpu, response, Q, state = sample()
    saved = [x.copy() for x in (response.a, response.b, response.direction,
        response.dp, response.stress, state.plastic_strain, state.equivalent)]
    local = BlockCompliance(Q, gpu, response, edge=edge, support=support,
        averaging=averaging, shift=shift, weight=.25)
    # Tiny assembled tangent is an oracle, never a production representation.
    columns = []
    for i in range(6):
        v = cp.zeros_like(response.stress); v[i] = 1
        columns.append(cp.asnumpy(gpu.tangent(v, response)))
    C = np.stack(columns, axis=1).transpose(2, 3, 4, 0, 1).astype(float)
    C += shift*np.eye(6)
    ids, dp = cp.asnumpy(gpu.ids), cp.asnumpy(response.dp)
    mask = np.ones(Q.shape, bool) if support == 'all' else ids == 0
    if support == 'plastic':
        mask &= dp > 0
    field = rng.normal(size=(6,)+Q.shape).astype(np.float32)
    expected = np.zeros_like(field); blocks = []
    for block in np.ndindex(local.grid):
        sl = tuple(slice(i*edge, min((i+1)*edge, n)) for i, n in zip(block, Q.shape))
        selected = mask[sl]
        if selected.any():
            matrices = C[sl][selected]
            D = np.linalg.inv(matrices.mean(axis=0)) if averaging == 'voigt' else np.linalg.inv(matrices).mean(axis=0)
            mean = field[(slice(None),)+sl][:, selected].mean(axis=1, dtype=np.float64)
            values = expected[(slice(None),)+sl]
            values[:, selected] = (D@mean)[:, None]
        else:
            D = np.eye(6)
        blocks.append(D)
    np.testing.assert_allclose(cp.asnumpy(local.blocks), blocks, rtol=8e-5, atol=5e-5)
    np.testing.assert_allclose(cp.asnumpy(local.local_action(cp.asarray(field))), expected, rtol=2e-4, atol=8e-5)
    assert np.linalg.eigvalsh(cp.asnumpy(local.blocks).astype(float)).min() > 0
    x = Q.adjoint(cp.asarray(field))
    y = Q.adjoint(cp.asarray(rng.normal(size=field.shape), dtype=cp.float32))
    bx, by = local.correction(x), local.correction(y)
    energy = float(device_dot(x, bx)); assert energy >= 0
    xy, yx = float(device_dot(x, by)), float(device_dot(bx, y))
    scale = np.sqrt(float(device_dot(x, bx))*float(device_dot(y, by)))
    assert abs(xy-yx) <= 2e-6*max(scale, 1.)
    np.testing.assert_allclose(cp.asnumpy(Q.adjoint(Q.strain(bx))), cp.asnumpy(bx), rtol=2e-5, atol=1e-5)
    assert float(cp.max(cp.abs(Q.strain(bx)[0].mean()))) < 1e-6
    for actual, before in zip((response.a, response.b, response.direction,
            response.dp, response.stress, state.plastic_strain, state.equivalent), saved):
        cp.testing.assert_array_equal(actual, before)


def test_empty_plastic_support_is_zero_and_invalid_arguments():
    _, gpu, response, Q, _ = sample()
    response.dp.fill(0)
    local = BlockCompliance(Q, gpu, response, support='plastic')
    cp.testing.assert_array_equal(local.correction(Q.adjoint(response.stress)), 0)
    for kwargs in ({'weight': -1}, {'edge': 0}, {'averaging': 'bad'}, {'support': 'bad'}, {'shift': -1}):
        with pytest.raises(ValueError):
            BlockCompliance(Q, gpu, response, **kwargs)


@pytest.mark.parametrize('support,averaging', [('all','voigt'), ('matrix','voigt'), ('plastic','reuss')])
@pytest.mark.parametrize('globalization', ['newton', 'ptc'])
@pytest.mark.parametrize('reduction', ['generic', 'warp'])
def test_real_curve_free_means_and_plastic_state(support, averaging, globalization, reduction):
    phase = np.zeros((6, 6, 6), dtype=np.uint8); phase[2:4, 2:4, :] = 1
    ori = np.zeros(phase.shape+(3,)); ori[..., 2] = 1
    p = dict(phase_array=phase, ori_array=ori, path=[0., .006, .002],
        initial_increment=.002, backend='cupy', dtype='float32', rtol=1e-5,
        coordinate_pcg='fused', return_fields=True, globalization=globalization)
    baseline = solve_nonlinear_homogenization(p)
    original = FusedCoordinatePCG.solve
    def solve(engine, action, rhs, dot, tolerance, maxiter, reference):
        fields, gpu = action_context(action)
        local = BlockCompliance(engine.Q, gpu, fields['response'], edge=2,
            support=support, averaging=averaging, weight=.1, shift=fields['shift'])
        pcg = SpatialPCG(engine.Q, local, reduction=reduction)
        with pcg.installed():
            return original(pcg, action, rhs, dot, tolerance, maxiter, reference)
    with patch.object(FusedCoordinatePCG, 'solve', solve):
        candidate = solve_nonlinear_homogenization(p)
    assert FusedCoordinatePCG.solve is original
    assert not candidate['rejected_steps']
    assert max(r['residual'] for r in candidate['history']) <= 1e-5
    assert max(r['macro_residual'] for r in candidate['history']) <= 1e-5
    np.testing.assert_allclose([r['stress'] for r in candidate['history']],
        [r['stress'] for r in baseline['history']], rtol=2e-4, atol=2e-7)
    np.testing.assert_allclose(candidate['fields']['strain'], baseline['fields']['strain'], rtol=3e-4, atol=4e-7)
    np.testing.assert_allclose(candidate['state'].plastic_strain, baseline['state'].plastic_strain, rtol=4e-4, atol=4e-7)
    np.testing.assert_allclose(candidate['state'].equivalent, baseline['state'].equivalent, rtol=4e-4, atol=4e-7)


def test_exception_restores_projector():
    _, gpu, response, Q, _ = sample()
    original = Q.precondition
    local = BlockCompliance(Q, gpu, response)
    with pytest.raises(RuntimeError, match='intentional'):
        with SpatialPCG(Q, local).installed():
            raise RuntimeError('intentional')
    assert Q.precondition == original
