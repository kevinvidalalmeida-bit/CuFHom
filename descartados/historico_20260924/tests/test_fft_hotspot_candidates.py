"""GPU implementation changes preserve transforms, means and tangent action."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from hotspot_candidates import cp, fft, installed, tangent_registers, coordinate_specialized
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_coordinates import DisplacementCoordinates
from pipeline.linear_krylov import SymmetricCoordinates
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    old = fft.get_fft_backend_status()
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')
    yield
    fft.set_fft_backend(old['requested']); fft.set_cupy_plan_mode(old['cupy_plan_mode'])


@pytest.mark.parametrize('shape', [(5,7,9), (6,8,10), (7,6,8)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_split_fft_roundtrip_and_input_ownership(shape, dtype):
    x = cp.asarray(np.random.default_rng(433).normal(size=(6,)+shape), dtype=dtype)
    saved = x.copy()
    ref = fft.rfftn_cached(x, shape, (1,2,3))
    tol = 2e-6 if dtype == np.float32 else 2e-14
    with installed(split_fft=True):
        h = fft.rfftn_cached(x, shape, (-3,-2,-1))
        cp.testing.assert_allclose(h, ref, atol=tol*100, rtol=tol)
        out = fft.irfftn_unscaled_cached(h, shape, (1,2,3))
        cp.testing.assert_allclose(out/np.prod(shape), saved, atol=tol, rtol=tol)
        cp.testing.assert_array_equal(x, saved)
        # Returned arrays cannot alias subsequent calls or the original input.
        cp.testing.assert_allclose(fft.irfftn_unscaled_cached(
            fft.rfftn_cached(2*x, shape, (1,2,3)), shape, (1,2,3))/np.prod(shape),
            2*saved, atol=2*tol, rtol=tol)
        cp.testing.assert_allclose(out/np.prod(shape), saved, atol=tol, rtol=tol)


@pytest.mark.parametrize('shape', [(5,7,9), (6,8,10)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('kind', ['mixed', 'free', 'root'])
def test_coordinates_all_modes(shape, dtype, kind):
    mask = [True,False,False,False,False,False] if kind=='mixed' else [kind=='root']*6
    P = MixedProjector(shape, (1.,2.,3.), mask, cp)
    Q = SymmetricCoordinates(P, 2.5, 1.15) if kind=='root' else DisplacementCoordinates(P)
    x = cp.asarray(np.random.default_rng(452).normal(size=(6,)+shape), dtype=dtype)
    h = fft.rfftn_cached(x, shape, (1,2,3))
    z = Q._map(h, True)
    for adjoint, value in [(True,h), (False,z)]:
        expected = Q._map(value, adjoint)
        actual = coordinate_specialized(Q, value, adjoint)
        eps = np.finfo(dtype).eps
        cp.testing.assert_allclose(actual, expected, atol=16*eps, rtol=16*eps)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('amplitude', [.00001, .01])
def test_tangent_and_immutable_state(dtype, amplitude):
    shape = (7,6,8)
    rng = np.random.default_rng(24)
    material = J2Material(E=3., nu=.3, yield_stress=.015, hardening=.03)
    ids = cp.asarray(rng.integers(0,3,size=shape), dtype=cp.int32)
    # Arbitrary SPD anisotropic elastic tables exercise off-diagonal terms.
    table = rng.normal(size=(3,6,6)); table = table@table.transpose(0,2,1)+np.eye(6)
    gpu = CuPyMaterial(cp, material, ids, cp.asarray(table,dtype=dtype), dtype)
    state = initial_state(shape, cp, dtype)
    saved_state = state.plastic_strain.copy(), state.equivalent.copy()
    strain = cp.asarray(amplitude*rng.normal(size=(6,)+shape), dtype=dtype)
    response = gpu.update(strain, state)
    direction = cp.asarray(rng.normal(size=strain.shape), dtype=dtype)
    expected = gpu.tangent(direction, response)
    saved_response = [v.copy() for v in vars(response).values()]
    actual = tangent_registers(gpu, direction, response)
    cp.testing.assert_array_equal(actual, expected)
    for actual, expected in zip((state.plastic_strain, state.equivalent), saved_state):
        cp.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(vars(response).values(), saved_response):
        cp.testing.assert_array_equal(actual, expected)


def test_pcg_geometry_specialization():
    from pcg_geometry_candidate import installed as specialized
    from pipeline.nonlinear_krylov import FusedCoordinatePCG
    from pipeline.nonlinear_reference import IsotropicReference
    for shape in [(5,7,9), (6,8,10)]:
        P = MixedProjector(shape, (1.,2.,3.), [True,False,False,False,False,False], cp)
        Q = DisplacementCoordinates(P)
        M = IsotropicReference(P, 2.5, 1.15); M.set_shift(3., np.dtype('float32'))
        base = FusedCoordinatePCG(Q)
        with specialized():
            candidate = FusedCoordinatePCG(Q)
        rng = np.random.default_rng(83)
        x,r,p,ap = [Q.adjoint(cp.asarray(rng.normal(size=(6,)+shape),dtype=cp.float32)) for _ in range(4)]
        xc,rc,pc = x.copy(),r.copy(),p.copy()
        a = base.update(p,ap,.13,x,r,M)
        b = candidate.update(pc,ap,.13,xc,rc,M)
        np.testing.assert_allclose(a,b,rtol=3e-7)
        cp.testing.assert_array_equal(x,xc); cp.testing.assert_array_equal(r,rc)
        base.direction(r,.15,p,M); candidate.direction(rc,.15,pc,M)
        cp.testing.assert_array_equal(p,pc)
