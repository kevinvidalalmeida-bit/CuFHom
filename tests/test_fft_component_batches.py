"""Production real-FFT batching: correct scaling, ownership and safe fallback."""
from types import SimpleNamespace
import numpy as np
import pytest

from cufhom.pipeline.fft_solver import cp
from cufhom_core.tensors import fft


@pytest.fixture(autouse=True)
def restore_settings():
    old = fft.get_fft_backend_status()
    yield
    fft.set_fft_backend(old['requested'])
    fft.set_cupy_plan_mode(old['cupy_plan_mode'])
    fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode'])


def require_gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual')


@pytest.mark.parametrize('shape', [(5,7,9),(6,8,10),(7,6,8),(6,8,9)])
@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('policy', ['component', 'batch2', 'batch3'])
def test_roundtrip_and_distinct_outputs(shape,dtype,policy):
    require_gpu()
    x = cp.asarray(np.random.default_rng(954).normal(size=(6,)+shape), dtype=dtype)
    saved = x.copy()
    fft.set_cupy_rfft_batch_mode('full')
    reference = fft.rfftn_cached(x,shape,(1,2,3))
    fft.set_cupy_rfft_batch_mode(policy)
    h = fft.rfftn_cached(x,shape,(-3,-2,-1))
    tol = 2e-6 if dtype==np.float32 else 2e-14
    cp.testing.assert_allclose(h,reference,atol=100*tol,rtol=tol)
    out = fft.irfftn_unscaled_cached(h,shape,(-3,-2,-1))
    cp.testing.assert_allclose(out/np.prod(shape),saved,atol=tol,rtol=tol)
    second = fft.irfftn_unscaled_cached(fft.rfftn_cached(2*x,shape,(1,2,3)),shape,(1,2,3))
    cp.testing.assert_allclose(second/np.prod(shape),2*saved,atol=2*tol,rtol=tol)
    cp.testing.assert_allclose(out/np.prod(shape),saved,atol=tol,rtol=tol)
    cp.testing.assert_array_equal(x,saved)


@pytest.mark.parametrize('layout', ['noncontiguous','three','odd','auto_plan'])
def test_gpu_fallbacks(layout):
    require_gpu()
    shape = (5,7,9) if layout=='odd' else (6,8,10)
    leading = 3 if layout=='three' else 6
    x = cp.asarray(np.random.default_rng(13).normal(size=(leading,)+shape),dtype=cp.float32)
    if layout=='noncontiguous':
        x = x[...,::-1]
        # Preserve the existing manual-plan contract; do not route unsupported
        # layouts through raw cuFFT or silently reinterpret their strides.
        for mode in ('full', 'component'):
            fft.set_cupy_rfft_batch_mode(mode)
            assert not fft._use_component_fft(x,shape,(1,2,3),False)
            with pytest.raises(ValueError,match='contiguous'):
                fft.rfftn_cached(x,shape,(1,2,3))
        return
    if layout=='auto_plan':
        fft.set_cupy_plan_mode('auto')
    fft.set_cupy_rfft_batch_mode('full')
    expected = fft.rfftn_cached(x,shape,(1,2,3))
    fft.set_cupy_rfft_batch_mode('component')
    actual = fft.rfftn_cached(x,shape,(1,2,3))
    cp.testing.assert_array_equal(actual,expected)


def test_auto_policy_only_measured_shape_precision_and_device(monkeypatch):
    require_gpu()
    fft.set_cupy_rfft_batch_mode('auto')
    fake = SimpleNamespace(ndim=4,shape=(6,240,240,240),dtype=np.dtype('float32'),
                           flags=SimpleNamespace(c_contiguous=True),data=SimpleNamespace(ptr=256))
    monkeypatch.setattr(fft,'_component_fft_device_supported',lambda:True)
    assert fft._use_component_fft(fake,(240,)*3,(1,2,3),False)
    monkeypatch.setattr(fft,'_component_fft_device_supported',lambda:False)
    assert not fft._use_component_fft(fake,(240,)*3,(1,2,3),False)
    monkeypatch.setattr(fft,'_component_fft_device_supported',lambda:True)
    fake.dtype=np.dtype('float64')
    assert not fft._use_component_fft(fake,(240,)*3,(1,2,3),False)
    fake.dtype=np.dtype('float32'); fake.data.ptr=260
    assert not fft._use_component_fft(fake,(240,)*3,(1,2,3),False)


def test_cpu_and_invalid_options():
    fft.set_fft_backend('scipy')
    fft.set_cupy_rfft_batch_mode('component')
    x = np.random.default_rng(573).normal(size=(6,5,7,9))
    h = fft.rfftn_cached(x,(5,7,9),(1,2,3))
    out = fft.irfftn_unscaled_cached(h,(5,7,9),(1,2,3))
    np.testing.assert_allclose(out/(5*7*9),x,atol=2e-15)
    with pytest.raises(ValueError,match='cupy_rfft_batch_mode'):
        fft.set_cupy_rfft_batch_mode('invalid')


def test_plastic_loading_unloading_preserves_fields_state_and_settings(monkeypatch):
    require_gpu()
    from cufhom.pipeline.nonlinear_solver import solve_nonlinear_homogenization
    phase = np.zeros((6,6,6),dtype=np.uint8); phase[2:4,2:4,:]=1
    ori = np.zeros(phase.shape+(3,)); ori[...,2]=1
    p = dict(phase_array=phase,ori_array=ori,backend='cupy',return_fields=True,
             path=[0.,.006,.004],initial_increment=.001)
    fft.set_cupy_rfft_batch_mode('full')
    before = fft.get_fft_backend_status()
    reference = solve_nonlinear_homogenization(dict(p,cupy_rfft_batch_mode='full'))
    original = fft._component_fft
    calls = []
    def counted(x,shape,inverse):
        calls.append(inverse)
        return original(x,shape,inverse)
    monkeypatch.setattr(fft,'_component_fft',counted)
    candidate = solve_nonlinear_homogenization(dict(p,cupy_rfft_batch_mode='component'))
    assert True in calls and False in calls
    assert fft.get_fft_backend_status() == before
    assert not candidate['rejected_steps']
    assert max(r['residual'] for r in candidate['history']) <= 1e-5
    np.testing.assert_allclose([r['stress'] for r in candidate['history']],
                               [r['stress'] for r in reference['history']],atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(candidate['fields']['strain'],reference['fields']['strain'],atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(candidate['state'].equivalent,reference['state'].equivalent,atol=1e-7,rtol=1e-5)
    np.testing.assert_allclose(candidate['state'].plastic_strain,reference['state'].plastic_strain,atol=1e-7,rtol=1e-5)
