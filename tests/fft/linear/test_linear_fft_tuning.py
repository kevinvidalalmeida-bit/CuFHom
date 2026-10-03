"""Real full-H tuning, persistent reuse, override and failed-cache contracts."""
import json
import numpy as np
import pytest
import cufhom.fft.linear.tuning as tuning
from cufhom.fft.linear.krylov import SymmetricCoordinates
from cufhom.fft.nonlinear.solver import MixedProjector, cp
from cufhom_core.tensors import fft


@pytest.fixture
def gpu_operator(monkeypatch,tmp_path):
    if cp is None:pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA unavailable')
    old=fft.get_fft_backend_status()
    monkeypatch.setenv('CUFHOM_FFT_TUNING_CACHE',str(tmp_path/'choice.json'))
    tuning._CHOICES.clear()
    fft.set_fft_backend('cupy');fft.set_cupy_plan_mode('manual');fft.set_cupy_rfft_batch_mode('auto')
    Q=SymmetricCoordinates(MixedProjector((8,10,12),(1.,2.,3.),np.ones(6,bool),cp),3.,2.)
    rng=np.random.default_rng(129)
    field=cp.asarray(rng.normal(size=(6,)+Q.shape),cp.float32)
    stiffness=cp.asarray(rng.uniform(.2,30,Q.shape),cp.float32)
    probe=Q.adjoint(field);count=[0]
    def action(p):
        count[0]+=1
        return Q.adjoint(stiffness*Q.strain(p))
    yield action,probe,Q.shape,count,tmp_path/'choice.json'
    fft.set_fft_backend(old['requested']);fft.set_cupy_plan_mode(old['cupy_plan_mode'])
    fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode']);tuning._CHOICES.clear()


def test_real_operator_persistent_reuse_and_explicit_override(gpu_operator,monkeypatch):
    action,probe,shape,count,path=gpu_operator;saved=probe.copy()
    first=tuning.tune(action,probe,shape)
    assert first['status']=='tuned' and first['persisted'] and count[0]==25
    cp.testing.assert_array_equal(probe,saved)
    assert all(x['relative_error']<2e-6 for x in first['candidates'].values())
    record=json.loads(path.read_text())['choices'][first['cache_key']]
    assert record['configuration']['cufft']>0 and record['tuning_s']>0
    fft.set_cupy_rfft_batch_mode('auto');count[0]=0
    assert tuning.tune(action,probe,shape)['status']=='memory_cache' and count[0]==0
    tuning._CHOICES.clear();fft.set_cupy_rfft_batch_mode('auto')
    second=tuning.tune(action,probe,shape)
    assert second['status']=='disk_cache' and count[0]==0 and second['policy']==first['policy']
    fft.set_cupy_rfft_batch_mode('batch3')
    assert tuning.tune(action,probe,shape)['status']=='explicit_policy' and count[0]==0
    original=tuning.configuration
    def changed(*args):
        config=original(*args);config['driver']+=1;return config
    monkeypatch.setattr(tuning,'configuration',changed)
    fft.set_cupy_rfft_batch_mode('auto')
    third=tuning.tune(action,probe,shape)
    assert third['status']=='tuned' and third['cache_key']!=first['cache_key'] and count[0]==25


def test_failed_write_and_broken_cache_do_not_break_solve(gpu_operator,monkeypatch):
    action,probe,shape,count,path=gpu_operator
    path.write_text('{"schema":1,"choices":null}')
    assert tuning.read_cache(path)['choices']=={}
    parent=path.parent/'regular_file';parent.write_text('occupied')
    monkeypatch.setenv('CUFHOM_FFT_TUNING_CACHE',str(parent/'cache.json'))
    result=tuning.tune(action,probe,shape)
    assert result['status']=='tuned' and not result['persisted']
    fft.set_cupy_rfft_batch_mode('auto');count[0]=0
    assert tuning.tune(action,probe,shape)['status']=='memory_cache' and count[0]==0


def test_invalid_candidate_rejected_and_exception_restores_policy(gpu_operator):
    action,probe,shape,count,path=gpu_operator
    def candidate(p):
        value=action(p)
        return 1.01*value if fft.CUPY_RFFT_BATCH_MODE=='batch2' else value
    first=tuning.tune(candidate,probe,shape)
    assert first['candidates']['batch2']['status']=='operator_mismatch'
    assert first['policy']!='batch2'
    path.unlink();tuning._CHOICES.clear();fft.set_cupy_rfft_batch_mode('auto')
    def failed(p):raise RuntimeError('operator failure')
    with pytest.raises(RuntimeError,match='operator failure'):tuning.tune(failed,probe,shape)
    assert fft.CUPY_RFFT_BATCH_MODE=='auto'
