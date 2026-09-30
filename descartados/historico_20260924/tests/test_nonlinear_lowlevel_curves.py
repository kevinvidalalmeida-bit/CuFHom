"""Full nonlinear-path checks for the isolated low-level candidates."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'FFT/benchmarks'))
from nonlinear_lowlevel_candidates import installed, tangent_reference, cp
from pipeline.nonlinear_solver import solve_nonlinear_homogenization, MixedProjector
from pipeline.nonlinear_krylov import FusedCoordinatePCG
from pipeline.nonlinear_material import CuPyMaterial, J2Material, initial_state
from pipeline.nonlinear_reference import IsotropicReference
from pipeline.nonlinear_coordinates import DisplacementCoordinates


@pytest.fixture(autouse=True)
def gpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('GPU unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('GPU unavailable')


def case():
    phase = np.zeros((6,6,6),dtype=np.uint8); phase[2:4,2:4,:]=1
    ori = np.zeros(phase.shape+(3,)); ori[...,2]=1
    return dict(phase_array=phase,ori_array=ori,path=[0.,.008,.003],
        initial_increment=.001,backend='cupy',dtype='float32',return_fields=True,
        rtol=1e-5,coordinate_pcg='fused',cupy_rfft_batch_mode='component')


@pytest.mark.parametrize('reference',['matrix','matrix_voigt','matrix_reuss','full_voigt'])
@pytest.mark.parametrize('fused_dot',[False,True])
def test_curve_fields_and_state(reference,fused_dot):
    p = case()
    original = FusedCoordinatePCG.solve
    baseline = solve_nonlinear_homogenization(p)
    with installed(reference=reference,fused_dot=fused_dot) as stats:
        candidate = solve_nonlinear_homogenization(p)
    assert FusedCoordinatePCG.solve is original
    assert not candidate['rejected_steps']
    assert max(r['residual'] for r in candidate['history'])<=1e-5
    np.testing.assert_allclose([r['stress'] for r in candidate['history']],
                              [r['stress'] for r in baseline['history']],atol=2e-7,rtol=2e-5)
    np.testing.assert_allclose(candidate['fields']['strain'],baseline['fields']['strain'],atol=3e-7,rtol=3e-4)
    np.testing.assert_allclose(candidate['state'].equivalent,baseline['state'].equivalent,atol=3e-7,rtol=3e-4)
    np.testing.assert_allclose(candidate['state'].plastic_strain,baseline['state'].plastic_strain,atol=3e-7,rtol=3e-4)
    if fused_dot:
        assert stats['fused_products']>0


def test_ptc_exact_fallback_and_restart(tmp_path):
    p = dict(case(),globalization='ptc')
    baseline = solve_nonlinear_homogenization(p)
    with installed(fused_dot=True) as stats:
        candidate = solve_nonlinear_homogenization(p)
    assert stats['shift_fallbacks']>0 and stats['fused_products']==0
    np.testing.assert_array_equal(candidate['state'].equivalent,baseline['state'].equivalent)
    checkpoint = tmp_path/'state.npz'
    with installed(fused_dot=True):
        solve_nonlinear_homogenization(dict(case(),path=[0.,.008],checkpoint_path=checkpoint))
        restarted = solve_nonlinear_homogenization(dict(case(),path=[.008,.003],restart_from=checkpoint))
        uninterrupted = solve_nonlinear_homogenization(case())
    np.testing.assert_allclose(restarted['state'].equivalent,uninterrupted['state'].equivalent,atol=3e-7)


@pytest.mark.parametrize('mode',['matrix_voigt','matrix_reuss','full_voigt'])
@pytest.mark.parametrize('amplitude',[1e-6,.01])
def test_reference_invariants_against_explicit_material_tangent(mode,amplitude):
    shape = (3,4,5); law = J2Material()
    rng = np.random.default_rng(872)
    ids = cp.asarray(rng.integers(0,2,size=shape),dtype=cp.int32)
    table = np.zeros((2,6,6)); table[0]=np.eye(6)*2*law.mu
    table[0,:3,:3]+=law.K-2*law.mu/3
    R = rng.normal(size=(6,6)); table[1]=R@R.T+np.eye(6)
    gpu = CuPyMaterial(cp,law,ids,cp.asarray(table,dtype=cp.float32),np.float32)
    strain = cp.asarray(amplitude*rng.normal(size=(6,)+shape),dtype=cp.float32)
    response = gpu.update(strain,initial_state(shape,cp,np.float32))
    Q = DisplacementCoordinates(MixedProjector(shape,(1.,2.,3.),[True,False,False,False,False,False],cp))
    baseline = IsotropicReference(Q.P,law.K,law.mu); baseline.set_shift(0.,np.dtype('float32'))
    selected, row = tangent_reference(Q,gpu,response,baseline,3.,mode,{})
    # Assemble only this tiny verification sample, never a production field.
    columns=[]
    for c in range(6):
        direction=cp.zeros_like(strain);direction[c]=1
        columns.append(cp.asnumpy(gpu.tangent(direction,response)).reshape(6,-1))
    matrices=np.stack(columns,axis=1).transpose(2,0,1).astype(np.float64)
    matrix_mask=cp.asnumpy(ids).ravel()==0
    trace_bulk=np.sum(matrices[:,:3,:3],axis=(1,2))/9
    trace_dev=np.trace(matrices,axis1=1,axis2=2)-3*trace_bulk
    if mode=='full_voigt':
        K=trace_bulk.mean();mu=trace_dev.mean()/10
    elif mode=='matrix_voigt':
        K=law.K;mu=trace_dev[matrix_mask].mean()/10
    else:
        compliance=np.linalg.inv(matrices[matrix_mask])
        J=np.zeros((6,6));J[:3,:3]=1/3
        compliance_dev=np.einsum('ij,nji->n',np.eye(6)-J,compliance).mean()
        K=law.K;mu=5/(2*compliance_dev)
    np.testing.assert_allclose([selected.K,selected.mu],[K,mu],rtol=3e-5)
    assert min(np.linalg.eigvalsh(cp.asnumpy(selected.mean)[1:,1:]))>0
    assert selected.shift==3.
    assert baseline.K==law.K and baseline.mu==law.mu and baseline.shift==0.


def test_restore_after_exception():
    original=FusedCoordinatePCG.solve
    with pytest.raises(RuntimeError,match='intentional'):
        with installed(fused_dot=True):
            raise RuntimeError('intentional')
    assert FusedCoordinatePCG.solve is original
