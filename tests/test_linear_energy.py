"""Independent material/coordinate references, including inadmissible slots."""
from types import SimpleNamespace
import numpy as np
import pytest
cp = pytest.importorskip('cupy')
from cufhom.pipeline.linear_energy import material_energy, energy_action
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom_core.tensors import fft


@pytest.fixture(autouse=True)
def require_gpu():
    try:
        if not cp.cuda.runtime.getDeviceCount():pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CUDA unavailable')


def test_default_pipeline_reports_energy_and_material_profile(tmp_path,monkeypatch):
    import json
    from contextlib import redirect_stdout
    import os
    from cufhom.pipeline.fft_solver import solve_homogenization
    monkeypatch.setenv('CUFHOM_FFT_TUNING_CACHE',str(tmp_path/'fft.json'))
    phase=np.zeros((8,8,8),np.uint8);phase[2:5,2:5,2:5]=1
    ori=np.zeros(phase.shape+(3,),np.float32);ori[...,2]=1
    p=dict(input_dir='.',phase_array=phase,ori_array=ori,fft_backend='cupy',
        Em=3.,nu_m=.3,Ef_L=90.,Ef_T=15.,nu_LT=.25,nu_TT=.3,G_LT=5.,
        solver_rtol=1e-5,solver_maxiter=1000,require_convergence=True,solver_verbose=False,
        profile_timing=True,solver_timing_path=str(tmp_path/'profile.json'))
    with open(os.devnull,'w') as sink,redirect_stdout(sink):C=solve_homogenization(p)
    detail=json.loads((tmp_path/'profile.json').read_text())
    primal=detail['cufhom_core_application_timing']['primaldual']['primal']
    infos=primal['load_solver_infos']
    assert len(infos)==6 and all(i['linear_energy_update'] for i in infos)
    assert all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
    assert infos[0]['linear_fft_tuning']['status']=='tuned'
    assert all(i['linear_fft_tuning']['status']=='same_solve' for i in infos[1:])
    assert np.linalg.eigvalsh((C+C.T)/2)[0]>0
    loads=detail['cufhom_core_solver_timing']['primal']['loads']
    assert sum(load['components']['A']['calls'] for load in loads)>=sum(i['coordinate_iterations'] for i in infos)


@pytest.mark.parametrize('shape', [(8,8,8), (9,7,5), (10,8,12)])
@pytest.mark.parametrize('smooth', [False, True])
def test_energy_and_fused_update(shape, smooth):
    rng = np.random.default_rng(758)
    old = fft.get_fft_backend_status()
    fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual'); fft.set_cupy_rfft_batch_mode('full')
    try:
        matrices = rng.normal(size=(7,6,6))
        matrices = np.einsum('mki,mkj->mij', matrices, matrices)+np.eye(6)
        ids = rng.integers(0,7,shape,dtype=np.int32)
        table = matrices[:, np.triu_indices(6)[0], np.triu_indices(6)[1]]
        descriptor = SimpleNamespace(val=cp.asarray(ids), material_table=cp.ascontiguousarray(cp.asarray(table, cp.float32)))
        Q = SymmetricCoordinates(MixedProjector(shape, (2.,3.,4.), np.ones(6,bool), cp), 3., 2.)
        cpu = SymmetricCoordinates(MixedProjector(shape, (2.,3.,4.), np.ones(6,bool), np), 3., 2.)
        source = rng.normal(size=(6,)+shape)
        fft.set_fft_backend('scipy')
        p64 = cpu.adjoint(source)
        p = cp.asarray(p64, cp.complex64)
        eps64 = cpu.strain(p64)
        sig64 = np.einsum('...ij,j...->i...', matrices[ids], eps64)
        expected_pap = np.vdot(eps64, sig64).real
        ap = cpu.adjoint(sig64)
        np.testing.assert_allclose(np.vdot(p64, ap).real, expected_pap, rtol=2e-14)
        fft.set_fft_backend('cupy')
        eps = Q.strain(p)
        stress, pap = material_energy(descriptor, eps)
        np.testing.assert_allclose(cp.asnumpy(stress), sig64, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(pap, expected_pap, rtol=8e-7)
        deferred, pap2 = energy_action(Q, descriptor, p)
        np.testing.assert_allclose(pap2, pap, rtol=1e-14)
        # All coordinate slots, including mean/Nyquist, exercise the update.
        x = cp.asarray(rng.normal(size=Q.size)+1j*rng.normal(size=Q.size), cp.complex64)
        r = x.copy(); y = 2*x; old_y = y.copy(); old_x = x.copy(); old_r = r.copy()
        alpha, blend = .07, .35
        new_rr = deferred.update(p, alpha, x, r, (y,blend) if smooth else None)
        assert deferred.spectrum is None
        np.testing.assert_allclose(cp.asnumpy(x), cp.asnumpy(old_x)+np.float32(alpha)*p64, atol=1e-6, rtol=2e-6)
        np.testing.assert_allclose(cp.asnumpy(r), cp.asnumpy(old_r)-np.float32(alpha)*ap, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(new_rr, np.vdot(cp.asnumpy(r), cp.asnumpy(r)).real, rtol=2e-7)
        if smooth:
            np.testing.assert_allclose(cp.asnumpy(y), cp.asnumpy(old_y)+np.float32(blend)*cp.asnumpy(old_x-old_y), atol=1e-6)
    finally:
        fft.set_fft_backend(old['requested']); fft.set_cupy_plan_mode(old['cupy_plan_mode'])
        fft.set_cupy_rfft_batch_mode(old['cupy_rfft_batch_mode'])
