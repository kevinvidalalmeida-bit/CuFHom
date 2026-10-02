"""Independent CPU64 operator references for the isolated identity pilot."""
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom_core.tensors import fft
from reference_split import action, prepare


@pytest.mark.parametrize('shape', [(8,8,8), (9,7,5), (10,8,12)])
@pytest.mark.parametrize('smooth', [False, True])
@pytest.mark.parametrize('direction', [False, True])
def test_split_against_full_CPU_operator(shape, smooth, direction):
    rng = np.random.default_rng(708)
    previous = fft.get_fft_backend_status()
    try:
        Q = SymmetricCoordinates(MixedProjector(shape,(2.,3.,4.),np.ones(6,bool),cp),3.,2.)
        cpu = SymmetricCoordinates(MixedProjector(shape,(2.,3.,4.),np.ones(6,bool),np),3.,2.)
        C0 = np.zeros((6,6)); C0[:3,:3] = 3.-4./3
        np.fill_diagonal(C0[:3,:3],3.+8./3); C0[3:,3:] = 4.*np.eye(3)
        a = rng.normal(size=(6,6)); extra = a.T@a
        # The contrast is indefinite for the soft phase, while C is SPD.
        matrices = np.asarray([C0,.2*C0+.01*extra,C0+3*extra,2*C0+extra])
        assert np.linalg.eigvalsh(matrices).min()>0
        assert np.linalg.eigvalsh(matrices[1]-C0).max()<0
        ids = rng.integers(0,len(matrices),shape,dtype=np.int32)
        indices = np.triu_indices(6)
        descriptor = SimpleNamespace(val=cp.asarray(ids),material_table=
            cp.ascontiguousarray(cp.asarray(matrices[:,indices[0],indices[1]],cp.float32)))
        delta = prepare(Q,descriptor)
        fft.set_fft_backend('scipy')
        p64 = cpu.adjoint(rng.normal(size=(6,)+shape))
        r64 = cpu.adjoint(rng.normal(size=(6,)+shape))
        beta = np.float32(.37)
        target = r64+beta*p64 if direction else p64
        eps64 = cpu.strain(target)
        sig64 = np.einsum('...ij,j...->i...',matrices[ids],eps64)
        hp64 = cpu.adjoint(sig64)
        pap64 = np.vdot(eps64,sig64).real
        np.testing.assert_allclose(np.vdot(target,hp64).real,pap64,rtol=3e-14)
        np.testing.assert_allclose(cpu.adjoint(np.einsum('ij,j...->i...',C0,eps64)),target,atol=3e-14)
        fft.set_fft_backend('cupy'); fft.set_cupy_plan_mode('manual'); fft.set_cupy_rfft_batch_mode('full')
        p = cp.asarray(p64,cp.complex64); r = cp.asarray(r64,cp.complex64)
        step,pap = action(Q,descriptor,delta,p,(r,beta) if direction else None)
        np.testing.assert_allclose(cp.asnumpy(p),target,atol=1e-6,rtol=2e-6)
        np.testing.assert_allclose(pap,pap64,rtol=3e-6)
        x = cp.asarray(r64,cp.complex64); old_x = x.copy(); old_r = r.copy()
        y = 2*x; old_y = y.copy(); alpha,blend = np.float32(.017),np.float32(.35)
        rr = step.update(p,alpha,x,r,(y,blend) if smooth else None)
        np.testing.assert_allclose(cp.asnumpy(x),cp.asnumpy(old_x)+alpha*target,atol=1e-6,rtol=3e-6)
        np.testing.assert_allclose(cp.asnumpy(r),cp.asnumpy(old_r)-alpha*hp64,atol=1e-5,rtol=3e-5)
        np.testing.assert_allclose(rr,np.vdot(cp.asnumpy(r),cp.asnumpy(r)).real,rtol=3e-7)
        assert step.spectrum is None
        if smooth:
            np.testing.assert_allclose(cp.asnumpy(y),cp.asnumpy(old_y)+blend*cp.asnumpy(old_x-old_y),atol=1e-6)
    finally:
        fft.set_fft_backend(previous['requested']); fft.set_cupy_plan_mode(previous['cupy_plan_mode'])
        fft.set_cupy_rfft_batch_mode(previous['cupy_rfft_batch_mode'])


def test_mismatched_reference_is_rejected():
    Q = SymmetricCoordinates(MixedProjector((5,5,5),(1.,1.,1.),np.ones(6,bool),cp),3.,2.)
    descriptor = SimpleNamespace(material_table=cp.ones((2,21),cp.float32))
    with pytest.raises(ValueError,match='isotropic reference'):
        prepare(Q,descriptor)
