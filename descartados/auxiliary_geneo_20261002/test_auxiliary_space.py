"""Archived checks of adjoints, symmetry/positivity, Q1 null modes and PCG."""
import numpy as np
import cupy as cp
import pytest
from scipy import sparse
from auxiliary_space import (prolongation, Transfer, element_matrices, assemble,
                             Multigrid, geneo_basis, project)
from metric_pcg import metric_pcg
from cufhom.pipeline.linear_krylov import SymmetricCoordinates
from cufhom.pipeline.nonlinear_solver import MixedProjector
from cufhom_core.tensors import fft


def test_q1_rigid_modes_and_spd_vcycle():
    rng=np.random.default_rng(401);shape=(8,8,8)
    C=np.eye(6)[None]*rng.uniform(.1,30,(np.prod(shape),1,1))
    elements=element_matrices(C,(2.,3.,4.),shape)
    # Full quadrature gives six, and only six, rigid null modes per element.
    values=np.linalg.eigvalsh(elements[0]);assert np.count_nonzero(values<values[-1]*1e-12)==6
    K=assemble(elements,shape)
    for c in range(3):
        rigid=np.zeros((3,)+shape);rigid[c]=1
        np.testing.assert_allclose(K@rigid.ravel(),0,atol=1e-13)
    S=Multigrid(K,shape,np)
    a=project(rng.normal(size=K.shape[0]));b=project(rng.normal(size=K.shape[0]))
    np.testing.assert_allclose(a@S(b),b@S(a),atol=2e-12,rtol=2e-12)
    assert a@S(a)>0
    Z,inv,info=geneo_basis(K,elements,shape)
    assert info['rank']>0 and info['max_scaled_eigen_residual']<1e-12
    assert all(row['selected']>=6 for row in info['local_spectra'])
    correction=lambda v:project(Z@(inv@(Z.T@project(v))))
    np.testing.assert_allclose(a@correction(b),b@correction(a),atol=1e-10,rtol=1e-10)
    assert a@correction(a)>=0


@pytest.mark.parametrize('fine,coarse', [((8,12,16),(4,4,4)), ((9,12,15),(3,4,5))])
def test_gpu_transfer_exact_adjoint_and_cpu_reference(fine,coarse):
    rng=np.random.default_rng(985)
    fft.set_fft_backend('cupy');fft.set_cupy_plan_mode('manual');fft.set_cupy_rfft_batch_mode('full')
    Q=SymmetricCoordinates(MixedProjector(fine,(2.,3.,4.),np.ones(6,bool),cp),3.,2.)
    T=Transfer(Q,coarse);P=prolongation(fine,coarse)
    u=rng.normal(size=3*np.prod(coarse));v=rng.normal(size=3*np.prod(fine))
    np.testing.assert_allclose(cp.asnumpy(T.prolong(cp.asarray(u,cp.float32))).ravel(),P@u,atol=6e-7,rtol=2e-6)
    np.testing.assert_allclose(cp.asnumpy(T.restrict(cp.asarray(v,cp.float32).reshape((3,)+fine))),P.T@v,atol=3e-6,rtol=2e-5)
    # R is an admissible real RFFT vector. Derivative multiplier is the
    # inverse of the spectral displacement-to-strain reference root.
    r=Q.adjoint(cp.asarray(rng.normal(size=(6,)+fine),cp.float32))
    lifted=T(cp.asarray(u,cp.float32));pulled=T.adjoint(r)
    left=float(cp.vdot(lifted,r).real);right=float(cp.asarray(u,cp.float32)@pulled)
    np.testing.assert_allclose(left,right,atol=1e-4,rtol=5e-6)
    # Independent NumPy transforms and explicit matrix frequency multiplier.
    h=np.fft.rfftn((P@u).reshape((3,)+fine),axes=(1,2,3))
    frequencies=[np.fft.fftfreq(n)*n/L for n,L in zip(fine,(2.,3.,4.))]
    frequencies[-1]=np.fft.rfftfreq(fine[-1])*fine[-1]/4.
    k=np.asarray(np.meshgrid(*frequencies,indexing='ij'));kn=np.linalg.norm(k,axis=0)
    n=k/np.where(kn>0,kn,1);weight=np.full(h.shape[-1],np.sqrt(2));weight[0]=1
    if fine[-1]%2==0:weight[-1]=1
    expected=(np.sqrt(2)*h+(np.sqrt(3+8/3)-np.sqrt(2))*n*np.sum(n*h,axis=0))
    expected*=2*np.pi*kn*weight/np.sqrt(np.prod(fine))
    expected[:,0,0,0]=0
    for axis,size in enumerate(fine):
        if size%2==0:
            sl=[slice(None)]*4;sl[axis+1]=size//2;expected[tuple(sl)]=0
    actual=cp.asnumpy(lifted).reshape(expected.shape)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<8e-7
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)


def test_metric_pcg_against_dense_spd_and_true_residual():
    rng=np.random.default_rng(685);n=128
    v=rng.normal(size=(n,n));H=v.T@v+3*np.eye(n)
    w=rng.normal(size=(n,n));B=w.T@w/n+np.eye(n)
    rhs=rng.normal(size=n);h=cp.asarray(H,cp.float32);b=cp.asarray(B,cp.float32)
    dot=lambda a,c:float(cp.sum(a.astype(cp.float64)*c.astype(cp.float64)))
    action=lambda x:h@x
    x,it=metric_pcg(action,cp.asarray(rhs,cp.float32),dot,2e-5,1000,cp,
        precondition=lambda x:b@x,residual_smoothing=True)
    error=np.linalg.norm(rhs-H@cp.asnumpy(x))/np.linalg.norm(rhs)
    assert error<2.5e-5 and it<1000
    np.testing.assert_allclose(cp.asnumpy(x),np.linalg.solve(H,rhs),rtol=1e-3,atol=2e-6)
