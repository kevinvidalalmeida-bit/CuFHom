"""Mandel GPU rotation against the independent CPU fourth-order tensor."""
import numpy as np
import pytest
from cufhom.pipeline.fft_solver import cp,rotate_C_mandel,rotation_matrix_from_vector


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('packed',[False,True])
def test_anisotropic_rotation(dtype,packed):
    if cp is None:pytest.skip('CuPy unavailable')
    from cufhom.pipeline.material_rotation import rotate_mandel_batch_gpu
    rng=np.random.default_rng(52)
    a=rng.normal(size=(6,6));C=a.T@a+np.eye(6)
    R=np.asarray([rotation_matrix_from_vector(v) for v in rng.normal(size=(31,3))])
    expected=np.asarray([rotate_C_mandel(C,r) for r in R])
    if packed:expected=expected[:,np.triu_indices(6)[0],np.triu_indices(6)[1]]
    actual=cp.asnumpy(rotate_mandel_batch_gpu(C,R,dtype,packed,7))
    np.testing.assert_allclose(actual,expected,rtol=5e-7 if dtype==np.float32 else 2e-13,
                               atol=2e-7 if dtype==np.float32 else 2e-13)
    if not packed:
        np.testing.assert_allclose(np.linalg.eigvalsh(actual),
            np.broadcast_to(np.linalg.eigvalsh(C),(31,6)),rtol=1e-6 if dtype==np.float32 else 2e-13)
    empty=rotate_mandel_batch_gpu(C,R[:0],dtype,packed)
    assert empty.shape==((0,21) if packed else (0,6,6))
