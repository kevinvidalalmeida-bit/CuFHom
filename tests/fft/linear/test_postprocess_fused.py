"""Full effective-matrix products versus independent CPU64 contraction."""
import numpy as np
import pytest

from cufhom_core.postprocess import assembly_matrix
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import cp


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_general_real_fields(backend, dtype, noncontiguous):
    if backend == 'cupy' and cp is None:
        pytest.skip('CuPy unavailable')
    xp = np if backend == 'numpy' else cp
    rng = np.random.default_rng(309)
    shape, dimension = (5, 4, 7), 3
    fields = []
    for i in range(dimension):
        data = xp.asarray(rng.normal(size=(dimension,)+shape).astype(dtype))
        if noncontiguous:
            data = data[..., ::-1]
        fields.append(Tensor(name=str(i), val=data, order=1, N=shape,
                             Fourier=False, fft_form='r'))
    # Deliberately non-symmetric map: test all entries independently instead
    # of mirroring a triangle. It need not represent an elastic material.
    matrix = xp.asarray(rng.normal(size=(dimension, dimension)).astype(dtype))
    class Operator:
        N = shape
        def __call__(self, field):
            value = (matrix @ field.val.reshape(dimension, -1)).reshape((dimension,)+shape)
            return field.copy(val=value)
    action = Operator()
    to_cpu = np.asarray if xp is np else cp.asnumpy
    before = [to_cpu(f.val).copy() for f in fields]
    expected = np.array([[np.sum(to_cpu(action(a).val).astype(np.float64)*b.astype(np.float64))
                          /np.prod(shape) for b in before] for a in fields])
    candidate = assembly_matrix(action, fields, method='fused')
    tol = 2e-13 if (dtype == np.float64 or (backend == 'cupy' and not noncontiguous)) else 5e-7
    np.testing.assert_allclose(candidate, expected, rtol=tol, atol=tol)
    for original, field in zip(before, fields):
        np.testing.assert_array_equal(to_cpu(field.val), original)
