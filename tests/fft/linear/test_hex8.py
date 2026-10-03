"""HEX8 energy, independent nodal assembly and true-residual acceptance."""
import numpy as np
import pytest
from cufhom.fft.linear.hex8 import PeriodicHex8, cp, solve_hex8


def dense_system(op, macro):
    size = op.phase.size
    K, b = np.zeros((3*size, 3*size)), np.zeros(3*size)
    for cell in np.ndindex(op.shape):
        nodes = [(np.array(cell)+node) % op.shape for node in op.nodes]
        dofs = np.array([c*size+np.ravel_multi_index(tuple(node), op.shape)
                         for node in nodes for c in range(3)])
        C = op.C[op.phase[cell]]
        for B in op.B:
            K[np.ix_(dofs, dofs)] += B.T@C@B/(8*size)
            b[dofs] -= B.T@C@macro/(8*size)
    return K, b


@pytest.mark.parametrize('shape', [(2, 3, 4), (3, 3, 3)])
def test_matches_dense_assembly_and_solve(shape):
    rng = np.random.default_rng(35)
    phase = rng.integers(0, 2, shape, dtype=np.uint8)
    op = PeriodicHex8(phase, lengths=(1.2, .7, 2.))
    macro = np.array([.025, -.05, .025, .01, -.02, .03])
    K, b = dense_system(op, macro)
    u = rng.normal(size=(3, *shape))
    np.testing.assert_allclose(op.action(u).ravel(), K@u.ravel(), rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(op.rhs(macro).ravel(), b, rtol=2e-13, atol=1e-14)
    actual = solve_hex8(op, macro, rtol=1e-10)
    reference = np.linalg.pinv(K, rcond=1e-13)@b
    np.testing.assert_allclose(actual['solution'].ravel(), reference, rtol=2e-8, atol=2e-11)
    residual = b-K@actual['solution'].ravel()
    assert np.max(np.abs(residual))/np.max(np.abs(b)) <= 1e-10


def test_affine_patch_and_rigid_rotation():
    op = PeriodicHex8(np.zeros((3, 4, 5), np.uint8), lengths=(.9, .8, 1.5))
    xyz = op.nodes[:, ::-1]*np.array([1.5/5, .8/4, .9/3])
    gradient = np.array([[.1, .2, -.3], [.4, .5, -.6], [.7, .8, .9]])
    displacement = xyz@gradient.T
    expected = np.r_[np.diag(gradient),
                      (gradient[0, 1]+gradient[1, 0])/np.sqrt(2),
                      (gradient[0, 2]+gradient[2, 0])/np.sqrt(2),
                      (gradient[1, 2]+gradient[2, 1])/np.sqrt(2)]
    for B in op.B:
        np.testing.assert_allclose(B@displacement.ravel(), expected, atol=2e-15)
        rotation = xyz@(gradient-gradient.T).T
        np.testing.assert_allclose(B@rotation.ravel(), 0, atol=2e-15)


def test_homogeneous_inverse_retains_nyquist_and_zero_load():
    op = PeriodicHex8(np.zeros((4, 4, 6), np.uint8), E=(75., 75.), nu=(.3, .3))
    rng = np.random.default_rng(18)
    u = rng.normal(size=(3, *op.shape))
    u -= u.mean((1, 2, 3), keepdims=True)
    np.testing.assert_allclose(op.precondition(op.action(u)), u, rtol=1e-12, atol=1e-12)
    checker = np.broadcast_to((-1.)**np.arange(6), (3, *op.shape)).copy()
    assert np.linalg.norm(op.action(checker)) > 1
    np.testing.assert_allclose(op.action(np.ones_like(u)), 0, atol=2e-13)
    result = solve_hex8(op, [0]*6)
    assert result['iterations'] == 0
    np.testing.assert_array_equal(result['solution'], np.zeros_like(u))


def test_nonconvergence_and_invalid_inputs():
    phase = np.random.default_rng(45).integers(0, 2, (3, 4, 5), dtype=np.uint8)
    op = PeriodicHex8(phase)
    with pytest.raises(RuntimeError, match='failed true residual'):
        solve_hex8(op, [.025, -.05, .025, 0, 0, 0], maxiter=1)
    with pytest.raises(ValueError):
        PeriodicHex8(phase, nu=(.3, .5))
    with pytest.raises(ValueError):
        PeriodicHex8(phase*2)
    with pytest.raises(ValueError):
        solve_hex8(op, [0]*6, rtol=0)


@pytest.mark.parametrize('phase_value', [0, 1])
def test_uniform_phase_affine_response_has_exact_zero_rhs(phase_value):
    op = PeriodicHex8(np.full((3, 4, 5), phase_value, np.uint8))
    macro = np.array([.03, .04, -.01, .02, -.05, .07])
    result = solve_hex8(op, macro)
    assert result['iterations'] == 0
    np.testing.assert_array_equal(op.rhs(macro), 0)
    np.testing.assert_allclose(op.mean_stress(result['solution'], macro), op.C[phase_value]@macro, atol=2e-13)
    mixed = PeriodicHex8(np.indices((3, 4, 5)).sum(0) % 2, E=(75., 75.), nu=(.3, .3))
    assert solve_hex8(mixed, macro)['iterations'] == 0


def test_gpu_operator_inverse_and_solution_match_cpu():
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(95)
    phase = rng.integers(0, 2, (4, 3, 6), dtype=np.uint8)
    cpu, gpu = PeriodicHex8(phase), PeriodicHex8(phase, xp=cp)
    u = rng.normal(size=(3, *phase.shape))
    np.testing.assert_allclose(cp.asnumpy(gpu.action(cp.asarray(u))), cpu.action(u), atol=2e-12)
    rhs = cpu.rhs([.025, -.05, .025, .01, .02, .03])
    np.testing.assert_allclose(cp.asnumpy(gpu.precondition(cp.asarray(rhs))), cpu.precondition(rhs), atol=1e-14)
    a, b = [solve_hex8(op, [.025, -.05, .025, .01, .02, .03], rtol=1e-10)
            for op in (cpu, gpu)]
    np.testing.assert_allclose(cp.asnumpy(b['solution']), a['solution'], atol=1e-12)
    assert a['iterations'] == b['iterations']
    assert b['true_relative_inf'] <= 1e-10
