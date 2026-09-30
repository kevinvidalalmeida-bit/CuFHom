"""Tiny analytic checks; no CUDA, campaign data or FFT fields saved."""
import numpy as np
from cufhom.pipeline.fft_solver import solve_homogenization
from cufhom.pipeline.nonlinear_solver import solve_nonlinear_homogenization


def main():
    phase = np.zeros((4, 4, 4), dtype=np.uint8)
    ori = np.zeros(phase.shape+(3,)); ori[..., 0] = 1.
    E, nu = 3., .3
    parameters = dict(input_dir='.', seed=0, phase_array=phase, ori_array=ori,
                      Em=E, nu_m=nu, Ef_L=70., Ef_T=8., nu_LT=.25, nu_TT=.3, G_LT=5.,
                      fft_backend='scipy', solver_real_dtype='float64',
                      solver_rtol=1e-10, require_convergence=True, solver_verbose=False)
    computed = solve_homogenization(parameters)
    lam, mu = E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))
    expected = np.eye(6)*2*mu
    expected[:3, :3] += lam
    np.testing.assert_allclose(computed, expected, atol=1e-10, rtol=1e-10)
    result = solve_nonlinear_homogenization(dict(
        phase_array=phase, ori_array=ori, backend='scipy', path=[0., .002],
        initial_increment=.001, matrix=dict(E=E, nu=nu, yield_stress=.015, hardening=.03)))
    np.testing.assert_allclose(result['history'][-1]['stress'][0], E*.002, atol=1e-10)
    print('PASS: homogeneous linear stiffness and nonlinear elastic limit, 4^3 CPU')


if __name__ == '__main__':
    main()
