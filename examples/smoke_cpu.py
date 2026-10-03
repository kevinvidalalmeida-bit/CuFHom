"""Tiny analytic checks; no CUDA, campaign data or FFT fields saved."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from cufhom.geometry import GeometryRequest, generate_geometry
from cufhom.fft import solve_homogenization
from cufhom.fft import solve_nonlinear_homogenization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", action="store_true",
                        help="Also generate and solve straight and flexible geometries")
    args = parser.parse_args()
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

    if args.geometry:
        with TemporaryDirectory(prefix='cufhom-smoke-') as folder:
            for model in ('straight', 'flexible'):
                output = Path(folder)/model
                summary = generate_geometry(GeometryRequest(
                    output=output, fiber_model=model,
                    shape='cylinder' if model == 'straight' else 'capsule',
                    solver='a4' if model == 'straight' else 'a2',
                    box=12., ar=2., vf=.02, grid=36, segments=2,
                    a2=(1., 0.), threads=1, seed=572, verbose=False))
                assert summary['accepted'], summary
                with np.load(output/'geometry.npz', allow_pickle=False) as data:
                    phase, ori = data['phase'], data['ori']
                C = solve_homogenization(dict(parameters, phase_array=phase, ori_array=ori))
                assert np.isfinite(C).all() and np.linalg.eigvalsh((C+C.T)/2).min() > 0
                result = solve_nonlinear_homogenization(dict(
                    phase_array=phase, ori_array=ori, backend='scipy', path=[0., .001],
                    initial_increment=.001,
                    matrix=dict(E=E, nu=nu, yield_stress=.015, hardening=.03)))
                assert len(result['history']) >= 2
                print(f"PASS: {model}, {summary['accepted_count']} fibers, "
                      f"Vf={summary['vf']:.6f}, linear and nonlinear FFT")


if __name__ == '__main__':
    main()
