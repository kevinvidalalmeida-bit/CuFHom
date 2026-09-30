"""CPU64 diagnostic of exact fiber-only polarization; not a production solver.

Derives the reduced system from the existing spectral operator, without
changing its grid or Nyquist convention. No G09 performance claim is made.
"""
import argparse
import json
import inspect
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from profile_g09_fft import MATERIAL, G09_N240, quiet_solver, solve_homogenization
from pipeline.fft_solver import (Enu_to_KG, ElasticTensor, TI_stiffness_voigt,
    voigt_to_mandel, _build_cfield_cpu)
from pipeline.nonlinear_solver import MixedProjector
from pipeline.nonlinear_reference import IsotropicReference
from ffthompy.tensors import fft


def check(shape, material, Cm, Cf, bulk, shear):
    rng = np.random.default_rng(93)
    phase = (rng.random(shape) < .25).astype(np.uint8)
    ori = rng.normal(size=shape+(3,)); ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    C, _, _ = _build_cfield_cpu(phase, ori, Cm, Cf, np.float64, storage='full')
    selected = np.flatnonzero(phase.ravel())
    local = np.moveaxis(C.reshape(6, 6, -1)[:, :, selected], -1, 0)
    w, v = np.linalg.eigh(local-Cm)
    if w.min() <= 0:
        raise ValueError('This square-root polarization derivation requires positive contrast')
    root = (v*np.sqrt(w)[:, None, :]) @ v.transpose(0, 2, 1)
    fft.set_fft_backend('scipy')
    P = MixedProjector(shape, (1., 1., 1.), np.ones(6, bool), np)
    Gamma = IsotropicReference(P, bulk, shear); Gamma.set_shift(0., np.dtype('float64'))

    def scatter(x):
        out = np.zeros((6, int(np.prod(shape))))
        out[:, selected] = np.einsum('nij,nj->ni', root, x.reshape(-1, 6)).T
        return out.reshape((6,)+shape)

    def restrict(x):
        return np.einsum('nij,nj->ni', root, x.reshape(6, -1)[:, selected].T).ravel()

    A = LinearOperator((6*len(selected),)*2,
                       matvec=lambda x: x+restrict(Gamma(scatter(x))), dtype=np.float64)
    ceff = np.zeros((6, 6)); residuals = []; iterations = []
    for load in range(6):
        E = np.zeros((6,)+shape); E[load] = 1
        count = [0]
        def callback(x):
            count[0] += 1
        tolerance = {'rtol' if 'rtol' in inspect.signature(cg).parameters else 'tol': 1e-12}
        p, status = cg(A, restrict(E), **tolerance, atol=0., maxiter=2000, callback=callback)
        if status != 0:
            raise RuntimeError(f'Reduced solve failed: {status}')
        strain = E-Gamma(scatter(p))
        stress = np.einsum('ij...,j...->i...', C, strain)
        residuals.append(float(np.linalg.norm(P(stress))/np.linalg.norm(P(np.einsum('ij...,j...->i...', C, E)))))
        ceff[:, load] = stress.mean(axis=(1, 2, 3))
        iterations.append(count[0])
    parameters = dict(input_dir='.', phase_array=phase, ori_array=ori,
        **{k: material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')},
        fft_backend='scipy', solver_real_dtype='float64', solver_rtol=1e-11,
        solver_fft_form='r', projection_storage='direct', cfield_storage='full',
        linear_formulation='strain')
    with quiet_solver():
        reference = solve_homogenization(parameters)
    difference = float(np.linalg.norm(ceff-reference)/np.linalg.norm(reference))
    if (not np.isfinite([difference, *residuals]).all()
            or difference > 1e-9 or max(residuals) > 1e-9):
        raise RuntimeError('Polarization equivalence or physical equilibrium failed')
    return dict(shape=shape, fiber_voxels=len(selected), cg=iterations,
                Ceff_relative_difference=difference, max_physical_residual=max(residuals))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output file')
    m = json.loads(MATERIAL.read_text()); K, mu = Enu_to_KG(m['Em'], m['nu_m'])
    Cm = ElasticTensor(bulk=K, mu=mu).mandel
    Cf = voigt_to_mandel(TI_stiffness_voigt(m['Ef_L'], m['Ef_T'], m['nu_LT'], m['nu_TT'], m['G_LT']))
    w = np.linalg.eigvalsh(Cf-Cm)
    phase = np.load(G09_N240/'phase.npy', mmap_mode='r'); nf = int(np.count_nonzero(phase==1))
    result = dict(g09_voxels=phase.size, fiber_voxels=nf, fiber_fraction=nf/phase.size,
        local_contrast_eigenvalues=w.tolist(), rank_one_relative_frobenius_discard=float(np.linalg.norm(w[:-1])/np.linalg.norm(w)),
        coordinate_vector_bytes=3*240*240*121*8, polarization_vector_bytes=6*nf*4,
        checks=[check(shape, m, Cm, Cf, K, mu) for shape in ((5, 5, 5), (6, 4, 8))],
        scope='CPU64 algebraic verification on small random microstructures; G09 material constants. '
              'G09 counts/eigenvalues only: no reduced G09 solve or speed claim.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
