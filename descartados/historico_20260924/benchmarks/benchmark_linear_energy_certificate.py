"""CPU64 discrete energy-error experiment, NOT a production stopping policy.

Synthetic small geometries with the G09 material parameters. The optional
tensor-only criterion deliberately differs from the physical 1e-5 criterion.
It cannot certify local fields, voxel discretization or experimental accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from profile_g09_fft import MATERIAL, ROOT
from pipeline.nonlinear_solver import _table, MixedProjector, _cg
from pipeline.nonlinear_material import J2Material, elastic_action
from pipeline.linear_krylov import SymmetricCoordinates
from linear_recycling_candidate import Builder, energy_error_certificate
from ffthompy.tensors import fft


def experiment(shape, material, relative_tensor_tolerance):
    fft.set_fft_backend('scipy')
    rng = np.random.default_rng(248)
    phase = (rng.random(shape)<.25).astype(np.uint8)
    ori = rng.normal(size=shape+(3,)); ori /= np.linalg.norm(ori, axis=-1, keepdims=True)
    phase.flags.writeable = ori.flags.writeable = False
    law = J2Material(E=material['Em'], nu=material['nu_m'])
    fiber = {s:material[k] for s,k in [('E_L','Ef_L'), ('E_T','Ef_T'), ('G_LT','G_LT'),
                                      ('nu_LT','nu_LT'), ('nu_TT','nu_TT')]}
    ids, table = _table(phase, ori, law, fiber); table = table.astype(np.float64)
    P = MixedProjector(shape, (1., 2., 3.), np.ones(6, bool), np)
    Q = SymmetricCoordinates(P, law.K, law.mu)
    def C(v): return elastic_action(v, ids, table)
    calls = 0
    def A(v):
        nonlocal calls
        calls += 1
        return Q.adjoint(C(Q.strain(v)))
    def dot(a, b): return float(np.vdot(a, b).real)
    def norm(v): return np.sqrt(dot(v, v))
    w, U = np.linalg.eigh(table[0]); invsqrt = (U*(1/np.sqrt(w)))@U.T
    eig = np.linalg.eigvalsh(np.einsum('ij,njk,kl->nil', invsqrt, table, invsqrt))
    alpha = float(eig.min())*(1-1e-12)
    N = int(np.prod(shape)); scale = material['Em']; goal = scale*relative_tensor_tolerance
    Es, bs, physical_rhs_norms = [], [], []
    for j in range(6):
        E = np.zeros((6,)+shape); E[j] = 1
        Es.append(E); bs.append(-Q.adjoint(C(E))); physical_rhs_norms.append(norm(P(C(E))))
    def tensor(fields):
        stresses = [C(e) for e in fields]
        return np.array([[np.sum(e*s)/N for s in stresses] for e in fields])
    reference = []
    for b in bs:
        z, _ = _cg(A, b, dot, 1e-12, 1000, np); reference.append(z)
    exact = tensor([E+Q.strain(z) for E,z in zip(Es, reference)])
    report = dict(shape=list(shape), material=material, seed=248, alpha=alpha,
        beta=float(eig.max()), tolerance_scale='matrix Young modulus, same units as input stiffness',
        scale_value=scale, target_absolute_tensor_operator_norm_error=goal, modes={})
    for mode in ['physical_1e-5', 'tensor_only']:
        started = time.perf_counter(); calls = 0; zs, residuals, ks = [], [], []
        physical = []; coarse = None
        builder = Builder(12, 4)
        for j, (b, denominator) in enumerate(zip(bs, physical_rhs_norms)):
            bn = norm(b)
            if mode == 'physical_1e-5':
                root_bound = np.sqrt(max(2*law.mu, law.K+4*law.mu/3))
                tolerance = 1e-5*denominator/(root_bound*bn) if bn else 1e-5
            else:
                tolerance = min(.3, np.sqrt(N*alpha*goal/6)/bn) if bn else 1e-5
            def capture(v):
                av = A(v); builder.capture(v, av, dot); return av
            z, k = _cg(capture if mode == 'tensor_only' and j == 0 else A,
                       b, dot, tolerance, 1000, np)
            if mode == 'tensor_only' and j == 0:
                coarse = builder.finish(A, k)
            zs.append(z); ks.append(k); residuals.append(b-A(z))
            physical.append(norm(P(C(Es[j]+Q.strain(z))))/denominator if denominator else 0.)
        fields = [E+Q.strain(z) for E,z in zip(Es, zs)]
        ceff = tensor(fields); R = np.column_stack(residuals)
        bound, trace = energy_error_certificate(R, alpha, N)
        error = ceff-exact
        deltas = [z-zstar for z,zstar in zip(zs, reference)]
        identity = np.array([[dot(a, A(b))/N for b in deltas] for a in deltas])
        entry = dict(cg=ks, total_cg=sum(ks), operator_calls_including_diagnostics=calls,
            cpu_wall_s_including_diagnostics=time.perf_counter()-started,
            max_physical_relative_residual=max(physical),
            physical_1e_5_passed=bool(max(physical)<=1e-5),
            Ceff=ceff.tolist(), measured_tensor_error_norm=float(np.linalg.norm(error, 2)),
            trace_upper_bound=trace, matrix_upper_bound_norm=float(np.linalg.norm(bound, 2)),
            energy_identity_max_difference=float(np.max(np.abs(error-identity))),
            certified_target_passed=trace<=goal,
            numerical_bound_margin_min_eigenvalue=float(np.linalg.eigvalsh(bound-error).min()))
        if np.linalg.norm(error, 2) > trace+1e-10 or entry['energy_identity_max_difference'] > 1e-10:
            raise RuntimeError('Energy identity/bound failed')
        if mode == 'physical_1e-5' and not entry['physical_1e_5_passed']:
            raise RuntimeError('Original physical residual failed')
        if mode == 'tensor_only':
            if not entry['certified_target_passed']:
                raise RuntimeError('Tensor-only trace bound failed')
            d = np.column_stack([coarse.ops.project(coarse.rows, r) for r in residuals])
            predicted = ceff-d.T@coarse.Hinv@d/N
            corrected = [z+coarse.seed(r) for z,r in zip(zs, residuals)]
            actual = tensor([E+Q.strain(z) for E,z in zip(Es, corrected)])
            corrected_r = np.column_stack([b-A(z) for b,z in zip(bs, corrected)])
            _, corrected_trace = energy_error_certificate(corrected_r, alpha, N)
            entry['coarse_correction'] = dict(
                identity_max_difference=float(np.max(np.abs(actual-predicted))),
                tensor_error_norm=float(np.linalg.norm(actual-exact, 2)),
                trace_upper_bound=corrected_trace, certified_target_passed=corrected_trace<=goal,
                decreases_energy_min_eigenvalue=float(np.linalg.eigvalsh(ceff-actual).min()),
                remains_upper_bound_min_eigenvalue=float(np.linalg.eigvalsh(actual-exact).min()))
            if (np.max(np.abs(actual-predicted)) > 1e-10 or
                    np.linalg.eigvalsh(ceff-actual).min() < -1e-10 or
                    np.linalg.eigvalsh(actual-exact).min() < -1e-10):
                raise RuntimeError('Coarse energy correction failed')
        report['modes'][mode] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relative-tensor-tolerance', type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists() or not np.isfinite(args.relative_tensor_tolerance) or args.relative_tensor_tolerance <= 0:
        parser.error('New output path and positive finite tensor tolerance required')
    material = json.loads(MATERIAL.read_text())
    paths = ['FFT/benchmarks/benchmark_linear_energy_certificate.py',
             'FFT/benchmarks/linear_recycling_candidate.py', 'FFT/pipeline/linear_krylov.py',
             'FFT/pipeline/nonlinear_solver.py', 'FFT/pipeline/nonlinear_coordinates.py']
    report = dict(scope=__doc__, source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        for p in paths}, cases=[experiment(s, material, args.relative_tensor_tolerance)
                              for s in [(9, 9, 9), (10, 8, 12)]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for case in report['cases']:
        print(case['shape'], [(k,v['total_cg'],v['max_physical_relative_residual'],
            v['measured_tensor_error_norm'],v['trace_upper_bound']) for k,v in case['modes'].items()])


if __name__ == '__main__':
    main()
