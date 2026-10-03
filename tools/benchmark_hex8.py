"""Compare CuFHom HEX8 with the unchanged official UNO-CG/FANS implementation.

Requires CuPy, PyTorch, and a checkout of DataAnalyticsEngineering/UNOCG.
The input phase.npy is the original Z,Y,X test/dset_1 extracted from DaRUS.
Both timed solvers include RHS, CG and the same original nodal residual audit.
"""
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def metrics(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    delta = actual-reference
    return {'relative_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-300)),
            'maximum_absolute': float(np.max(np.abs(delta)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--small-only', action='store_true')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('Positive repetition count required')
    if (args.output/'comparison.json').exists():
        parser.error('Output already contains a comparison')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.reference.resolve()))
    import torch
    from cufhom.fft.linear.hex8 import PeriodicHex8, cp, solve_hex8
    from unocg.materials.mechanical import LinearElasticity
    from unocg.preconditioners.torch import FansPreconditioner
    from unocg.problems import BC
    from unocg.problems.mechanical import MechanicalProblem
    from unocg.solvers.torch import CgSolver

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    macro = np.array([.025, -.05, .025, 0., 0., 0.])
    E, nu = np.array([75., 400.]), np.array([.3, .2])
    lame = np.column_stack([E*nu/((1+nu)*(1-2*nu)), E/(2*(1+nu))])

    def reference(phase, device):
        config = {'device': device, 'dtype': torch.float64}
        material = LinearElasticity(n_dim=3, **config)
        problem = MechanicalProblem(phase.shape, material=material, bc=BC.PERIODIC, quad_degree=2)
        params = torch.tensor(lame, **config)
        fields = problem.get_param_fields(torch.tensor(phase, **config).unsqueeze(0), params)
        loads = torch.tensor(macro[None], **config)
        prec = FansPreconditioner(problem, problem.stiffness(params.mean(-2)))
        cg = CgSolver(problem, preconditioner=prec)
        return problem, fields, loads, prec, cg

    report = {'reference_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=args.reference, text=True).strip(),
        'operator': 'periodic fully integrated HEX8; all nonzero frequencies retained',
        'dtype': 'float64', 'rtol': 1e-6, 'norm': 'true nodal infinity norm relative to the same original affine RHS',
        'macro': macro.tolist(), 'E_GPa': E.tolist(), 'nu': nu.tolist(), 'small_checks': [], 'runs': []}

    def save():
        (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')

    with torch.inference_mode():
        rng = np.random.default_rng(20261003)
        for shape in [(3, 4, 5), (4, 4, 4)]:
            phase = rng.integers(0, 2, shape, dtype=np.uint8)
            op = PeriodicHex8(phase)
            problem, fields, loads, prec, cg = reference(phase, 'cpu')
            matvec = problem.get_matvec_module(dtype=torch.float64, device='cpu')
            rhs_module = problem.get_rhs_module(dtype=torch.float64, device='cpu')
            u = rng.normal(size=(3, *shape))
            u -= u.mean((1, 2, 3), keepdims=True)
            tensor = torch.tensor(u[None, None])
            expected = matvec(tensor, fields).numpy()[0, 0]
            b = rhs_module(tensor, fields, loads).numpy()[0, 0]
            conditioned = problem.reshape_field(prec.apply(problem.reshape_vec(tensor))).numpy()[0, 0]
            result = solve_hex8(op, macro, rtol=1e-10)
            solution = problem.reshape_field(cg.solve(fields, loads, rtol=1e-10)['sol']).numpy()[0, 0]
            check = {'shape': shape, 'K': metrics(op.action(u), expected),
                     'b': metrics(op.rhs(macro), b),
                     'preconditioner': metrics(op.precondition(u), conditioned/phase.size),
                     'solution': metrics(result['solution'], solution), 'iterations': result['iterations']}
            for key in ['K', 'b', 'preconditioner']:
                assert check[key]['relative_l2'] < 1e-12, check
            assert check['solution']['relative_l2'] < 1e-8, check
            report['small_checks'].append(check)
            print('SMALL CHECK', json.dumps(check), flush=True)
        save()
        if args.small_only:
            return

        phase = np.load(args.phase)
        if phase.shape != (192, 192, 192):
            parser.error('Published benchmark requires the original 192 cubed sample')
        report['phase_sha256'] = hashlib.sha256(phase.tobytes()).hexdigest()
        if report['phase_sha256'] != '7e852b07e3d36e5dc58e9cd686ae331b1e6e494f318e485dd1abdb4917c86111':
            parser.error('Input does not match the original published test/dset_1 sample')
        if cp is None or not torch.cuda.is_available():
            parser.error('The 192 cubed benchmark requires CuPy and PyTorch with CUDA')
        report['phase1_fraction'] = float(phase.mean())
        report['gpu'] = torch.cuda.get_device_name()
        report['torch_version'], report['cupy_version'] = torch.__version__, cp.__version__

        def synchronize():
            cp.cuda.get_current_stream().synchronize()
            torch.cuda.synchronize()

        synchronize()
        start = time.perf_counter()
        problem, fields, loads, prec, cg = reference(phase, 'cuda:0')
        module = cg.get_module(dtype=torch.float64, device='cuda:0', rtol=1e-6)
        common_residual = problem.get_residual_module(dtype=torch.float64, device='cuda:0')
        guess = module.zero_guess(fields, loads)
        common_rhs = common_residual(guess, fields, loads)
        denominator = float(common_rhs.abs().max())
        synchronize()
        report['fans_preparation_s'] = time.perf_counter()-start
        start = time.perf_counter()
        op = PeriodicHex8(phase, xp=cp)
        # The official inverse contains an extra positive scalar N. Apply the
        # identical normalization to our stored weights, without per-step cost.
        op.inverse *= phase.size
        synchronize()
        report['cufhom_preparation_s'] = time.perf_counter()-start
        report['large_rhs_match'] = metrics(cp.asnumpy(op.rhs(macro)), common_rhs.cpu().numpy()[0, 0])
        assert report['large_rhs_match']['relative_l2'] < 1e-12
        report['reference_preconditioner_scalar'] = int(phase.size)
        probe = cp.random.RandomState(27).standard_normal((3, *phase.shape), dtype=cp.float64)
        original_action = problem.get_matvec_module(dtype=torch.float64, device='cuda:0')(
            torch.utils.dlpack.from_dlpack(probe)[None, None], fields)
        delta = op.action(probe)-cp.from_dlpack(original_action)[0, 0]
        report['large_operator_relative_l2'] = float(cp.linalg.norm(delta)/cp.linalg.norm(cp.from_dlpack(original_action)))
        assert report['large_operator_relative_l2'] < 1e-12
        del probe, original_action, delta

        count = [0]

        def count_iteration(*unused):
            count[0] += 1

        module.iteration_layers[0].register_forward_hook(count_iteration)
        last_solutions = {}

        def run(version, repetition):
            gc.collect()
            torch.cuda.empty_cache()
            cp.get_default_memory_pool().free_all_blocks()
            synchronize()
            start = time.perf_counter()
            if version == 'fans':
                count[0] = 0
                u = module(guess, fields, loads)
                iterations = count[0]
                u -= u.mean(dim=(-3, -2, -1), keepdim=True)
            else:
                result = solve_hex8(op, macro)
                u = torch.utils.dlpack.from_dlpack(result['solution'])[None, None]
                iterations = result['iterations']
            # The same original HEX8 residual implementation audits BOTH solvers.
            residual = common_residual(u, fields, loads)
            relative = float(residual.abs().max())/denominator
            synchronize()
            elapsed = time.perf_counter()-start
            assert relative <= 1e-6, (version, relative)
            row = {'solver': version, 'repetition': repetition, 'wall_s': elapsed,
                   'iterations': iterations, 'common_true_relative_inf': relative}
            last_solutions[version] = u.cpu().numpy()[0, 0].copy()
            del u, residual
            if version == 'cufhom':
                del result
            report['runs'].append(row)
            print('RUN', json.dumps(row), flush=True)
            save()

        for version in ['fans', 'cufhom']:
            run(version, 'warmup')
        for repetition in range(args.repetitions):
            order = ['fans', 'cufhom'] if repetition % 2 == 0 else ['cufhom', 'fans']
            for version in order:
                run(version, repetition)
        report['median_s'] = {version: float(np.median([r['wall_s'] for r in report['runs']
            if r['solver'] == version and r['repetition'] != 'warmup'])) for version in ['fans', 'cufhom']}
        report['local_speed_ratio'] = report['median_s']['fans']/report['median_s']['cufhom']
        report['displacement_comparison'] = metrics(last_solutions['cufhom'], last_solutions['fans'])
        assert report['displacement_comparison']['relative_l2'] < 1e-6
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        reference_field = problem.compute_field(
            problem.reshape_vec(torch.tensor(last_solutions['fans'][None, None], device='cuda:0')),
            fields, loads).cpu().numpy()[0, 0, 3:]
        actual_field = op.stress_field(last_solutions['cufhom'], macro)
        report['cell_stress_comparison'] = metrics(actual_field, reference_field)
        assert report['cell_stress_comparison']['relative_l2'] < 1e-6
        report['mean_stress_fans_GPa'] = reference_field.mean((1, 2, 3)).tolist()
        report['mean_stress_cufhom_GPa'] = actual_field.mean((1, 2, 3)).tolist()
        report['timing_scope'] = 'RHS, CG and common original true residual audit, after warmup; excludes construction, CPU field transfers and validation fields'
        report['hex8_source_sha256'] = hashlib.sha256(Path(sys.modules[PeriodicHex8.__module__].__file__).read_bytes()).hexdigest()
        save()
        print('FINAL', json.dumps({k: report[k] for k in ['median_s', 'local_speed_ratio',
            'displacement_comparison', 'cell_stress_comparison']}), flush=True)


if __name__ == '__main__':
    main()
