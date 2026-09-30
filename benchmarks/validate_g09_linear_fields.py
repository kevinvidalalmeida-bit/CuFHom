"""Six-load field comparison outside timing, with immutable G09 inputs."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path

import numpy as np
from profile_g09_linear import VERIFIED_SETTINGS, sha256
from cufhom.pipeline.fft_solver import compile_cfield_geometry, solve_homogenization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    files = [args.geometry/'phase.npy', args.geometry/'ori.npy', args.material]
    hashes = [sha256(p) for p in files]
    phase, ori = (np.load(p) for p in files[:2])
    phase.flags.writeable = ori.flags.writeable = False
    material = json.loads(args.material.read_text())
    common = dict(input_dir=str(args.geometry), phase_array=phase, ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase, ori),
        fft_backend='cupy', solver_rtol=1e-5, solver_atol=0., require_convergence=True,
        **{k: material[k] for k in ('Em', 'nu_m', 'Ef_L', 'Ef_T', 'nu_LT', 'nu_TT', 'G_LT')})
    references, fields = {}, []
    def compare(load_id, value):
        ref = references.pop(load_id)
        diff = value.astype(np.float64)-ref
        error = float(np.linalg.norm(diff.ravel())/np.linalg.norm(ref.ravel()))
        pointwise = float(np.max(np.abs(diff))/np.max(np.abs(ref)))
        if not np.isfinite(error) or error > 3e-5 or pointwise > 3e-5:
            raise RuntimeError(f'Field mismatch in load {load_id}: {error}, {pointwise}')
        fields.append(dict(load_id=int(load_id), relative_l2=error,
                           max_difference_over_reference_max=pointwise))
    with open(os.devnull, 'w') as sink, redirect_stdout(sink):
        before = solve_homogenization(dict(common, **VERIFIED_SETTINGS,
            solution_field_consumer=lambda i, field: references.__setitem__(i, field.copy()),
            solver_timing_path=str(args.output/'before.json')))
        after = solve_homogenization(dict(common, solution_field_consumer=compare,
            solver_timing_path=str(args.output/'after.json')))
    assert len(fields) == 6 and not references
    error = float(np.linalg.norm(after-before)/np.linalg.norm(before))
    assert error < 1e-6
    assert hashes == [sha256(p) for p in files]
    for name in ('before', 'after'):
        report = json.loads((args.output/f'{name}.json').read_text())
        infos = report['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert all(i['true_residual_converged'] and i['true_norm_res_rel'] <= 1e-5 for i in infos)
    result = dict(scope='Complete fluctuation strain fields; separate from performance measurements.',
        fields=fields, Ceff_relative_difference=error, input_sha256=hashes, inputs_unchanged=True)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
