"""Paired, warmed A2/A4 generation timings with independent accepted exports."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import numba
import numpy as np

from cufhom.cli.geometry import run
from cufhom.pipeline import sam_staged_insertion as staged


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.runs.exists() or not 1 <= args.repeats <= 10:
        parser.error('Use new output paths and 1 to 10 repeats')
    args.runs.mkdir(parents=True)
    numba.set_num_threads(1)
    reference_path = Path(__file__).with_name('reference_before.py')
    spec = importlib.util.spec_from_file_location('cufhom.pipeline._sam_baseline', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    implementations = dict(before=(reference.propose, reference.relax_contact_energy),
                           after=(staged.propose, staged.relax_contact_energy))
    report = dict(scope=__doc__, numba_threads=1, repeats=args.repeats,
                  compilation_excluded=True, cases={})
    for mode in ('a2', 'a4'):
        baseline = ROOT/f'benchmarks/results/sam_optimization_{mode}_before_20261002'
        parameters = json.loads((baseline/'request.json').read_text())['parameters']
        state_reference = None
        rows = {version: [] for version in implementations}
        for repetition in range(-1, args.repeats):
            # Alternate order after both implementations have warmed up.
            versions = list(implementations)
            if repetition % 2 == 1:
                versions.reverse()
            for version in versions:
                folder = args.runs/f'{mode}_{version}_{"warmup" if repetition < 0 else repetition}'
                values = dict(parameters, output=folder)
                for name in ('resume', 'orientation_json'):
                    values[name] = Path(values[name]) if values[name] else None
                options = argparse.Namespace(**values)
                propose, relax = implementations[version]
                with (args.runs/f'{folder.name}.log').open('w') as log, \
                     contextlib.redirect_stdout(log), \
                     patch.object(staged, 'propose', propose), \
                     patch.object(staged, 'relax_contact_energy', relax):
                    code = run(options)
                summary = json.loads((folder/'summary.json').read_text())
                assert code == 0 and summary['accepted'], (mode, version, summary['status'])
                with np.load(folder/'last_geometrically_valid.npz', allow_pickle=False) as saved:
                    state = saved['state']
                if state_reference is None:
                    state_reference = state.copy()
                else:
                    np.testing.assert_array_equal(state, state_reference)
                with np.load(folder/'geometry.npz', allow_pickle=False) as geometry:
                    assert geometry['phase'].shape == (192, 192, 192)
                    assert np.isclose(np.count_nonzero(geometry['phase'])/192**3, summary['vf'])
                    assert bool(geometry['A4_target'].size) == (mode == 'a4')
                row = dict(directory=str(folder), generation_s=summary['wall_s'],
                           total_s=summary['total_wall_s'], count=len(state),
                           vf=summary['vf'], voxel_A2_error=summary['voxel_A2_error'],
                           voxel_A4_error=summary.get('voxel_A4_error'))
                if repetition >= 0:
                    rows[version].append(row)
                print(f'{mode} {version} repeat={repetition}: {row["total_s"]:.4f}s', flush=True)
        medians = {version: float(np.median([row['total_s'] for row in samples]))
                   for version, samples in rows.items()}
        report['cases'][mode] = dict(samples=rows, median_total_s=medians,
                                    speedup=medians['before']/medians['after'],
                                    identical_continuous_state=True,
                                    state_sha256=hashlib.sha256(state_reference.tobytes()).hexdigest())
    sources = [reference_path, Path(__file__),
               ROOT/'src/cufhom/pipeline/sam_contact_energy.py',
               ROOT/'src/cufhom/pipeline/sam_staged_insertion.py',
               ROOT/'src/cufhom/cli/geometry.py']
    report['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({mode: row['median_total_s'] | {'speedup': row['speedup']}
                      for mode, row in report['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
