"""Current CuFHom timings on audited short-fiber masters, one sequential GPU."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from profile_g09_linear import run
from cufhom.pipeline.fft_solver import cp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=ROOT/'benchmarks/results/performance_audit_20261001/fiber_geometries/manifest.json')
    parser.add_argument('--material',type=Path,default=ROOT/'benchmarks/performance_audit_20261001/material.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--grids',type=int,nargs='+',default=[32,48,64,96,128,192,256])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    manifest=json.loads(args.prepared.read_text())
    cases={row['case']:row for row in manifest['cases'] if row['AR'] in (5,10,20)}
    assert len(cases)==9
    for case in cases.values():
        assert case['periodic_audit']['accepted'] and case['box_um']==2*case['length_um']
        assert sha(ROOT/case['table'])==case['master_sha256']
    rows=[row for row in manifest['rasters'] if row['case'] in cases
          and (row['N'] in args.grids or row['native_resolution_6'])]
    assert len({(r['case'],r['N']) for r in rows})==len(rows)
    assert all((ROOT/r['geometry']/'phase.npy').exists() and (ROOT/r['geometry']/'ori.npy').exists() for r in rows)
    rng=np.random.default_rng(20261002);ordered=[]
    for n in sorted({r['N'] for r in rows}):
        group=[r for r in rows if r['N']==n]
        ordered.extend(group[i] for i in rng.permutation(len(group)))
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')}
    campaign=dict(schema=1,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Nine audited continuous masters, random orientation, d=1um, capsule axis L=AR*d, cubic box=2L. '
              'One warmup and five timed full six-load solves per geometry/grid. Current defaults, FP32 fields/FFT, '
              'FP64 products, physical residual 1e-5. Compilation/loading before timing. One realization per Vf/AR.',
        repeats=args.repeats,source_sha256=source,runner_sha256=sha(Path(__file__)),
        prepared_sha256=sha(args.prepared),material_sha256=sha(args.material),cases=cases,
        grid_order=[dict(case=r['case'],N=r['N']) for r in ordered],completed=[])
    path=args.output/'campaign.json'
    if path.exists():
        previous=json.loads(path.read_text())
        assert previous['source_sha256']==source and previous['runner_sha256']==campaign['runner_sha256']
        assert previous['repeats']==args.repeats and previous['grid_order']==campaign['grid_order']
    path.write_text(json.dumps(campaign,indent=2)+'\n')
    for index,row in enumerate(ordered):
        name=row['case']+f"_N{row['N']}";folder=args.output/name
        if (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text())
            assert report['source_sha256']==source and len(report['runs'])==args.repeats and report['inputs_unchanged']
        else:
            if folder.exists():raise RuntimeError(f'Incomplete prior run: {folder}')
            report=run(SimpleNamespace(geometry=ROOT/row['geometry'],material=args.material,
                       output=folder,repeats=args.repeats),dict(defaults=({},nullcontext)))
            report['fiber_metadata']=row;report['fiber_case']=cases[row['case']]
            details=[]
            for run_row in report['runs']:
                info=json.loads((folder/f"defaults_{run_row['repetition']}.json").read_text())
                loads=info['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
                assert info['linear_fft_autotune'] and info['linear_energy_update'] and info['linear_final_audit_reuse']
                assert len(loads)==6 and all(x['true_norm_res_rel']<=1e-5 for x in loads)
                C=np.asarray(run_row['Ceff']);assert np.linalg.eigvalsh((C+C.T)/2)[0]>0
                details.append(dict(repetition=run_row['repetition'],
                    coordinate_iterations=sum(x['coordinate_iterations'] for x in loads),
                    correction_iterations=sum(x['correction_iterations'] for x in loads),
                    solve_s=info['cufhom_core_application_timing']['primaldual']['primal']['load_solve_s'],
                    postprocess_s=info['cufhom_core_application_timing']['primaldual']['primal']['postprocess_s'],
                    load_infos=loads))
            report['details']=details
            (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        campaign['completed'].append(name);path.write_text(json.dumps(campaign,indent=2)+'\n')
        print(f"CURVE {index+1}/{len(ordered)} {name}: {report['median_s']['defaults']:.6f}s; "
              f"{report['runs'][0]['cg']} CG; voxel Vf={row['Vf_voxel']:.5f}; d={row['voxels_per_diameter']:.2f} vox",flush=True)
        cp.cuda.get_current_stream().synchronize();cp.get_default_memory_pool().free_all_blocks()
        assert source=={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*.py')}


if __name__=='__main__':main()
