"""Complete fields and audit comparison with final reuse enabled/disabled."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from audit_linear_performance import inputs
sys.path.insert(0,str(ROOT/'benchmarks/linear_strategy_20261002'))
from validate_fields import compare
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['G08','G09','N64','N256','R128','F192'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)
    prior=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
    geometry=Path(prior['paths']['phase']).parent
    phase,ori,mat,hashes=inputs(0,geometry,Path(prior['paths']['material']))
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    common=dict(input_dir=str(geometry),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),solver_rtol=1e-5,
        solver_atol=0.,solver_maxiter=1000,require_convergence=True,solver_verbose=False,
        solution_field_return_in_memory=True,solver_fft_form='r',cfield_origin='zero',
        fft_backend='cupy',**mat)
    report=dict(case=args.case,shape=list(phase.shape),input_sha256=hashes,
        scope='Six full fields; bitwise flags and numerical differences, original physical residual 1e-5; repeated disabled control for G08/N64 cleanup; no timing claim.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'src').rglob('*.py')},rows=[])
    baseline_C=baseline_fields=baseline_infos=None
    variants=[('current',False)]+([('current_repeat',False)] if args.case in ('G08','N64') else [])+[('reuse',True)]
    for name,enabled in variants:
        options=dict(common,linear_final_audit_reuse=enabled,solver_timing_path=str(args.output/f'{name}.json'))
        with open(os.devnull,'w') as sink,redirect_stdout(sink):C=solve_homogenization(options)
        fields=options.pop('_solution_fields_result')
        detail=json.loads((args.output/f'{name}.json').read_text())
        infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert len(infos)==len(fields)==6
        if baseline_C is None:baseline_C,baseline_fields,baseline_infos=C,fields,infos
        error=float(np.linalg.norm(C-baseline_C)/np.linalg.norm(baseline_C))
        metrics=[compare(x,y) for x,y in zip(fields,baseline_fields)]
        assert error<2e-6 and max(m['relative_l2'] for m in metrics)<2e-4
        for x,y in zip(infos,baseline_infos):
            assert x['cg_iterations']==y['cg_iterations']
            assert x['linear_final_audit_reuse']==enabled
        residual=max(i['true_norm_res_rel'] for i in infos)
        eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0])
        assert residual<=1e-5 and eigenvalue>0
        report['rows'].append(dict(variant=name,Ceff=C.tolist(),Ceff_bitwise_equal=bool(np.array_equal(C,baseline_C)),
            Ceff_relative_error=error,fields=metrics,
            fields_bitwise_equal=all(np.array_equal(x,y) for x,y in zip(fields,baseline_fields)),
            physical_residuals_bitwise_equal=all(x['true_norm_res_rel']==y['true_norm_res_rel'] for x,y in zip(infos,baseline_infos)),
            fields_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in fields],
            max_physical_residual=residual,min_Ceff_eigenvalue=eigenvalue,physical_infos=infos))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,'Ceff relative difference',error,'field L2',max(m['relative_l2'] for m in metrics),residual,flush=True)
        if fields is not baseline_fields:del fields


if __name__=='__main__':main()
