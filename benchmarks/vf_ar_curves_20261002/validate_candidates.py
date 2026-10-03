"""Full six-field controls, outside timing samples, including CPU64 on 8³."""
import argparse
from contextlib import nullcontext,redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'benchmarks/linear_strategy_20261002'))
from audit_linear_performance import inputs
from validate_fields import compare
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from candidates import rotation_context,spectrum_context,combined_context,LAST


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['G08','G09','N64','N256','F192','R128','N8'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    args.output.mkdir(parents=True)
    if args.case=='N8':
        geometry=Path('.')
        phase,ori,mat,hashes=inputs(8,None,None,'compact')
    else:
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
    variants=dict(current=({},nullcontext),spectrum=({},spectrum_context),
                  rotation=({},rotation_context),combined=({},combined_context))
    if args.case=='N8':
        variants=dict(cpu=(dict(fft_backend='scipy',solver_real_dtype='float64',
            linear_formulation='strain',solver_rtol=1e-10),nullcontext),**variants)
    report=dict(case=args.case,shape=list(phase.shape),input_sha256=hashes,
        scope='Six full strain fields; physical residual 1e-5; N8 CPU64 reference at 1e-10. Not timed.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (ROOT/'src').rglob('*.py')},
        pilot_source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__),Path(__file__).with_name('candidates.py'))},rows=[])
    base_C=base_fields=None
    for name,(settings,context) in variants.items():
        options=dict(common,**settings,solver_timing_path=str(args.output/f'{name}.json'))
        LAST.update(spectrum_reuses=0,rotation_groups=0)
        with context(),open(os.devnull,'w') as sink,redirect_stdout(sink):
            C=solve_homogenization(options)
        fields=options.pop('_solution_fields_result')
        if base_C is None:base_C,base_fields=C,fields
        detail=json.loads((args.output/f'{name}.json').read_text())
        infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert len(infos)==len(fields)==6
        error=float(np.linalg.norm(C-base_C)/np.linalg.norm(base_C))
        metrics=[compare(x,y) for x,y in zip(fields,base_fields)]
        residual=max(i.get('true_norm_res_rel',i['final_norm_res_rel']) for i in infos)
        eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0])
        assert error<2e-6 and residual<=1e-5 and eigenvalue>0
        assert max(m['relative_l2'] for m in metrics)<2e-4
        if name in ('spectrum','combined'):assert LAST['spectrum_reuses']==6
        if name in ('rotation','combined'):assert LAST['rotation_groups']>0
        report['rows'].append(dict(variant=name,Ceff=C.tolist(),Ceff_relative_error=error,
            fields=metrics,fields_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in fields],
            max_physical_residual=residual,min_Ceff_eigenvalue=eigenvalue,physical_infos=infos,
            activations=dict(LAST),iterations=sum(i.get('cg_iterations',i.get('kit',0)) for i in infos)))
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,error,max(m['relative_l2'] for m in metrics),residual,dict(LAST),flush=True)
        if fields is not base_fields:del fields


if __name__=='__main__':main()
