"""Compare final defaults to the exact previous rotation and spectrum policy."""
import argparse
import ast
from contextlib import contextmanager,nullcontext,redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASELINE='9300f9302064e3b4feb88a63eb8b84cd36beb65b'
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'benchmarks/linear_strategy_20261002'))
from profile_g09_linear import run
from audit_linear_performance import inputs
from validate_fields import compare
from cufhom.pipeline import fft_solver as fs
from unittest.mock import patch

OLD_SOURCE=subprocess.check_output(['git','show',f'{BASELINE}:src/cufhom/pipeline/fft_solver.py'],
                                  cwd=ROOT,text=True)
node=next(n for n in ast.parse(OLD_SOURCE).body if isinstance(n,ast.FunctionDef) and n.name=='_build_cfield_gpu')
OLD_FUNCTION='\n'.join(OLD_SOURCE.splitlines()[node.lineno-1:node.end_lineno])+'\n'
namespace=dict(vars(fs))
exec(compile(OLD_FUNCTION,'<previous material builder>','exec'),namespace)


@contextmanager
def previous_rotation():
    with patch.object(fs,'_build_cfield_gpu',namespace['_build_cfield_gpu']):yield


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['G08','G09','N64','N256','F192','R128','N8'],required=True)
    p.add_argument('--mode',choices=['timing','fields'],default='timing')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.output.exists():p.error('Use a new output directory')
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    if args.case=='N8':
        if args.mode!='fields':p.error('N8 is a CPU field control only')
        geometry=Path('.');phase,ori,material,hashes=inputs(8,None,None,'compact')
    else:
        source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)
        prior=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
        geometry=Path(prior['paths']['phase']).parent;material_path=Path(prior['paths']['material'])
    variants=dict(before=(dict(linear_final_spectrum_reuse=False),previous_rotation),defaults=({},nullcontext))
    if args.mode=='timing':
        report=run(SimpleNamespace(geometry=geometry,material=material_path,
                   output=args.output,repeats=args.repeats),variants)
        for row in report['runs']+report['warmups']:
            d=json.loads((args.output/f"{row['variant']}_{row['repetition']}.json").read_text())
            infos=d['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            assert sum(i['linear_final_spectrum_reuse'] for i in infos)==(6 if row['variant']=='defaults' else 0)
        report['baseline_commit']=BASELINE
    else:
        args.output.mkdir(parents=True)
        if args.case!='N8':phase,ori,material,hashes=inputs(0,geometry,material_path)
        if args.case=='N8':
            variants=dict(cpu=(dict(fft_backend='scipy',solver_real_dtype='float64',
                linear_formulation='strain',solver_rtol=1e-10),nullcontext),**variants)
        common=dict(input_dir=str(geometry),phase_array=phase,ori_array=ori,
            compiled_cfield_geometry=fs.compile_cfield_geometry(phase,ori),solver_rtol=1e-5,
            solver_atol=0.,solver_maxiter=1000,require_convergence=True,solver_verbose=False,
            solution_field_return_in_memory=True,fft_backend='cupy',**material)
        report=dict(case=args.case,shape=list(phase.shape),input_sha256=hashes,
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT/'src').rglob('*.py')},rows=[])
        base_C=base_fields=None
        for name,(settings,context) in variants.items():
            options=dict(common,**settings,solver_timing_path=str(args.output/f'{name}.json'))
            with context(),open(os.devnull,'w') as sink,redirect_stdout(sink):
                C=fs.solve_homogenization(options)
            fields=options.pop('_solution_fields_result')
            if base_C is None:base_C,base_fields=C,fields
            detail=json.loads((args.output/f'{name}.json').read_text())
            infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            error=float(np.linalg.norm(C-base_C)/np.linalg.norm(base_C));metrics=[compare(x,y) for x,y in zip(fields,base_fields)]
            residual=max(i.get('true_norm_res_rel',i['final_norm_res_rel']) for i in infos)
            eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0])
            assert len(fields)==len(infos)==6 and error<2e-6 and residual<=1e-5 and eigenvalue>0
            assert max(m['relative_l2'] for m in metrics)<2e-4
            if name!='cpu':assert sum(i['linear_final_spectrum_reuse'] for i in infos)==(6 if name=='defaults' else 0)
            report['rows'].append(dict(variant=name,Ceff_relative_error=error,Ceff=C.tolist(),
                fields=metrics,fields_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in fields],
                max_physical_residual=residual,min_Ceff_eigenvalue=eigenvalue,physical_infos=infos))
            print(name,error,max(m['relative_l2'] for m in metrics),residual,flush=True)
            if fields is not base_fields:del fields
    report['baseline_rotation_source']=OLD_FUNCTION
    report['baseline_rotation_sha256']=hashlib.sha256(OLD_FUNCTION.encode()).hexdigest()
    report['pilot_source_sha256']={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
