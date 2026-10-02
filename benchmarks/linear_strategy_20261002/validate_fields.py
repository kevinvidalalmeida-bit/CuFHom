"""Complete six-field comparisons and small CPU64 reference, outside timings."""
import argparse
from contextlib import nullcontext, redirect_stdout
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'descartados/auxiliary_geneo_20261002'))
from audit_linear_performance import inputs
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from auxiliary_space import auxiliary_context,geneo_context,fine_geneo_context


def compare(field,reference):
    x,y=field.ravel(),reference.ravel();num=den=maximum=scale=0.
    for first in range(0,x.size,1024*1024):
        a=x[first:first+1024*1024].astype(np.float64)
        b=y[first:first+1024*1024].astype(np.float64)
        diff=a-b;num+=float(diff@diff);den+=float(b@b)
        maximum=max(maximum,float(np.max(abs(diff))));scale=max(scale,float(np.max(abs(b))))
    return dict(relative_l2=float(np.sqrt(num/den)) if den else float(np.sqrt(num)),
                absolute_linf=maximum,relative_linf=maximum/scale if scale else maximum)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--geneo-threshold',type=float,default=.15)
    p.add_argument('--variants',nargs='+',default=['energy','tuned','combined','auxiliary','geneo'])
    a=p.parse_args()
    os.environ['CUFHOM_GENEO_THRESHOLD']=str(a.geneo_threshold)
    if a.output.exists():raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True)
    if a.case in ('G08','G09'):
        with gzip.open(ROOT/'benchmarks/performance_audit_20261001/evidence.json.gz','rt') as f:e=json.load(f)
        paths=e[a.case+'/report.json']['input_sha256']
        geometry=next(Path(f).parent for f in paths if f.endswith('/phase.npy'))
        material=next(Path(f) for f in paths if f.endswith('/material.json'))
        phase,ori,mat,hashes=inputs(0,geometry,material)
    elif a.case.startswith('F'):
        geometry=ROOT/f'benchmarks/results/performance_audit_20261001/fiber_geometries/sfrp_vf15_ar10/N{a.case[1:]}'
        phase,ori,mat,hashes=inputs(0,geometry,ROOT/'benchmarks/performance_audit_20261001/material.json')
    else:
        geometry=Path('.')
        phase,ori,mat,hashes=inputs(int(a.case[1:]),None,None,'regions' if a.case[0]=='R' else 'compact')
    common=dict(input_dir=str(geometry),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),solver_rtol=1e-5,
        solver_atol=0.,solver_maxiter=1000,require_convergence=True,solver_verbose=False,
        solution_field_return_in_memory=True,linear_fft_autotune=False,linear_energy_update=False,
        solver_fft_form='r',cfield_origin='zero',**mat)
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(a.output/'fft_cache.json')
    variants=dict(current=(dict(fft_backend='cupy'),nullcontext),
        cpu=(dict(fft_backend='scipy',solver_real_dtype='float64',linear_formulation='strain'),nullcontext),
        energy=(dict(fft_backend='cupy',linear_energy_update=True),nullcontext),
        tuned=(dict(fft_backend='cupy',linear_fft_autotune=True),nullcontext),
        combined=(dict(fft_backend='cupy',linear_energy_update=True,linear_fft_autotune=True),nullcontext),
        auxiliary=(dict(fft_backend='cupy'),auxiliary_context),geneo=(dict(fft_backend='cupy'),geneo_context),
        fine_geneo=(dict(fft_backend='cupy'),fine_geneo_context))
    baselineC=baselineFields=None;rows=[]
    order=(['cpu'] if a.cpu else [])+['current']+a.variants
    report=dict(shape=list(phase.shape),input_sha256=hashes,
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src').rglob('*.py')},
        strategy_source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__),ROOT/'descartados/auxiliary_geneo_20261002/auxiliary_space.py',ROOT/'descartados/auxiliary_geneo_20261002/metric_pcg.py']},
        scope='Six complete fluctuation fields; CPU64 reference when requested; physical rtol 1e-5, maxiter 1000; no timing claim.',rows=rows)
    for name in order:
        settings,context=variants[name]
        options=dict(common,**settings,solver_timing_path=str(a.output/f'{name}.json'))
        with context(),open(os.devnull,'w') as sink,redirect_stdout(sink):C=solve_homogenization(options)
        fields=options.pop('_solution_fields_result')
        if baselineC is None:baselineC,baselineFields=C,fields
        detail=json.loads((a.output/f'{name}.json').read_text())
        infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert len(fields)==len(infos)==6
        metrics=[compare(field,reference) for field,reference in zip(fields,baselineFields)]
        error=float(np.linalg.norm(C-baselineC)/np.linalg.norm(baselineC))
        residual=max(i.get('true_norm_res_rel',i['final_norm_res_rel']) for i in infos)
        eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0])
        assert error<2e-6 and residual<=1e-5 and eigenvalue>0
        assert max(m['relative_l2'] for m in metrics)<2e-4
        if name in ('energy','combined'):assert all(i['linear_energy_update'] for i in infos)
        row=dict(variant=name,Ceff=C.tolist(),Ceff_relative_error=error,fields=metrics,
            fields_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in fields],
            max_physical_residual=residual,min_Ceff_eigenvalue=eigenvalue,
            iterations=sum(i['cg_iterations'] for i in infos),physical_infos=infos)
        rows.append(row);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,error,max(m['relative_l2'] for m in metrics),residual,flush=True)
        if fields is not baselineFields:del fields


if __name__=='__main__':main()
