"""Full field validation of isolated methodological pilots, outside all timings."""
import argparse,json,os
from pathlib import Path
from contextlib import redirect_stdout,nullcontext
import numpy as np
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from audit_linear_performance import inputs
from cufhom_single_reduction import trial as single
from cufhom_dual_metric import trial as dual
from cufhom_dual_metric_strict import trial as dual_strict
from cufhom_physical_metric import trial as metric
from cufhom_acoustic_local import trial as acoustic

def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variants',nargs='+',default=['single','dual','metric','acoustic','blend']);a=p.parse_args()
    if a.output.exists():raise ValueError('Use new output directory')
    a.output.mkdir(parents=True)
    phase,ori,material,hashes=inputs(0,a.geometry,a.material)
    common=dict(input_dir=str(a.geometry),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),fft_backend='cupy',
        solver_rtol=1e-5,solver_atol=0.,solver_maxiter=1500,require_convergence=True,
        solver_verbose=False,solution_field_return_in_memory=True,
        **{k:material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')})
    variants=dict(current=nullcontext,single=single,dual=dual,dual_strict=dual_strict,metric=metric,
                  fft_full=nullcontext,fft_component=nullcontext,
                  acoustic=lambda:acoustic([],weight=1.),blend=lambda:acoustic([],weight=.25))
    refC=refFields=None;rows=[]
    for name in ['current']+a.variants:
        settings=dict(common,solver_timing_path=str(a.output/f'{name}.json'))
        if name.startswith('fft_'):settings['cupy_rfft_batch_mode']=name[4:]
        with variants[name](),open(os.devnull,'w') as f,redirect_stdout(f):C=solve_homogenization(settings)
        fields=settings.pop('_solution_fields_result')
        infos=json.loads((a.output/f'{name}.json').read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        if refC is None:refC,refFields=C,fields
        comparisons=[]
        for load,(field,reference) in enumerate(zip(fields,refFields)):
            # FP64 accumulation in chunks; no full FP64 copy of 2 GB fields.
            x,y=field.ravel(),reference.ravel();sq=den=0.;maxdiff=maxref=0.
            for i in range(0,x.size,1024*1024):
                v=x[i:i+1024*1024].astype(np.float64);r=y[i:i+1024*1024].astype(np.float64)
                sq+=np.dot(v-r,v-r);den+=np.dot(r,r)
                maxdiff=max(maxdiff,float(np.max(np.abs(v-r))));maxref=max(maxref,float(np.max(np.abs(r))))
            comparisons.append(dict(load=load,relative_l2=float(np.sqrt(sq/den)),
                absolute_l2=float(np.sqrt(sq)),absolute_linf=maxdiff,relative_linf=maxdiff/maxref))
        relative=float(np.linalg.norm(C-refC)/np.linalg.norm(refC))
        assert relative<1e-6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
        rows.append(dict(variant=name,fields=comparisons,Ceff_absolute_error=float(np.linalg.norm(C-refC)),
            Ceff_relative_error=relative,iterations=sum(i['cg_iterations'] for i in infos),
            max_physical_residual=max(i['true_norm_res_rel'] for i in infos),physical_infos=infos,
            min_Ceff_eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0]),
            symmetry_relative=float(np.linalg.norm(C-C.T)/np.linalg.norm(C))))
        print(name,relative,max(r['relative_l2'] for r in comparisons),flush=True)
        (a.output/'report.json').write_text(json.dumps(dict(shape=list(phase.shape),rows=rows,
            input_sha256=hashes,scope='All six complete fluctuation strain fields; validation outside timings. '
            'maxiter=1500 for validation only, to cover the already rejected acoustic pilot; stopping tolerance unchanged.'),indent=2)+'\n')
        if name!='current':del fields
if __name__=='__main__':main()
