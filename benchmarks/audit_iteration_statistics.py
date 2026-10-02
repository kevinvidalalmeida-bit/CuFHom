"""Repeated synchronized CG timing, kept separate from the wall-time baseline."""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline.fft_solver import compile_cfield_geometry, solve_homogenization
from audit_linear_performance import inputs, digest, stats


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=9)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True)
    phase,ori,material,hashes=inputs(0,a.geometry,a.material)
    common=dict(input_dir=str(a.geometry),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),fft_backend='cupy',
        solver_rtol=1e-5,solver_atol=0.,solver_maxiter=1000,require_convergence=True,
        solver_verbose=False,gpu_timing_sync=True,
        **{k:material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')})
    with open(os.devnull,'w') as f,redirect_stdout(f):reference=solve_homogenization(common)
    original=lc.fused_cg;factory=lc.make_linear_solver
    coord=[];correction=[];rows=[]
    def time_call(fun,store,args,kwargs):
        cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
        result=fun(*args,**kwargs)
        cp.cuda.get_current_stream().synchronize();store.append(time.perf_counter()-start)
        return result
    def observed(*args,**kw):return time_call(original,coord,args,kw)
    def build(**kw):
        fallback=kw['fallback']
        def observed_fallback(*args,**kw):return time_call(fallback,correction,args,kw)
        return factory(**dict(kw,fallback=observed_fallback))
    with patch.object(lc,'fused_cg',observed),patch.object(lc,'make_linear_solver',build):
        for rep in range(a.repeats):
            coord.clear();correction.clear();timing=a.output/f'run_{rep}.json'
            with open(os.devnull,'w') as f,redirect_stdout(f):C=solve_homogenization(dict(common,solver_timing_path=str(timing)))
            infos=json.loads(timing.read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
            assert len(coord)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
            error=float(np.linalg.norm(C-reference)/np.linalg.norm(reference));assert error<1e-6
            iterations=sum(i['coordinate_iterations'] for i in infos)
            rows.append(dict(rep=rep,coordinate_cg_s=sum(coord),correction_cg_s=sum(correction),
                coordinate_cg_per_iteration_s=sum(coord)/iterations,coordinate_iterations=iterations,
                correction_iterations=sum(i['correction_iterations'] for i in infos),
                coordinate_load_s=list(coord),correction_load_s=list(correction),
                max_physical_residual=max(i['true_norm_res_rel'] for i in infos),Ceff_relative_difference=error))
    report=dict(shape=list(phase.shape),runs=rows,input_sha256=hashes,
        statistics_s={key:stats([r[key] for r in rows]) for key in
            ('coordinate_cg_s','correction_cg_s','coordinate_cg_per_iteration_s')},
        scope='Separate diagnostic, 9 warm synchronized repetitions; coordinate and fallback CG measured independently. '
            'Synchronization at six load boundaries affects diagnostic wall times; excludes geometry/setup/postprocess.',
        benchmark_sha256=digest(Path(__file__)))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(a.output.name,report['statistics_s'],flush=True)


if __name__=='__main__':main()
