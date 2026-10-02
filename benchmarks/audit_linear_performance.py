"""Reproducible audit harness; never changes the production solver defaults.

Run one grid per process. JSON timings are included in warm wall measurements.
NVTX annotations are confined to a separate, explicitly instrumented trace.
"""
import time
START = time.perf_counter()
import argparse
from contextlib import redirect_stdout, ExitStack
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline.fft_solver import solve_homogenization, compile_cfield_geometry
from cufhom.pipeline import linear_coordinates as lc, linear_krylov as lk
from cufhom_core.tensors import fft
from cufhom_core import applications as appmod
from cufhom.pipeline import fft_solver as fsol

IMPORT_S = time.perf_counter()-START

def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def inputs(n, geometry, material_path, family='compact', contrast=30.):
    if geometry:
        phase = np.load(geometry/'phase.npy')
        ori = np.load(geometry/'ori.npy')
        material = json.loads(material_path.read_text())
        paths = [geometry/'phase.npy', geometry/'ori.npy', material_path]
        hashes = {str(p): digest(p) for p in paths}
    else:
        # The same analytic, general inclusion geometry at every resolution.
        # No fiber generator, resampling, geometry-dependent solver or tolerance.
        axes = [(np.arange(n, dtype=np.float64)+.5)/n for _ in range(3)]
        x,y,z = np.meshgrid(*axes, indexing='ij', sparse=True)
        phase = np.zeros((n,n,n), np.uint8)
        ori = np.empty(phase.shape+(3,), np.float32)
        ori[:] = np.array([1.,2.,3.])/np.sqrt(14.)
        if family=='laminate':phase[:int(np.count_nonzero(axes[0]<1/3))]=1
        elif family=='regions':
            phase[(x+.6*y+.3*z) % .5 < .2]=1
            ori[n//2:]=np.array([-2.,1.,1.])/np.sqrt(6.)
        elif family=='random_inclusions':
            rng=np.random.default_rng(20261001)
            centers=rng.uniform(0,1,(12,3));radii=rng.uniform(.085,.125,12)
            directions=rng.normal(size=(12,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
            for center,radius,direction in zip(centers,radii,directions):
                dist=[np.minimum(abs(v-c),1-abs(v-c)) for v,c in zip((x,y,z),center)]
                mask=sum(v*v for v in dist)<radius*radius
                phase[mask]=1;ori[mask]=direction
        else:
            for a,b,c in ((.25,.25,.25),(.70,.60,.65)):
                phase[(x-a)**2+(y-b)**2+(z-c)**2 < .17**2]=1
        material = dict(Em=3., nu_m=.3, Ef_L=3*contrast, Ef_T=3*np.sqrt(contrast),
                        nu_LT=.25, nu_TT=.3, G_LT=1.2*np.sqrt(contrast))
        hashes = {'phase':hashlib.sha256(phase.tobytes()).hexdigest(),
                  'ori':hashlib.sha256(ori.tobytes()).hexdigest()}
    phase.flags.writeable = ori.flags.writeable = False
    return phase, ori, material, hashes

def stats(values):
    v=np.asarray(values, dtype=float)
    return dict(median=float(np.median(v)),mean=float(np.mean(v)),
                std=float(np.std(v,ddof=1)) if len(v)>1 else 0.,
                minimum=float(np.min(v)),p95=float(np.percentile(v,95)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--geometry',type=Path)
    p.add_argument('--material',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=9)
    p.add_argument('--trace',action='store_true')
    p.add_argument('--fft-policy',choices=['auto','full','component'],default='auto')
    p.add_argument('--family',choices=['compact','laminate','regions','random_inclusions'],default='compact')
    p.add_argument('--contrast',type=float,default=30.)
    a=p.parse_args()
    if a.output.exists(): p.error('Use a new output directory')
    a.output.mkdir(parents=True)
    t=time.perf_counter();cp.cuda.get_current_stream().synchronize();context_s=time.perf_counter()-t
    t=time.perf_counter();phase,ori,material,hashes=inputs(a.size,a.geometry,a.material,a.family,a.contrast)
    input_s=time.perf_counter()-t
    t=time.perf_counter();compiled=compile_cfield_geometry(phase,ori);compile_s=time.perf_counter()-t
    common=dict(input_dir=str(a.geometry or '.'),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compiled,fft_backend='cupy',solver_real_dtype='float32',
        solver_rtol=1e-5,solver_atol=0.,solver_maxiter=1000,require_convergence=True,
        solver_verbose=False,gpu_timing_sync=True,cupy_rfft_batch_mode=a.fft_policy,
        **{k:material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')})
    ref=None;records=[];cg_times=[];peak=[0];minimum_free=[cp.cuda.runtime.memGetInfo()[0]]
    def memory():
        peak[0]=max(peak[0],cp.get_default_memory_pool().used_bytes())
        minimum_free[0]=min(minimum_free[0],cp.cuda.runtime.memGetInfo()[0])
    def execute(rep):
        nonlocal ref
        path=a.output/f'timing_{rep}.json'
        cp.cuda.get_current_stream().synchronize();t=time.perf_counter()
        with open(os.devnull,'w') as f, redirect_stdout(f):
            C=solve_homogenization(dict(common,solver_timing_path=str(path)))
        cp.cuda.get_current_stream().synchronize();wall=time.perf_counter()-t
        detail=json.loads(path.read_text());app=detail['cufhom_core_application_timing']
        primal=app['primaldual']['primal'];infos=primal['load_solver_infos']
        assert len(infos)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
        if ref is None: ref=C.copy()
        absolute=float(np.linalg.norm(C-ref));relative=absolute/float(np.linalg.norm(ref))
        assert relative<1e-6
        stages=dict(input_loading=detail['input_loading_s'],
            material_setup=detail['cfield_build_s']+detail['cfield_pack_s'],
            projection=app['projection_s'],material_operator=primal['material_operator_s'],
            six_load_solves=primal['load_solve_s'],postprocess=primal['postprocess_s'],
            output=detail['problem_postprocessing_s'])
        stages['other']=wall-sum(stages.values())
        row=dict(repetition=rep,wall_s=wall,stages_s=stages,
            iterations=sum(i['cg_iterations'] for i in infos),
            coordinate_iterations=sum(i['coordinate_iterations'] for i in infos),
            correction_iterations=sum(i['correction_iterations'] for i in infos),
            physical_infos=infos,Ceff=C.tolist(),Ceff_absolute_difference=absolute,
            Ceff_relative_difference=relative,max_residual=max(i['true_norm_res_rel'] for i in infos),
            symmetry_relative=float(np.linalg.norm(C-C.T)/np.linalg.norm(C)),
            minimum_Ceff_eigenvalue=float(np.linalg.eigvalsh((C+C.T)/2)[0]))
        print(f'{phase.shape[0]} {rep}: {wall:.6f}s, {row["iterations"]} iterations',flush=True)
        return row
    first=execute('first');warm=execute('warm')
    if not a.trace:
        for rep in range(a.repeats): records.append(execute(rep))
    original=lc.fused_cg
    def annotated(name, fun):
        def wrap(*args,**kw):
            cp.cuda.nvtx.RangePush(name)
            try:return fun(*args,**kw)
            finally:cp.cuda.nvtx.RangePop()
        return wrap
    def observed_cg(action,rhs,dot,*args,**kw):
        if a.trace:
            action=annotated('operator_plain',action)
            dot=annotated('dot_norm',dot)
            if kw.get('action_and_dot'):kw['action_and_dot']=annotated('operator_fused',kw['action_and_dot'])
        cp.cuda.get_current_stream().synchronize();t=time.perf_counter();memory()
        cp.cuda.nvtx.RangePush('inner_cg')
        try:result=original(action,rhs,dot,*args,**kw)
        finally:cp.cuda.nvtx.RangePop()
        cp.cuda.get_current_stream().synchronize();cg_times.append(time.perf_counter()-t);memory()
        return result
    with ExitStack() as s:
        s.enter_context(patch.object(lc,'fused_cg',observed_cg))
        if a.trace:
            s.enter_context(patch.object(lk,'fused_update',annotated('update_reduce',lk.fused_update)))
            s.enter_context(patch.object(appmod,'postprocess',annotated('postprocess',appmod.postprocess)))
            s.enter_context(patch.object(fsol,'_build_cfield_gpu',annotated('setup_material',fsol._build_cfield_gpu)))
            for name in ('rfftn_cached','irfftn_unscaled_cached'):
                s.enter_context(patch.object(fft,name,annotated(name,getattr(fft,name))))
            cp.cuda.profiler.start()
        cp.cuda.nvtx.RangePush('TOTAL')
        diagnostic=execute('trace' if a.trace else 'diagnostic')
        cp.cuda.nvtx.RangePop()
        if a.trace:cp.cuda.profiler.stop()
    diagnostic['inner_cg_s']=sum(cg_times)
    diagnostic['rhs_coordinates_audits_s']=diagnostic['stages_s']['six_load_solves']-sum(cg_times)
    root=Path(__file__).resolve().parents[1]
    report=dict(scope='Warm six-load Ceff, compiled immutable geometry; FP32 fields, FP64 scalar reductions, '
        'rtol=1e-5,atol=0,maxiter=1000; same production defaults. No exports. Timing JSON included. '
        'Diagnostic/profile timings are excluded from statistical baseline. Memory sampled at CG boundaries.',
        shape=list(phase.shape),voxels=int(phase.size),vf=float(phase.mean()),material=material,
        stored_real_dof=6*int(phase.size),coordinate_real_slots=6*int(np.prod(phase.shape[:-1])*(phase.shape[-1]//2+1)),
        admissible_real_dof=3*(int(np.prod([n-1 if n%2==0 else n for n in phase.shape]))-1),
        analytic_family=a.family if not a.geometry else None,contrast_longitudinal=material['Ef_L']/material['Em'],
        import_s=IMPORT_S,context_s=context_s,input_generation_or_read_s=input_s,
        geometry_compile_s=compile_s,first_call=first,warmup=warm,runs=records,diagnostic=diagnostic,
        statistics_s=stats([r['wall_s'] for r in records]) if records else None,
        peak_active_pool_bytes=peak[0],reserved_pool_bytes=cp.get_default_memory_pool().total_bytes(),
        minimum_driver_free_bytes=minimum_free[0],driver_total_bytes=cp.cuda.runtime.memGetInfo()[1],
        plans_cached=len(fft._CUPY_PLAN_CACHE),input_sha256=hashes,
        inputs_unchanged=(not a.geometry or hashes=={str(f):digest(f) for f in (a.geometry/'phase.npy',a.geometry/'ori.npy',a.material)}),
        source_sha256={str(f.relative_to(root)):digest(f) for f in (root/'src').rglob('*.py')},
        benchmark_sha256=digest(Path(__file__)),cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        device=str(cp.cuda.runtime.getDeviceProperties(0)['name']),fft_batch_mode=a.fft_policy)
    assert report['inputs_unchanged']
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
