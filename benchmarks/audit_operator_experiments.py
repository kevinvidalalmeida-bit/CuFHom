"""Isolated fixed-input operator experiments, not alternative homogenization solvers."""
import argparse,json,os
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import cupy as cp
from cupyx.scipy.fft import get_fft_plan
from audit_linear_performance import inputs
from audit_gpu_calibration import timed
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from cufhom.pipeline import linear_coordinates as lc
from cufhom_core.tensors import fft
from cufhom_core.tensors.objects import indexed_sym21_array,_get_cupy_indexed_sym21_kernel

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,default=64)
    p.add_argument('--geometry',type=Path);p.add_argument('--material',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rhs',type=int,default=6);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output file')
    phase,ori,material,hashes=inputs(a.size,a.geometry,a.material)
    common=dict(input_dir=str(a.geometry or '.'),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),fft_backend='cupy',
        solver_rtol=1e-5,solver_atol=0.,solver_maxiter=1000,require_convergence=True,solver_verbose=False,
        **{k:material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')})
    records={};original=lc.fused_cg
    def cg(action,rhs,dot,*args,**kw):
        result=original(action,rhs,dot,*args,**kw)
        if records:return result
        captures=dict(zip(action.__code__.co_freevars,(c.cell_contents for c in action.__closure__)))
        Q=captures['Q'];A=captures['A'];base=getattr(A,'op',A)
        # The production material operator is a Tensor or a TimedOperation around it.
        descriptor=base
        value=Q.strain(rhs);reference=action(rhs)
        def error(out,ref):
            x,y=out.ravel(),ref.ravel();num=den=0.
            for i in range(0,x.size,1024*1024):
                num+=float(cp.sum(cp.abs(x[i:i+1024*1024]-y[i:i+1024*1024])**2,dtype=cp.float64))
                den+=float(cp.sum(cp.abs(y[i:i+1024*1024])**2,dtype=cp.float64))
            return float(np.sqrt(num/den))
        records['plain_H']=timed(lambda:action(rhs),batch=12)
        # A frozen H contains no host scalar reduction. Captured outputs and
        # intermediate allocations stay valid while the graph lives.
        stream=cp.cuda.get_current_stream();stream.synchronize()
        capture_stream=cp.cuda.Stream(non_blocking=True)
        with capture_stream:
            for _ in range(3):action(rhs)
            capture_stream.synchronize();capture_stream.begin_capture()
            graph_out=action(rhs);graph=capture_stream.end_capture()
        graph.launch();stream.synchronize();err=error(graph_out,reference)
        assert err<1e-6
        records['graph_H']=timed(lambda:graph.launch(),batch=12)
        records['graph_H']['relative_error']=err
        records['graph_H']['scope']='Frozen plain H only; excludes both scalar decisions and CG recurrence.'
        # FFT plan/buffer test uses exactly the same transform. Persistent
        # buffers remove CuPy object/pool overhead; they do not remove cuFFT work.
        hshape=phase.shape[:-1]+(phase.shape[-1]//2+1,)
        out=cp.empty((6,)+hshape,cp.complex64)
        plan=get_fft_plan(value,axes=(1,2,3),value_type='R2C')
        def raw_fft():plan.fft(value,out,cp.cuda.cufft.CUFFT_FORWARD)
        raw_fft();ref=fft.rfftn_cached(value,phase.shape,(1,2,3));stream.synchronize()
        err=error(out,ref);assert err<1e-6
        records['fft_allocating']=timed(lambda:fft.rfftn_cached(value,phase.shape,(1,2,3)),batch=12)
        records['fft_persistent_full_batch']=timed(raw_fft,batch=12)
        records['fft_persistent_full_batch']['relative_error']=err
        # Six identical-shape independent RHS; same material/orientation table.
        # Layout is component, RHS, spatial -- production already supports it.
        batch=a.rhs;xb=cp.stack([value*(j+1)/batch for j in range(batch)],axis=1)
        singles=[cp.ascontiguousarray(xb[:,j]) for j in range(batch)]
        Y=cp.empty_like(xb);Ys=[cp.empty_like(v) for v in singles]
        ids=descriptor.val;table=descriptor.material_table
        material_kernel=_get_cupy_indexed_sym21_kernel();spatial=phase.size
        def apply(x,y,b):
            total=b*spatial
            material_kernel(((total+255)//256,),(256,),
                (ids,table,x,y,np.int64(b),np.int64(spatial),np.int64(total)))
        def seq_material():
            for x,y in zip(singles,Ys):apply(x,y,1)
        def bat_material():apply(xb,Y,batch)
        seq_material();bat_material();stream.synchronize()
        err=error(Y,cp.stack(Ys,axis=1));assert err<1e-6
        records['material_sequential']=timed(seq_material,batch=12)
        records['material_batched']=timed(bat_material,batch=12)
        records['material_batched']['relative_error']=err
        # A material+forward FFT subchain. No scalar synchronization or CG.
        HB=cp.empty((6,batch)+hshape,cp.complex64);hs=[cp.empty((6,)+hshape,cp.complex64) for j in range(batch)]
        bp=get_fft_plan(Y,axes=(2,3,4),value_type='R2C')
        plans=[get_fft_plan(y,axes=(1,2,3),value_type='R2C') for y in Ys]
        def sequential():
            for x,y,h,pl in zip(singles,Ys,hs,plans):
                apply(x,y,1);pl.fft(y,h,cp.cuda.cufft.CUFFT_FORWARD)
        def batched():apply(xb,Y,batch);bp.fft(Y,HB,cp.cuda.cufft.CUFFT_FORWARD)
        sequential();batched();stream.synchronize();err=error(HB,cp.stack(hs,axis=1));assert err<1e-6
        records['material_fft_sequential']=timed(sequential,batch=6)
        records['material_fft_batched']=timed(batched,batch=6)
        records['material_fft_batched']['relative_error']=err
        # Separate plans AND workspaces per stream; dependencies completed before timing.
        streams=[cp.cuda.Stream(non_blocking=True) for _ in range(batch)]
        def parallel():
            for x,y,h,pl,s in zip(singles,Ys,hs,plans,streams):
                with s:apply(x,y,1);pl.fft(y,h,cp.cuda.cufft.CUFFT_FORWARD)
            for s in streams:s.synchronize()
        parallel();err=error(cp.stack(hs,axis=1),HB);assert err<1e-6
        records['material_fft_streams']=timed(parallel,batch=6)
        records['material_fft_streams']['relative_error']=err
        records['material_fft_streams']['timing']='Use wall time: null-stream events do not cover independent streams.'
        records['memory_active_bytes']=cp.get_default_memory_pool().used_bytes()
        return result
    with open(os.devnull,'w') as f,redirect_stdout(f):ref=solve_homogenization(common)
    with patch.object(lc,'fused_cg',cg),open(os.devnull,'w') as f,redirect_stdout(f):C=solve_homogenization(common)
    err=float(np.linalg.norm(C-ref)/np.linalg.norm(ref));assert err<1e-6
    report=dict(shape=list(phase.shape),rhs_count=a.rhs,scope='Fixed input operator/subchain only. No estimated six-load solver speedup '
        'is claimed from these timings. FP32 FFT/fields; compare frozen outputs with production. '
        'Streams are independent only in the microbenchmark. Persistent full FFT may differ from auto batching on G09.',
        records=records,Ceff_relative_error=err,input_sha256=hashes)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v['median_wall_s'] for k,v in records.items() if isinstance(v,dict) and 'median_wall_s' in v},indent=2))
if __name__=='__main__':main()
