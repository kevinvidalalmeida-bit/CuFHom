"""Warm allocation accounting and coordinate-H counts outside timing runs."""
import argparse
from contextlib import redirect_stdout
import gzip,json,hashlib,os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import cupy as cp

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from audit_linear_performance import inputs
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from cufhom.pipeline import linear_coordinates as lc
from cufhom.pipeline import linear_energy as energy


class AllocationAccount(cp.cuda.MemoryHook):
    def __init__(self):
        self.initial=self.active=cp.get_default_memory_pool().used_bytes()
        self.peak=self.active;self.device=cp.cuda.runtime.getDevice();self.allocations=0
    def malloc_postprocess(self,**kw):
        if kw['device_id']==self.device and kw['pmem_id']:
            self.active+=kw['mem_size'];self.peak=max(self.peak,self.active);self.allocations+=1
    def free_postprocess(self,**kw):
        if kw['device_id']==self.device:self.active-=kw['mem_size']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new output directory')
    a.output.mkdir(parents=True)
    if a.case in ('G08','G09'):
        with gzip.open(ROOT/'benchmarks/performance_audit_20261001/evidence.json.gz','rt') as f:e=json.load(f)
        paths=e[a.case+'/report.json']['input_sha256']
        geometry=next(Path(f).parent for f in paths if f.endswith('/phase.npy'))
        material=next(Path(f) for f in paths if f.endswith('/material.json'))
        phase,ori,mat,hashes=inputs(0,geometry,material)
    else:
        geometry=Path('.');phase,ori,mat,hashes=inputs(int(a.case[1:]),None,None)
    common=dict(input_dir=str(geometry),phase_array=phase,ori_array=ori,
        compiled_cfield_geometry=compile_cfield_geometry(phase,ori),fft_backend='cupy',solver_verbose=False,
        solver_rtol=1e-5,solver_atol=0.,solver_maxiter=1000,require_convergence=True,**mat)
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(a.output/'fft_cache.json')
    rows=[];original=lc.fused_cg
    for name,settings in dict(current=dict(linear_energy_update=False,linear_fft_autotune=False),
                              energy=dict(linear_energy_update=True,linear_fft_autotune=False),
                              combined=dict(linear_energy_update=True,linear_fft_autotune=True)).items():
        options=dict(common,**settings,solver_timing_path=str(a.output/f'{name}.json'))
        with open(os.devnull,'w') as sink,redirect_stdout(sink):solve_homogenization(options)
        counts=dict(plain_H=0,fused_H=0)
        def cg(action,*args,**kw):
            def plain(v):counts['plain_H']+=1;return action(v)
            step=kw.get('action_and_dot')
            if step is not None:
                def fused(v,**k):counts['fused_H']+=1;return step(v,**k)
                kw['action_and_dot']=fused
            return original(plain,*args,**kw)
        hook=AllocationAccount()
        with hook,patch.object(lc,'fused_cg',cg),open(os.devnull,'w') as sink,redirect_stdout(sink):C=solve_homogenization(options)
        cp.cuda.get_current_stream().synchronize()
        final=cp.get_default_memory_pool().used_bytes()
        assert final==hook.active, (final,hook.active)
        detail=json.loads((a.output/f'{name}.json').read_text())
        infos=detail['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert len(infos)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
        rows.append(dict(variant=name,initial_pool_bytes=hook.initial,peak_active_pool_bytes=hook.peak,
            final_pool_bytes=final,pool_allocations=hook.allocations,coordinate_H_calls=counts,
            iterations=sum(i['cg_iterations'] for i in infos),physical_infos=infos,Ceff=C.tolist()))
    report=dict(shape=list(phase.shape),input_sha256=hashes,rows=rows,
        compiled_energy_kernel_attributes={name:kernel.attributes for name,kernel in energy._KERNELS.items()},
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'src').rglob('*.py')},
        scope='Warm full solves; allocation hooks count active CuPy device-pool bytes and check final accounting. '
        'Excludes allocations outside the CuPy pool, desktop and driver contexts. Coordinate H counts include CG '
        'and its true residual, excluding physical Afun audits, RHS and cold tuning. Hooked wall times are not performance results.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print([(r['variant'],r['peak_active_pool_bytes'],r['coordinate_H_calls']) for r in rows])


if __name__=='__main__':main()
