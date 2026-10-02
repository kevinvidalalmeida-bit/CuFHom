"""Profile actual CG spectrum, operator throughput and GPU memory; no solver changes."""
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import argparse,json,os,time
import numpy as np
import cupy as cp
from cufhom.pipeline.fft_solver import solve_homogenization,compile_cfield_geometry
from cufhom.pipeline import linear_coordinates as lc,linear_krylov as lk
from profile_g09_linear import sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--geometry',type=Path,required=True);p.add_argument('--material',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);args=p.parse_args()
if args.output.exists():raise ValueError('Use a new output directory')
args.output.mkdir(parents=True)
files=[args.geometry/'phase.npy',args.geometry/'ori.npy',args.material];hashes=[sha256(f) for f in files]
phase,ori=(np.load(f) for f in files[:2]);phase.flags.writeable=ori.flags.writeable=False
material=json.loads(args.material.read_text())
common=dict(input_dir=str(args.geometry),phase_array=phase,ori_array=ori,
 compiled_cfield_geometry=compile_cfield_geometry(phase,ori),fft_backend='cupy',
 solver_rtol=1e-5,solver_atol=0,solver_maxiter=1000,solver_verbose=False,
 **{k:material[k] for k in ('Em','nu_m','Ef_L','Ef_T','nu_LT','nu_TT','G_LT')})
with open(os.devnull,'w') as sink,redirect_stdout(sink):ref=solve_homogenization(common)
records=[];original_cg=lc.fused_cg;original_update=lk.fused_update;operator_profile={}
def cg(action,rhs,dot,*a,**kw):
 record=dict(alpha=[],beta=[],coordinate_size=rhs.size,coordinate_bytes=rhs.nbytes,
  reduction_count=0,plain_actions=0,fused_actions=0,peak_pool_active_bytes=0,minimum_driver_free_bytes=10**30)
 previous=[None]
 def sampled():
  record['peak_pool_active_bytes']=max(record['peak_pool_active_bytes'],cp.get_default_memory_pool().used_bytes())
  record['minimum_driver_free_bytes']=min(record['minimum_driver_free_bytes'],cp.cuda.runtime.memGetInfo()[0])
 def observed_dot(x,y):
  value=dot(x,y);record['reduction_count']+=1
  if previous[0] is None:previous[0]=value
  return value
 def observed_action(x):record['plain_actions']+=1;return action(x)
 fused=kw.get('action_and_dot')
 if fused:
  def observed_fused(*a,**k):record['fused_actions']+=1;return fused(*a,**k)
  kw['action_and_dot']=observed_fused
 def update(*v):
  value=original_update(*v);record['alpha'].append(float(v[2]));record['beta'].append(float(value/previous[0]));previous[0]=value
  sampled();return value
 sampled();start=time.perf_counter();event0=cp.cuda.Event();event1=cp.cuda.Event();event0.record()
 with patch.object(lk,'fused_update',update):result,it=original_cg(observed_action,rhs,observed_dot,*a,**kw)
 event1.record();event1.synchronize();record['wall_s_instrumented']=time.perf_counter()-start
 record['cuda_elapsed_s_instrumented']=cp.cuda.get_elapsed_time(event0,event1)/1000
 record['iterations']=it;record['no_restarts']=record['plain_actions']==1
 if record['alpha'] and record['no_restarts']:
  alpha=np.array(record['alpha']);beta=np.array(record['beta']);diag=1/alpha;diag[1:]+=beta[:-1]/alpha[:-1]
  off=np.sqrt(beta[:-1])/alpha[:-1];ritz=np.linalg.eigvalsh(np.diag(diag)+np.diag(off,1)+np.diag(off,-1))
  record.update(ritz=ritz.tolist(),ritz_ratio=float(ritz[-1]/ritz[0]))
 records.append(record)
 if not operator_profile:
  # Plain coordinate H, preallocated input, warm plan. Event batch includes
  # stream idle gaps from host launch code; it is NOT isolated kernel time.
  for _ in range(3):out=action(rhs)
  cp.cuda.get_current_stream().synchronize()
  batches=[]
  for _ in range(5):
   e0=cp.cuda.Event();e1=cp.cuda.Event();e0.record();start=time.perf_counter()
   for j in range(20):out=action(rhs)
   e1.record();e1.synchronize();batches.append(dict(wall_s=(time.perf_counter()-start)/20,cuda_stream_elapsed_s=cp.cuda.get_elapsed_time(e0,e1)/20000))
  operator_profile.update(batches=batches,median_wall_s=float(np.median([r['wall_s'] for r in batches])),
   median_stream_elapsed_s=float(np.median([r['cuda_stream_elapsed_s'] for r in batches])))
 return result,it
with patch.object(lc,'fused_cg',cg),open(os.devnull,'w') as sink,redirect_stdout(sink):
 result=solve_homogenization(dict(common,solver_timing_path=str(args.output/'timing.json')))
difference=float(np.linalg.norm(result-ref)/np.linalg.norm(ref))
print('Diagnostic Ceff relative difference',difference,flush=True)
assert difference<2e-7,difference
assert hashes==[sha256(f) for f in files]
infos=json.loads((args.output/'timing.json').read_text())['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
assert len(infos)==6 and all(i['true_residual_converged'] and i['true_norm_res_rel']<=1e-5 for i in infos)
root=Path(__file__).resolve().parents[1]
report=dict(shape=list(phase.shape),vf=float(phase.mean()),scope='Diagnostic only. CG memory sampled at updates; '
 'driver allocation includes CuPy pool, FFT workspaces, context and other users. CUDA events include host scheduling gaps. '
 'Ritz values are estimates, not certified bounds. No recurrence restarts in interpreted spectra.',
 Ceff_relative_difference=difference,coordinate_bytes=records[0]['coordinate_bytes'],six_strain_field_bytes=6*6*phase.size*4,
 driver_total_bytes=cp.cuda.runtime.memGetInfo()[1],operator=operator_profile,loads=records,
 physical_load_infos=infos,input_sha256=hashes,inputs_unchanged=True,paths=[str(f) for f in files],
 source_sha256={str(f.relative_to(root)):sha256(f) for f in (root/'src').rglob('*.py')},diagnostic_sha256=sha256(Path(__file__)))
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(shape=report['shape'],operator=operator_profile['median_wall_s'],
 peak_active_pool=max(r['peak_pool_active_bytes'] for r in records),
 minimum_driver_free=min(r['minimum_driver_free_bytes'] for r in records),
 ritz_ranges=[[r.get('ritz',[None])[0],r.get('ritz',[None])[-1],r.get('ritz_ratio')] for r in records]),indent=2))
