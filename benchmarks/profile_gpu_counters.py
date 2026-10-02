"""Collect one warm launch per major kernel variant with an elevated ncu.

Profiler wall times are deliberately excluded from performance baselines.
Kernel replay and cache control settings are recorded for interpretation.
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys


def export_csv(ncu, report):
    """Export an existing capture atomically; imports do not require root."""
    temporary=report.with_suffix('.csv.tmp')
    with temporary.open('w') as out,report.with_suffix('.export.log').open('w') as err:
        result=subprocess.run([str(ncu),'--import',str(report),'--page','raw','--csv',
            '--print-units','base'],stdout=out,stderr=err)
    if result.returncode==0:
        temporary.replace(report.with_suffix('.csv'))
        return True
    temporary.unlink(missing_ok=True)
    return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ncu',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',help='Restrict collection to named cases')
    p.add_argument('--kernel',action='append',help='Restrict collection to named kernel tags')
    p.add_argument('--export-only',action='store_true',help='Read existing reports without collecting new counters')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    with gzip.open(root/'benchmarks/performance_audit_20261001/evidence.json.gz','rt') as f:
        evidence=json.load(f)
    cases=[]
    for name in ('G08','G09'):
        hashes=evidence[name+'/report.json']['input_sha256']
        phase=next(Path(path) for path in hashes if path.endswith('/phase.npy'))
        material=next(Path(path) for path in hashes if path.endswith('/material.json'))
        cases.append((name,['--geometry',str(phase.parent),'--material',str(material)]))
    cases.extend((('N64',['--size','64']),('N256',['--size','256'])))
    geometry=root/'benchmarks/results/performance_audit_20261001/fiber_geometries/sfrp_vf15_ar10/N192'
    material=root/'benchmarks/performance_audit_20261001/material.json'
    for policy in ('auto','component'):
        cases.append(('fiber192_'+policy,['--geometry',str(geometry),'--material',str(material),'--fft-policy',policy]))
    kernels=[('regular_fft',r'^regular_fft(<|$)',3,None),
        ('r2c',r'^(regular_fft_r2c|vector_fft_r2c|vector_fft_symm_r2c)(<|$)',3,None),
        ('c2r',r'^(regular_fft_c2r|vector_fft_c2r|vector_fft_symm_c2r)(<|$)',3,None),
        ('material',r'^indexed_sym21_apply$',3,None),('coordinates',r'^coordinates$',3,None),
        ('adjoint_dot',r'^adjoint_dot$',3,None),('update',r'^update_rr$',3,None),
        ('update_late',r'^update_rr$',41,None),('warp_dot',r'^warp_dot$',3,None),
        ('postprocess_dot',r'^warp_dot$',3,'postprocess/')]
    # Late/normal labels are hypotheses; inspect launch register metadata to
    # identify the actual fused variant. The two binaries share a kernel name.
    sections=['LaunchStats','Occupancy','SpeedOfLight','ComputeWorkloadAnalysis',
        'MemoryWorkloadAnalysis','InstructionStats','SchedulerStats','WarpStateStats','SourceCounters']
    if a.case:
        unknown=set(a.case)-{name for name,_ in cases}
        if unknown:p.error('Unknown cases: '+str(sorted(unknown)))
        cases=[row for row in cases if row[0] in a.case]
    if a.kernel:
        unknown=set(a.kernel)-{row[0] for row in kernels}
        if unknown:p.error('Unknown kernels: '+str(sorted(unknown)))
        kernels=[row for row in kernels if row[0] in a.kernel]
    manifest_path=a.output/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        scope='One warm sampled launch per kernel/variant, FP32 fields/FFTs and FP64 scalars. '
        'Original six-load physical audit retained. Profiler duration is not benchmark time.',
        replay_mode='kernel',cache_control='none',clock_control='none',sections=sections,
        source_commit='90d49d88638d357b946c5ed4f318ef338143c37c',jobs=[])
    jobs={(r['case'],r['kernel']):r for r in manifest['jobs']}
    total=len(cases)*len(kernels);index=0;export_failures=[]
    for case,flags in cases:
        for kind,regex,skip,nvtx in kernels:
            index+=1;stem=a.output/(case+'_'+kind)
            cmd=[str(a.ncu),'--profile-from-start','off','--target-processes','all',
                '--replay-mode','kernel','--cache-control','none','--clock-control','none',
                '--kernel-name-base','function','--kernel-name','regex:'+regex,
                '--launch-skip',str(skip),'--launch-count','1','--export',str(stem)]
            for section in sections:cmd+=['--section',section]
            if nvtx:cmd+=['--nvtx','--nvtx-include',nvtx]
            cmd+=[sys.executable,str(root/'benchmarks/audit_linear_performance.py'),
                '--output',str(a.output/('solve_'+case+'_'+kind)),'--trace',*flags]
            print(f'{index}/{total}: {case}, {kind}',flush=True)
            report=stem.with_suffix('.ncu-rep')
            if report.exists():
                print('Reusing existing capture',flush=True)
                row=jobs.get((case,kind),dict(case=case,kernel=kind,report_exists=True))
                row['capture_reused']=True
                row['csv_export_ok']=export_csv(a.ncu,report)
                jobs[(case,kind)]=row
                manifest['jobs']=list(jobs.values())
                manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
                if not row['csv_export_ok']:export_failures.append(str(report))
                continue
            if a.export_only:
                print('No existing capture for '+kind,flush=True)
                continue
            solve_dir=a.output/('solve_'+case+'_'+kind)
            if solve_dir.exists():
                # Preserve an interrupted attempt and choose a fresh harness directory.
                from datetime import datetime,timezone
                fresh=solve_dir.with_name(solve_dir.name+'_retry_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
                cmd[cmd.index('--output')+1]=str(fresh)
            with stem.with_suffix('.log').open('w') as f:
                result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
            log=stem.with_suffix('.log').read_text();report=stem.with_suffix('.ncu-rep')
            if 'ERR_NVGPUCTRPERM' in log:
                raise RuntimeError('Administrator execution still cannot access counters. See '+str(stem.with_suffix('.log')))
            row=dict(case=case,kernel=kind,command=cmd,returncode=result.returncode,
                report_exists=report.exists(),log=str(stem.with_suffix('.log')))
            jobs[(case,kind)]=row
            manifest['jobs']=list(jobs.values())
            manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            if result.returncode!=0:
                print('Profiling failed; details: '+row['log'],flush=True)
                print('\n'.join(log.splitlines()[-12:]),flush=True)
                raise RuntimeError('Nsight Compute failed')
            if report.exists():
                row['csv_export_ok']=export_csv(a.ncu,report)
                if not row['csv_export_ok']:
                    export_failures.append(str(report))
                    print('CSV export failed; capture retained. Continuing collection.',flush=True)
                manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
            else:
                print('No matching launch; recorded as missing: '+kind,flush=True)
    if export_failures:
        print('Captures retained; failed CSV exports: '+str(export_failures),flush=True)
        raise SystemExit(2)


if __name__=='__main__':main()
