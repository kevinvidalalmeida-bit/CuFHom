"""Trace the current defaults after FFT/energy integration; no solver patches."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from audit_trace_summary import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['G08','G09','N64','F192'],default=['G08','G09','N64','F192'])
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    nsys=shutil.which('nsys')
    if nsys is None:raise RuntimeError('Nsight Systems is unavailable')
    env=dict(os.environ,CUFHOM_FFT_TUNING_CACHE=str(args.output.resolve()/'fft_cache.json'))
    records={}
    for case in args.cases:
        prefix=args.output/f'trace_{case}'
        command=[nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                 '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                 '--output='+str(prefix),sys.executable,str(ROOT/'benchmarks/audit_linear_performance.py'),
                 '--output',str(args.output/case),'--trace']
        if case=='N64':command+=['--size','64','--family','compact']
        else:
            source=('final_'+case if case=='G08' else 'verified_'+case)
            prior=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
            command+=['--geometry',str(Path(prior['paths']['phase']).parent),
                      '--material',prior['paths']['material']]
        print('Capturando',case,flush=True)
        with (args.output/f'{case}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
            sqlite=prefix.with_suffix('.sqlite')
            subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=true',
                            '--output='+str(sqlite),str(prefix)+'.nsys-rep'],
                           stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
        summary=summarize(sqlite,dict(material_energy='operator_other',adjoint_update='operator_update_fused'))
        report=json.loads((args.output/case/'report.json').read_text())
        assert report['inputs_unchanged'] and len(report['diagnostic']['physical_infos'])==6
        records[case]=dict(trace=summary,solve=report)
        print(case,round(summary['wall_ms'],3),'ms',flush=True)
    snapshot={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'src').rglob('*.py')}
    sources={str(p.relative_to(ROOT)):p.read_text() for p in
             [Path(__file__),ROOT/'benchmarks/audit_linear_performance.py',ROOT/'benchmarks/audit_trace_summary.py']}
    bundle=dict(schema=1,baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                scope='One profiled warm solve per case; separate from uninstrumented timing statistics.',
                nsys_version=subprocess.check_output([nsys,'--version'],text=True).strip(),
                cases=records,source_sha256=snapshot,trace_sources=sources)
    data=(json.dumps(bundle,indent=2,allow_nan=False)+'\n').encode()
    (args.output/'trace_evidence.json.gz').write_bytes(gzip.compress(data,mtime=0))
    (args.output/'summary.json').write_text(json.dumps(bundle,indent=2)+'\n')


if __name__=='__main__':main()
