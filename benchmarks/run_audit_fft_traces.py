"""Profile one short-fiber case before and after the existing FFT policy switch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from audit_trace_summary import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True);a=p.parse_args()
    case='sfrp_vf15_ar10_N192';geometry=a.input/'fiber_geometries/sfrp_vf15_ar10/N192'
    records={}
    for policy in ('auto','component'):
        name=case if policy=='auto' else 'component_'+case
        prefix=a.input/f'trace_{name}';sqlite=prefix.with_suffix('.sqlite')
        # Dots in case names are avoided here; suffix replaces only .nsys-rep.
        if not sqlite.exists():
            with (a.input/'logs'/f'trace_{name}.log').open('w') as f:
                subprocess.run(['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                    '--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true',
                    '--output='+str(prefix),sys.executable,str(Path(__file__).with_name('audit_linear_performance.py')),
                    '--geometry',str(geometry),'--material',str(a.material),
                    '--output',str(a.input/f'trace_case_{name}'),'--trace','--fft-policy',policy],
                    stdout=f,stderr=subprocess.STDOUT,check=True)
                subprocess.run(['nsys','export','--type=sqlite','--force-overwrite=true','--output='+str(sqlite),
                    str(prefix)+'.nsys-rep'],stdout=f,stderr=subprocess.STDOUT,check=True)
        records[name]=summarize(sqlite)
        print(name,records[name]['wall_ms'],flush=True)
    (a.input/'fft_trace_summary.json').write_text(json.dumps(records,indent=2)+'\n')
    original=json.loads((a.input/'trace_summary.json').read_text())
    original[case]=records[case]
    (a.input/'trace_summary.json').write_text(json.dumps(original,indent=2)+'\n')


if __name__=='__main__':main()
