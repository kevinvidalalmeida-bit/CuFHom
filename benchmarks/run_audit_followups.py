"""Sequential GPU experiments after the statistical fiber baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--material',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    archive=root/'descartados/performance_audit_20261001'
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(root/'benchmarks'),str(archive))),OPENBLAS_NUM_THREADS='1')
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    jobs=[]
    for case,n,reps in (('sfrp_vf25_ar12.5',256,7),('sfrp_vf05_ar10',128,9),('sfrp_vf05_ar10',256,7)):
        jobs.append((f'dual_{case}_N{n}',archive/'cufhom_dual_metric_benchmark.py',case,n,
            ['--repeats',str(reps)]))
    for n in (96,128,192,256):
        jobs.append((f'fft_sfrp_vf15_ar10_N{n}',root/'benchmarks/audit_fft_policy.py',
            'sfrp_vf15_ar10',n,['--repeats','7']))
    for case,n in (('sfrp_vf25_ar12.5',256),('sfrp_vf05_ar10',128),('sfrp_vf05_ar10',256)):
        jobs.append((f'fields_dual_{case}_N{n}',root/'benchmarks/audit_variant_fields.py',case,n,
            ['--variants','dual']))
    # Validate both policies at the largest newly examined power-of-two grid.
    jobs.append(('fields_fft_sfrp_vf15_ar10_N256',root/'benchmarks/audit_variant_fields.py',
        'sfrp_vf15_ar10',256,['--variants','fft_full','fft_component']))
    for n in (128,192):
        jobs.append((f'fields_fft_sfrp_vf15_ar10_N{n}',root/'benchmarks/audit_variant_fields.py',
            'sfrp_vf15_ar10',n,['--variants','fft_component']))
    for case,n in [('sfrp_vf15_ar10',n) for n in (32,48,64,96,128,192,256)]+[
            ('sfrp_vf05_ar10',256),('sfrp_vf25_ar12.5',256)]:
        jobs.append((f'iteration_{case}_N{n}',root/'benchmarks/audit_iteration_statistics.py',case,n,
            ['--repeats','9']))
    jobs.append(('accuracy_dual_sfrp_vf05_ar10_N256',root/'benchmarks/audit_dual_accuracy.py',
        'sfrp_vf05_ar10',256,['--repeats','7']))
    jobs.append(('fields_accuracy_dual_sfrp_vf05_ar10_N256',root/'benchmarks/audit_variant_fields.py',
        'sfrp_vf05_ar10',256,['--variants','dual_strict']))
    for name,script,case,n,flags in jobs:
        out=a.output/name
        if (out/'report.json').exists():continue
        geometry=a.output/'fiber_geometries'/case/f'N{n}'
        with (logs/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,str(script),'--geometry',str(geometry),
                '--material',str(a.material),'--output',str(out),*flags],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        r=json.loads((out/'report.json').read_text())
        summary=r.get('median_s') or r.get('statistics_s')
        if summary is None:
            summary=[(x['variant'],x['Ceff_relative_error'],max(y['relative_l2'] for y in x['fields'])) for x in r['rows']]
        print(name,summary,flush=True)


if __name__=='__main__':main()
