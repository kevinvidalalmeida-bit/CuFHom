"""Run each fiber/grid audit in a fresh process; GPU jobs remain sequential."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--material',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=9);a=p.parse_args()
    m=json.loads(a.prepared.read_text());a.output.mkdir(parents=True,exist_ok=True)
    logs=a.output/'logs';logs.mkdir(exist_ok=True);script=Path(__file__).with_name('audit_linear_performance.py')
    for ix,row in enumerate(m['rasters']):
        name=row['case']+f'_N{row["N"]}';out=a.output/name
        if (out/'report.json').exists():continue
        if out.exists():raise RuntimeError(f'Incomplete old result directory: {out}')
        with (logs/(name+'.log')).open('w') as f:
            subprocess.run([sys.executable,str(script),'--geometry',row['geometry'],'--material',str(a.material),
                '--output',str(out),'--repeats',str(a.repeats)],stdout=f,stderr=subprocess.STDOUT,check=True)
        report=json.loads((out/'report.json').read_text());report['fiber_metadata']=row
        report['fiber_case']=next(c for c in m['cases'] if c['case']==row['case'])
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{ix+1}/{len(m["rasters"])} {name}: {report["statistics_s"]["median"]:.6f}s, '
              f'{report["runs"][0]["iterations"]} CG, Vf={row["Vf_voxel"]:.5f}',flush=True)
if __name__=='__main__':main()
