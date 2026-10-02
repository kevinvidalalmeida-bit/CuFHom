"""Full six-load timings for final audit reuse, against the same code disabled."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from profile_g09_linear import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['G08','G09','N64','N256','R128','F192'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)
    previous=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    variants=dict(current=(dict(linear_final_audit_reuse=False),nullcontext),
                  reuse=(dict(linear_final_audit_reuse=True),nullcontext))
    report=run(SimpleNamespace(geometry=Path(previous['paths']['phase']).parent,
        material=Path(previous['paths']['material']),output=args.output,repeats=args.repeats),variants)
    for row in report['runs']:
        details=json.loads((args.output/f"{row['variant']}_{row['repetition']}.json").read_text())
        infos=details['cufhom_core_application_timing']['primaldual']['primal']['load_solver_infos']
        assert all(i['linear_final_audit_reuse']==(row['variant']=='reuse') for i in infos)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
