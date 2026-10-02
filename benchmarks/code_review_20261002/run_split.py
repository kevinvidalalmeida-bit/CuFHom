"""Compare the reference split with current FFT/energy defaults on six loads."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
sys.path.insert(0,str(ROOT/'descartados/reference_split_20261002'))
from profile_g09_linear import run
from reference_split import split_context,recurrence_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['G08','G09','N64','N256','R128','F192'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--rtol',type=float,default=1e-5)
    args=parser.parse_args()
    source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)
    previous=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
    geometry=Path(previous['paths']['phase']).parent;material=Path(previous['paths']['material'])
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    variants=dict(current=({},nullcontext),reference_split=({},split_context),
                  reference_split_recurrence=({},recurrence_context))
    variants={name:(dict(settings,solver_rtol=args.rtol,linear_final_audit_reuse=False),context)
              for name,(settings,context) in variants.items()}
    report=run(SimpleNamespace(geometry=geometry,material=material,output=args.output,repeats=args.repeats),variants)
    report['pilot_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__),ROOT/'descartados/reference_split_20261002/reference_split.py')}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
