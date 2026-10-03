"""Six-load code-review candidates; same current production source and physics."""
import argparse
from contextlib import nullcontext,contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from profile_g09_linear import run
from candidates import rotation_context,spectrum_context,combined_context,LAST


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['G08','G09','N64','N256','F192','R128'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=5)
    args=parser.parse_args()
    source=('verified_'+args.case if args.case in ('G09','F192') else 'final_'+args.case)
    prior=json.loads((ROOT/'benchmarks/results/linear_strategy_20261002'/source/'report.json').read_text())
    os.environ['CUFHOM_FFT_TUNING_CACHE']=str(args.output.resolve()/'fft_cache.json')
    activations=[]
    def observed(name,context):
        @contextmanager
        def wrapped():
            LAST.update(spectrum_reuses=0,rotation_groups=0)
            with context():yield
            activations.append(dict(variant=name,**LAST))
        return wrapped
    variants=dict(current=({},nullcontext),spectrum=({},spectrum_context),rotation=({},rotation_context),
                  combined=({},combined_context))
    variants={name:(settings,observed(name,context)) for name,(settings,context) in variants.items()}
    report=run(SimpleNamespace(geometry=Path(prior['paths']['phase']).parent,
               material=Path(prior['paths']['material']),output=args.output,repeats=args.repeats),variants)
    report['pilot_source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (Path(__file__),Path(__file__).with_name('candidates.py'))}
    report['activations']=activations
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len(activations)==4*(args.repeats+1)
    for row in activations:
        if row['variant'] in ('spectrum','combined'):assert row['spectrum_reuses']==6
        if row['variant'] in ('rotation','combined'):assert row['rotation_groups']>0
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
