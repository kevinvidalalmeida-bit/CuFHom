from pathlib import Path
import json
from unittest.mock import patch
import argparse
from profile_linear_smoothing import run
from cufhom.pipeline import linear_coordinates as lc
from cufhom_single_reduction import cg
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--size',type=int,default=64);p.add_argument('--repeats',type=int,default=9)
a=p.parse_args();old=lc.fused_cg
# False is current production; True selects the new recurrence with the SAME smoothing.
def choose(*args,**kwargs):
 if kwargs.pop('residual_smoothing'):
  return cg(*args,**kwargs,residual_smoothing=True)
 return old(*args,**kwargs,residual_smoothing=True)
with patch.object(lc,'fused_cg',choose):run(a)
path=a.output/'report.json';r=json.loads(path.read_text());r['variant_note']='smoothing=False denotes current classical CG WITH smoothing; smoothing=True denotes single-reduction CG WITH smoothing. Only temporary adapter selects the recurrence.'
path.write_text(json.dumps(r,indent=2)+'\n')
