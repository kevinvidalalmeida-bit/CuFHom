from contextlib import nullcontext
from pathlib import Path
import json
from profile_g09_linear import parser,run,sha256
from cufhom_blockcg import trial
args=parser().parse_args();records=[]
report=run(args,variants={'current':({'linear_residual_smoothing':False},nullcontext),'block':({'linear_residual_smoothing':False},lambda:trial(records))})
report['block_solves']=records
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_blockcg.py'))
report['note']='Block cg field in runner = six times block iterations plus any physical cleanup. Compare actual block operator_calls and setup, not that count alone.'
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
