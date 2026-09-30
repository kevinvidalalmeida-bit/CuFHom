from contextlib import nullcontext
from pathlib import Path
import json
from profile_g09_linear import parser,run,sha256
from cufhom_cr import trial
args=parser().parse_args()
report=run(args,variants={'current':({'linear_residual_smoothing':False},nullcontext),'cr':({},trial)})
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_cr.py'))
report['note']='For CR, runner cg is total CR iterations plus any original-equation CG cleanup.'
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
