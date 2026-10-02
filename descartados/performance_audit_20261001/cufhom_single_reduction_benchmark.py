from contextlib import nullcontext
from pathlib import Path
import json
from profile_g09_linear import parser,run,sha256
from cufhom_single_reduction import trial
args=parser().parse_args()
report=run(args,variants={'current':({},nullcontext),'single':({},trial)})
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_single_reduction.py'))
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
