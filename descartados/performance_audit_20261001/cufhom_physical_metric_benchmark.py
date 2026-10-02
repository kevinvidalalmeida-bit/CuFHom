from contextlib import nullcontext
from pathlib import Path
import json
from profile_g09_linear import parser,run,sha256
from cufhom_physical_metric import trial
args=parser().parse_args()
report=run(args,variants={'current':({},nullcontext),'physical_metric':({},trial)})
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_physical_metric.py'))
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
