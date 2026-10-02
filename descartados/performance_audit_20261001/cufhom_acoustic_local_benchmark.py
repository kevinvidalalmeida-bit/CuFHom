from contextlib import nullcontext
from pathlib import Path
import json
from profile_g09_linear import parser,run,sha256
from cufhom_acoustic_local import trial
args=parser().parse_args();records=[]
report=run(args,variants={'current':({},nullcontext),'acoustic':({},lambda:trial(records))})
report['preconditioner_records']=records
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_acoustic_local.py'))
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
