from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
from profile_g09_linear import parser,run,sha256
from cufhom_twolevel import trial

args=parser().parse_args()
shape=np.load(args.geometry/'phase.npy',mmap_mode='r').shape
records=[]
variants={'current':({'linear_residual_smoothing':False},nullcontext),
    'block18':({'linear_residual_smoothing':False},lambda:trial(shape,1,'block',records)),
    'balanced18':({'linear_residual_smoothing':False},lambda:trial(shape,1,'balanced',records))}
report=run(args,variants=variants)
report['coarse_setups']=records
report['candidate_sha256']=sha256(Path(__file__).with_name('cufhom_twolevel.py'))
(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
