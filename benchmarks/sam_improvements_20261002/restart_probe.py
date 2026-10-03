"""Check whether a numerically stopped L-BFGS trial still has useful descent."""
import json
from pathlib import Path
import time

import numpy as np
from cufhom.pipeline.sam_paper_reference import relax_paper_reference

HERE=Path(__file__).resolve().parent
with np.load(HERE/'diagnostic_lbfgs_paper_trial.npz',allow_pickle=False) as z:
    f=z['relaxed_fibers'].copy()
    a2,a4=z['A2'].copy(),z['A4'].copy()
rows=[]
for attempt in range(8):
    started=time.perf_counter()
    f,info=relax_paper_reference(f,70.,a2,target_a4=a4,optimizer='lbfgs',maxiter=20000,
                               min_gap=.001,overlap_tolerance=1e-6,a4_tolerance=1e-7)
    row=dict(attempt=attempt,iterations=info['iterations'],evaluations=info['evaluations'],
             accepted=info['contact_A2_A4_accepted'],message=info['optimizer_message'],
             gap_violation=info['gap_audit']['maximum_relative_overlap'],
             A4_error=info['A4_error'],wall_seconds=time.perf_counter()-started,
             first=info['history'][0],last=info['history'][-1])
    rows.append(row)
    (HERE/'restart_probe_after_fix.json').write_text(json.dumps(rows,indent=2)+'\n')
    np.savez_compressed(HERE/'restart_probe_after_fix.npz',fibers=f,diagnostic_only=True)
    print(json.dumps(row),flush=True)
    if row['accepted']:
        break
