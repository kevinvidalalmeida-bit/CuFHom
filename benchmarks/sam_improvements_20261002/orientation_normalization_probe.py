"""Compare printed-energy A4 weight with the eq. (9) vector-field weight.

The 2017 article's two expressions differ by N when A4 is an average. This
pilot cannot establish which expression was used by its original C++ code.
"""
import json
from pathlib import Path
import time

import numpy as np
from cufhom.pipeline.sam_paper_reference import relax_paper_reference

HERE=Path(__file__).resolve().parent
with np.load(HERE/'diagnostic_lbfgs_paper_trial.npz',allow_pickle=False) as saved:
    f=saved['proposed_fibers'].copy()
    a2,a4=saved['A2'].copy(),saved['A4'].copy()
results=[]
for label,weight in [('printed_energy',1/len(f)),('restore_vector_field',1.)]:
    started=time.perf_counter()
    f,info=relax_paper_reference(f,70.,a2,target_a4=a4,optimizer='lbfgs',maxiter=20000,
                               min_gap=.001,orientation_weight=weight,
                               overlap_tolerance=1e-6,a4_tolerance=1e-7)
    row=dict(label=label,orientation_weight=weight,wall_seconds=time.perf_counter()-started,
             optimization=info)
    results.append(row)
    (HERE/'orientation_normalization_probe.json').write_text(json.dumps(results,indent=2)+'\n')
    np.savez_compressed(HERE/(label+'_trial_1986.npz'),fibers=f,diagnostic_only=True)
    print(json.dumps(dict(label=label,weight=weight,iterations=info['iterations'],
                         accepted=info['contact_A2_A4_accepted'],A4_error=info['A4_error'],
                         gap_violation=info['gap_audit']['maximum_relative_overlap'],
                         first_energy=info['history'][0]['energy'],last_energy=info['history'][-1]['energy'],
                         fallback_step=info['fallback_step'],message=info['optimizer_message'])),flush=True)
