"""Fixed-distribution insertion and one-fiber reduction in the 2017 control."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from cufhom.pipeline.sam_staged_insertion import Settings, generate_staged, propose


def test_random_insertions_allow_contacts_and_use_fixed_isotropic_distribution():
    # Existing aligned fibers do not change the prescribed isotropic sampling.
    accepted = np.array([[.5,.5,.5,1.,0.,0.,2.,1.]]*6)
    settings = Settings(box=1., length=2., insertion_policy='random', candidates=100)
    original = accepted.copy()
    with patch('cufhom.pipeline.sam_staged_insertion.draw_directions',
               side_effect=AssertionError('finite ACG fit would bias isotropic draws')), \
         patch('cufhom.pipeline.sam_staged_insertion.candidate_contact_energies',
               side_effect=AssertionError('random policy must permit soft overlaps')):
        trial = propose(accepted, 4000, np.eye(3)/3, settings, np.random.default_rng(293))
    np.testing.assert_array_equal(accepted, original)
    np.testing.assert_array_equal(trial[:6], original)
    np.testing.assert_allclose(np.linalg.norm(trial[6:,3:6],axis=1), 1., atol=1e-14)
    assert np.all((trial[6:,:3] >= 0.) & (trial[6:,:3] < 1.))
    np.testing.assert_array_equal(trial[:,6:], np.tile([2.,1.],(len(trial),1)))
    # Sampling fluctuation is allowed; matching the previous fiber residual is
    # deliberately absent, and final orientation still requires certification.
    assert np.linalg.norm(trial[6:,3:6].T@trial[6:,3:6]/4000-np.eye(3)/3) < .025


def test_one_fiber_reductions_and_resume_preserve_full_trial_budget():
    initial = np.array([[2.,2.,2.,1.,0.,0.,2.,1.], [2.,5.,2.,1.,0.,0.,2.,1.]])
    settings = Settings(box=12.,length=2.,target_count=12,batch_size=6,candidates=1,
                        insertion_policy='random',batch_reduction='one',attempts=1,
                        relax_iterations=19,max_relax_iterations=19,max_stages=20)
    budgets = []
    def relax(f,*args,**kwargs):
        budgets.append(kwargs['maxiter'])
        candidate = f.copy()
        if len(f)-int(candidate[0,1]) > 2:
            # Store the old accepted count in a harmless marker for this
            # scheduler test so additions above two always fail.
            candidate[-1,0] = -1.
        else:
            candidate[0,1] = len(candidate)
        return candidate,dict(iterations=kwargs['maxiter'],evaluations=20)
    initial[0,1] = len(initial)
    def certify(f,*args,**kwargs):
        return dict(accepted=bool(np.all(f[:,0] >= 0.)))
    with patch('cufhom.pipeline.sam_staged_insertion.relax_contact_energy',relax), \
         patch('cufhom.pipeline.sam_staged_insertion.certificate',certify):
        full,info = generate_staged(settings,np.diag([1.,0.,0.]),initial=initial)
        part,partial = generate_staged(replace(settings,max_stages=1),np.diag([1.,0.,0.]),initial=initial)
        resumed,end = generate_staged(settings,np.diag([1.,0.,0.]),initial=part,
                                      resume_state=partial['resume_state'])
    assert info['status'] == end['status'] == 'target_count_reached'
    assert [e['batch'] for e in info['events'] if e['event']=='reduce_batch'] == [5,4,3,2]
    assert partial['resume_state']['batch_size'] == 5
    assert set(budgets) == {19}
    np.testing.assert_array_equal(full,resumed)


def test_cli_control_policy_is_saved_and_inherited(tmp_path):
    root = Path(__file__).resolve().parents[1]
    def invoke(output,*args):
        return subprocess.run([sys.executable,'-m','cufhom.cli.geometry','--output',str(output),*args],
                              cwd=root,capture_output=True,text=True,timeout=40)
    result = invoke(tmp_path/'first','--box','12','--ar','2','--vf','.02',
                    '--shape','cylinder','--a2','1','0','--initial-count','12',
                    '--insertion-policy','random','--batch-reduction','one',
                    '--no-adaptive-batches','--retry-shake','0','--max-stages','1',
                    '--threads','1','--geometry-only')
    assert result.returncode == 2, result.stdout+result.stderr
    result = invoke(tmp_path/'second','--resume',str(tmp_path/'first/last_geometrically_valid.npz'),
                    '--max-stages','40','--threads','1','--geometry-only')
    assert result.returncode == 0, result.stdout+result.stderr
    request = json.loads((tmp_path/'second/request.json').read_text())
    assert request['settings']['insertion_policy']=='random'
    assert request['settings']['batch_reduction']=='one'
    assert request['settings']['adaptive_batches'] is False
    summary = json.loads((tmp_path/'second/summary.json').read_text())
    assert summary['status']=='target_count_reached' and summary['geometric_certificate']['accepted']
    invalid = invoke(tmp_path/'invalid','--fiber-model','flexible','--insertion-policy','random')
    assert invalid.returncode != 0 and not (tmp_path/'invalid').exists()
