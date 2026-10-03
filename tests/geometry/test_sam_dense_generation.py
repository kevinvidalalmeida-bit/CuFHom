"""Independent geometry, optimizer and checkpoint checks for dense straight SAM."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from cufhom.geometry.contacts import normalized_coordinates, coordinate_gradient, minimize_a4
from cufhom.geometry.contacts import a4_objective, relax_a4
from cufhom.geometry.audit import audit_periodic_fibers
from cufhom.geometry.voxel import fiber_volume, rasterize_cylinders
from cufhom.geometry.straight import Settings, generate_staged, perturb_trial

ROOT = Path(__file__).resolve().parents[2]


def test_failed_relaxation_preserves_energy_progress_when_worst_contact_grows():
    from types import SimpleNamespace
    c = np.zeros((6,3))
    u = np.tile([1.,0.,0.],(6,1))
    def evaluate(c,u):
        residual = c[:,0]-1.
        gc = np.zeros_like(c)
        gc[:,0] = residual
        return .5*np.sum(residual**2),gc,np.zeros_like(u),.1+.1*c[:,0].mean(),.01
    def numerical_stop(fun,x0,**kwargs):
        x = x0.reshape(-1,6).copy()
        x[:,0] = .5
        x = x.ravel()
        fun(x)
        kwargs['callback'](x)
        return SimpleNamespace(x=x,message='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH')
    with patch('cufhom.geometry.contacts.minimize',numerical_stop):
        result,_,info = minimize_a4(c,u,evaluate,1.,np.eye(3)/3,maxiter=19,
            overlap_tolerance=1e-6,a2_tolerance=.035,a4_tolerance=1e-7)
    assert np.all(result[:,0] > .5)
    assert info['returned_state'] == 'lowest_energy_infeasible'
    assert info['recoverable_stop'] and info['final_gradient_inf_norm'] > 0
    assert info['fallback_step'] == 'full_gradient'
    assert evaluate(result,u)[0] < evaluate(c,u)[0]


def test_center_fallback_can_descend_at_an_angular_contact_ridge():
    from types import SimpleNamespace
    c=np.zeros((6,3))
    u=np.tile([1.,0.,0.],(6,1))
    def evaluate(c,u):
        gc=np.zeros_like(c)
        gc[:,0]=c[:,0]-1.
        gu=np.zeros_like(u)
        gu[:,1]=10.  # a valid extremal subgradient of 10*abs(u_y) at zero
        energy=.5*np.sum((c[:,0]-1.)**2)+10*np.sum(np.abs(u[:,1]))
        return energy,gc,gu,.2,.01
    def stopped(fun,x0,**kwargs):
        fun(x0)
        kwargs['callback'](x0)
        return SimpleNamespace(x=x0,message='CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH')
    with patch('cufhom.geometry.contacts.minimize',stopped):
        result,axes,info=minimize_a4(c,u,evaluate,1.,np.eye(3)/3,maxiter=19,
            overlap_tolerance=1e-6,a2_tolerance=.035,a4_tolerance=1e-7)
    assert info['fallback_step']=='centers' and info['recoverable_stop']
    assert evaluate(result,axes)[0]<evaluate(c,u)[0]
    np.testing.assert_array_equal(axes,u)


def test_numerical_stop_reuses_remaining_budget_and_current_trial():
    settings = Settings(box=12.,length=2.,target_count=8,initial_count=8,candidates=1,
                        relax_iterations=19,max_relax_iterations=19,attempts=1,max_stages=1)
    calls=[]
    def relax(f,*args,**kwargs):
        calls.append((kwargs['maxiter'],f.copy()))
        candidate=f.copy()
        candidate[:,0]+=.1
        return candidate,dict(iterations=3 if len(calls)==1 else 16,evaluations=20,
                              recoverable_stop=len(calls)==1,optimizer_success=len(calls)==2,
                              history=[dict(overlap_relative=1.,A2_error=0.,energy=e)
                                       for e in (1.,.9999)])
    with patch('cufhom.geometry.straight.relax_contact_energy',relax), \
         patch('cufhom.geometry.straight.certificate',return_value={'accepted':True}):
        _,info=generate_staged(settings,np.eye(3)/3)
    assert [budget for budget,_ in calls]==[19,16]
    np.testing.assert_allclose(calls[1][1][:,0],calls[0][1][:,0]+.1)
    assert next(e for e in info['events'] if e['event']=='commit')['optimizer_iterations']==19


def test_block_audit_matches_exhaustive_images_and_ties():
    from tests.geometry.reference import geometry_reference as reference
    rng = np.random.default_rng(492)
    for box in (np.array([4., 5., 6.]), np.full(3, 16.)):
        u = rng.normal(size=(19, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
        f = np.c_[rng.uniform(-8, 12, (19, 3)), u, rng.uniform(1, 13, 19), rng.uniform(.3, 1.2, 19)]
        f[1] = f[0]  # equal maximal penetrations in different blocks
        expected = reference.audit_periodic_fibers(f, box, 1e-6)
        for block in (1, 7, 83, 65536):
            with patch('numpy.triu_indices', side_effect=AssertionError('unbounded allocation')):
                actual = audit_periodic_fibers(f, box, 1e-6, block_pairs=block)
            assert actual == expected


def test_inertia_coordinate_gradient_including_normalization_and_gap():
    rng = np.random.default_rng(768)
    x = np.c_[rng.uniform(0, 5, (9, 3)), 3*rng.normal(size=(9, 3))].ravel()
    def evaluate(x):
        c, u, norms = normalized_coordinates(x)
        e, gc, gu, _, _ = a4_objective(c, u, 4., np.full(3, 5.), contact_diameter=1.1)
        return e, coordinate_gradient(gc, gu, u, norms)
    _, gradient = evaluate(x)
    for k in range(len(x)):
        plus, minus = x.copy(), x.copy()
        plus[k] += 1e-6; minus[k] -= 1e-6
        fd = (evaluate(plus)[0]-evaluate(minus)[0])/2e-6
        np.testing.assert_allclose(gradient[k], fd, atol=3e-8, rtol=3e-5)


def test_lbfgs_feasibility_and_input_preservation():
    rng = np.random.default_rng(983)
    u = rng.normal(size=(24,3)); u /= np.linalg.norm(u,axis=1)[:,None]
    f = np.c_[rng.uniform(0, 10, (24,3)),u,np.full(24,3.),np.ones(24)]
    original = f.copy()
    result, info = relax_a4(f,10,np.eye(3)/3,maxiter=1000,
                                        overlap_tolerance=1e-6,min_gap=.02)
    assert info['contact_A2_A4_accepted']
    assert info['gap_audit']['accepted']
    np.testing.assert_array_equal(f,original)
    np.testing.assert_array_equal(result[:,6:],f[:,6:])
    np.testing.assert_allclose(np.linalg.norm(result[:,3:6],axis=1),1.,rtol=0,atol=1e-14)
    assert np.all(np.diff([r['energy'] for r in info['history']]) <= 1e-12)


def test_cylinder_raster_matches_independent_periodic_membership():
    from tests.geometry.reference.raster_reference import flat_cylinder_interior
    rng = np.random.default_rng(992)
    grid = 16
    points = np.indices((grid,)*3).reshape(3,-1).T
    u = rng.normal(size=(8,3)); u /= np.linalg.norm(u,axis=1)[:,None]
    u[:3] = np.eye(3)
    c = rng.uniform(-5,20,(8,3))
    for length in (5.,23.):
        phase,ori,count,used = rasterize_cylinders(c,u,length,1.13,grid)
        expected=np.zeros(len(points),dtype=np.uint8)
        expected_u=np.zeros((len(points),3),dtype=np.float32)
        for ci,ui in zip(c,u):
            inside=flat_cylinder_interior(points,np.r_[ci,ui,length,2.26],grid)
            expected[inside]=1;expected_u[inside]=ui
        np.testing.assert_array_equal(phase.ravel(),expected)
        np.testing.assert_array_equal(ori.reshape(-1,3),expected_u)
        assert count==expected.sum() and used==len(c)
        moved=rasterize_cylinders(c+[32,-16,48],-u,length,1.13,grid)
        np.testing.assert_array_equal(moved[0],phase)


def test_cylinder_volume_and_removed_end_caps():
    assert fiber_volume(30,1,'capsule')/fiber_volume(30,1,'cylinder') == pytest.approx(1+2/90)
    phase,*_=rasterize_cylinders(np.array([[8.,8.,8.]]),np.array([[1.,0.,0.]]),4,2,16)
    assert phase[8,8,8] and not phase[11,8,8] and not phase[10,8,8]


def test_resumed_generation_is_identical_to_uninterrupted():
    s=Settings(box=12,length=3,target_count=36,initial_count=12,batch_size=8,
               candidates=4,solver='a4',orientation_weight=1,
               overlap_tolerance=1e-6,relax_iterations=1000,adaptive_batches=True,max_stages=20)
    full,full_info=generate_staged(s,np.eye(3)/3)
    partial,info=generate_staged(replace(s,max_stages=2),np.eye(3)/3)
    assert len(partial)<len(full)
    resumed,end=generate_staged(s,np.eye(3)/3,initial=partial,resume_state=info['resume_state'])
    assert full_info['status']==end['status']=='target_count_reached'
    np.testing.assert_array_equal(resumed,full)


def test_adaptive_budget_reuses_trial_without_resampling():
    s=Settings(box=12,length=2,target_count=8,initial_count=8,candidates=1,
               relax_iterations=10,max_relax_iterations=30,attempts=1,max_stages=1)
    calls=[]
    def relax(f,*args,**kwargs):
        calls.append(f.copy())
        returned=f.copy();returned[:,0]+=0.1
        return returned,dict(iterations=kwargs['maxiter'],evaluations=kwargs['maxiter']+1,
                             optimizer_success=len(calls)==2,
                             history=[dict(overlap_relative=v,A2_error=0.)
                                      for v in (1.,.1,.0999,.0998,.0997,.0996)])
    with patch('cufhom.geometry.straight.relax_contact_energy',relax), \
         patch('cufhom.geometry.straight.certificate',return_value={'accepted':True}):
        _,info=generate_staged(s,np.eye(3)/3)
    assert len(calls)==2
    np.testing.assert_allclose(calls[1][:,0],calls[0][:,0]+.1)
    commit=next(e for e in info['events'] if e['event']=='commit')
    assert commit['optimizer_iterations']==30


def test_collective_retry_preserves_axes_dimensions_and_accepted_state():
    s = Settings(box=12,length=2,target_count=8,initial_count=8,candidates=2,
                 solver='a4',retry_shake=.05,
                 relax_iterations=10,attempts=2,max_stages=1)
    calls = []
    def relax(f,*args,**kwargs):
        calls.append(f.copy())
        candidate = f.copy()
        if len(calls) == 1:
            candidate[-1,0] = -1
        return candidate,dict(iterations=1,evaluations=1)
    def certify(f,*args,**kwargs):
        return dict(accepted=bool(np.all(f[:,0] >= 0)))
    with patch('cufhom.geometry.contacts.relax_a4',relax), \
         patch('cufhom.geometry.straight.certificate',certify):
        final,info = generate_staged(s,np.eye(3)/3)
    assert info['status'] == 'target_count_reached'
    assert [e['attempt'] for e in info['events'] if e['event'] == 'shake_trial'] == [1]
    assert len(calls) == 2 and not np.array_equal(calls[0],calls[1])
    np.testing.assert_array_equal(final[:,6:],calls[0][:,6:])
    np.testing.assert_allclose(np.linalg.norm(final[:,3:6],axis=1),1.,atol=1e-14)
    original = calls[0].copy()
    first = perturb_trial(original,s,np.random.default_rng(79))
    second = perturb_trial(original,s,np.random.default_rng(79))
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(original,calls[0])


def test_resume_after_failed_batch_keeps_reduced_schedule_and_rng():
    initial = np.array([[2.,2.,2.,1.,0.,0.,2.,1.], [2.,5.,2.,1.,0.,0.,2.,1.]])
    s = Settings(box=12,length=2,target_count=6,batch_size=4,candidates=2,
                 adaptive_batches=True,relax_iterations=10,max_stages=10)
    def relax(f,*args,**kwargs):
        candidate = f.copy()
        if len(f) == 6:
            candidate[-1,0] = -1  # deterministic failed proposal
        return candidate,dict(iterations=1,evaluations=1)
    def certify(f,*args,**kwargs):
        return dict(accepted=bool(np.all(f[:,0] >= 0)))
    with patch('cufhom.geometry.straight.relax_contact_energy',relax), \
         patch('cufhom.geometry.straight.certificate',certify):
        full,end = generate_staged(s,np.diag([1.,0.,0.]),initial=initial)
        partial,info = generate_staged(replace(s,max_stages=1),np.diag([1.,0.,0.]),initial=initial)
        assert info['resume_state']['batch_size'] == 2
        resumed,last = generate_staged(s,np.diag([1.,0.,0.]),initial=partial,
                                       resume_state=info['resume_state'])
    assert end['status'] == last['status'] == 'stalled'
    np.testing.assert_array_equal(full,resumed)


@pytest.mark.parametrize('interrupt_at', ['relax', 'rollback'])
def test_wall_budget_preserves_next_trial_and_rng(interrupt_at):
    initial = np.array([[2.,2.,2.,1.,0.,0.,2.,1.], [2.,5.,2.,1.,0.,0.,2.,1.]])
    s = Settings(box=12,length=2,target_count=3,batch_size=1,candidates=2,
                 relax_iterations=10,attempts=3,max_stages=3)
    clock = [0.]
    proposals = []
    def relax(f,*args,**kwargs):
        proposals.append(f.copy())
        candidate = f.copy()
        if interrupt_at == 'relax':
            clock[0] = 2.
            kwargs['progress'](dict(iteration=0))
        else:
            candidate[-1,0] = -1
        return candidate,dict(iterations=1,evaluations=1)
    def certify(f,*args,**kwargs):
        return dict(accepted=bool(np.all(f[:,0] >= 0)))
    def report(event):
        if event['event'] == 'rollback':
            clock[0] = 2.
    with patch('cufhom.geometry.straight.relax_contact_energy',relax), \
         patch('cufhom.geometry.straight.certificate',certify), \
         patch('cufhom.geometry.straight.time.perf_counter',lambda: clock[0]):
        partial,info = generate_staged(s,np.diag([1.,0.,0.]),initial=initial,
                                       progress=report,wall_seconds=1.)
    assert info['status'] == 'wall_budget_exhausted'
    np.testing.assert_array_equal(partial,initial)
    assert info['resume_state']['next_attempt'] == (interrupt_at == 'rollback')
    def succeed(f,*args,**kwargs):
        return f.copy(),dict(iterations=1,evaluations=1)
    with patch('cufhom.geometry.straight.relax_contact_energy',succeed), \
         patch('cufhom.geometry.straight.certificate',certify):
        final,_ = generate_staged(s,np.diag([1.,0.,0.]),initial=partial,
                                  resume_state=info['resume_state'])
    if interrupt_at == 'relax':
        np.testing.assert_array_equal(final,proposals[0])
    else:
        # A completed failed trial must consume its draws; do not repeat it.
        assert not np.array_equal(final,proposals[0])


def test_scalar_contacts_match_frozen_reference():
    from cufhom.geometry.contacts import contact_energy
    from tests.geometry.reference import contact_reference as reference
    rng = np.random.default_rng(419)
    c = rng.uniform(-4,8,(17,3))
    u = rng.normal(size=c.shape); u /= np.linalg.norm(u,axis=1)[:,None]
    c[1] = c[0]; u[1] = u[0]  # coincident parallel axes
    u[2] = u[0]+1e-9; u[2] /= np.linalg.norm(u[2])  # nearly parallel
    c[4] = c[3]; u[3:5] = np.eye(3)[:2]  # perpendicular intersecting axes
    for length in (2.,9.):
        for first_new,segments in ((0,1),(11,1),(0,3)):
            args = (c,u,length,1.001,np.array([4.,5.,6.]))
            kwargs = dict(first_new=first_new,segments_per_fiber=segments)
            expected = reference.contact_energy(*args,**kwargs)
            actual = contact_energy(*args,**kwargs)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b,rtol=2e-10,atol=2e-10)


def test_cli_cylinder_resume_preserves_shape_and_validates_dimensions():
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        def invoke(output,*args):
            return subprocess.run([sys.executable,'-m','cufhom.cli.geometry','--output',str(output),*args],
                                  cwd=ROOT,capture_output=True,text=True,timeout=45)
        first=invoke(temp/'first','--box','12','--ar','2','--vf','.02','--shape','cylinder',
                     '--a2','1','0','--grid','48','--max-stages','1','--initial-count','12',
                     '--geometry-only','--threads','1')
        assert first.returncode==2,first.stdout+first.stderr
        source=temp/'first/last_geometrically_valid.npz'
        resumed=invoke(temp/'resumed','--resume',str(source),'--max-stages','40','--threads','1')
        assert resumed.returncode==0,resumed.stdout+resumed.stderr
        info=json.loads((temp/'resumed/summary.json').read_text())
        assert info['accepted'] and info['shape']=='cylinder' and info['resume_rng_restored']
        assert info['nominal_vf']==info['nominal_cylinder_vf']
        with np.load(temp/'resumed/geometry.npz') as f: assert str(f['shape'])=='cylinder'
        invalid=invoke(temp/'invalid','--resume',str(source),'--ar','3')
        assert invalid.returncode!=0 and not (temp/'invalid').exists()
