"""General A2/A4, gap and nested-neighbor tests. No dense packing campaign."""
import unittest
from unittest.mock import patch
import numpy as np
from cufhom.geometry.orientation import (acg_closure, resolve_orientation, validate_a2,
                                      validate_a4, _principal_moments)
from cufhom.geometry.contacts import fourth_moment, isotropic_a4, a4_objective, sphere_exp
from cufhom.geometry.straight import Settings, generate_staged, certificate


class OrientationTests(unittest.TestCase):
    def test_isotropic_exact(self):
        np.testing.assert_allclose(acg_closure(np.eye(3)/3), isotropic_a4(), atol=1e-15)

    def test_contraction_interior_boundary_and_near_boundary(self):
        for diagonal in ([.5,.3,.2], [.8,.15,.05], [.999,.0009,.0001],
                         [.999999,9e-7,1e-7], [.7,.3,0.], [1.,0.,0.], [.5,.5,0.]):
            with self.subTest(diagonal=diagonal):
                a2 = np.diag(diagonal)
                fourth = acg_closure(a2)
                validate_a4(fourth, a2)
                np.testing.assert_allclose(np.einsum('ijkk->ij', fourth), a2, atol=1e-10)

    def test_covariance_under_rotation_and_repeated_eigenvalues(self):
        q, _ = np.linalg.qr(np.random.default_rng(763).normal(size=(3, 3)))
        for a2 in (np.diag([.6,.2,.2]), np.diag([.7,.3,0.]), np.diag([1.,0.,0.])):
            expected = np.einsum('ia,jb,kc,ld,abcd->ijkl', q,q,q,q,acg_closure(a2))
            np.testing.assert_allclose(acg_closure(q@a2@q.T), expected, atol=1e-10)

    def test_quadrature_independent_monte_carlo(self):
        c = np.array([1.,4.,9.])
        a2, _ = _principal_moments(np.log(c))
        u = np.random.default_rng(718).normal(size=(150000,3))*np.sqrt(c)
        u /= np.linalg.norm(u, axis=1)[:,None]
        np.testing.assert_allclose(acg_closure(np.diag(a2)), fourth_moment(u), atol=.003)

    def test_cache_cannot_be_mutated(self):
        a = acg_closure(np.eye(3)/3)
        a[:] = 0.
        np.testing.assert_allclose(acg_closure(np.eye(3)/3), isotropic_a4(), atol=1e-15)

    def test_user_a4_not_replaced_by_closure(self):
        a4 = fourth_moment(np.eye(3))
        a2, chosen, metadata = resolve_orientation(a4=a4)
        np.testing.assert_array_equal(chosen, a4)
        np.testing.assert_allclose(a2, np.eye(3)/3)
        self.assertEqual(metadata['source'], 'explicit_A4')
        self.assertGreater(np.linalg.norm(chosen-acg_closure(a2)), .1)

    def test_reject_inconsistent_or_malformed_tensors(self):
        for bad in (np.eye(3), np.diag([1.1,0.,-.1]), np.full((3,3),np.nan)):
            with self.assertRaises(ValueError):
                validate_a2(bad)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            resolve_orientation(np.diag([.5,.3,.2]), isotropic_a4())
        with self.assertRaises(ValueError):
            validate_a4(np.ones((6,6)))
        bad = isotropic_a4(); bad[0,1,0,1] += .1
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            validate_a4(bad)

    def test_anisotropic_gradient_with_gap(self):
        rng = np.random.default_rng(739)
        c = rng.uniform(0,5,(9,3)); u = rng.normal(size=(9,3)); u /= np.linalg.norm(u,axis=1)[:,None]
        dc = rng.normal(size=c.shape); du = rng.normal(size=u.shape)
        du -= u*np.sum(u*du,axis=1)[:,None]
        kwargs = dict(target_a4=acg_closure(np.diag([.6,.3,.1])),contact_diameter=1.2)
        _, gc, gu, _, _ = a4_objective(c,u,3.,np.full(3,5.),**kwargs)
        h=1e-6
        fd = (a4_objective(c+h*dc,sphere_exp(u,h*du),3.,np.full(3,5.),**kwargs)[0]
              -a4_objective(c-h*dc,sphere_exp(u,-h*du),3.,np.full(3,5.),**kwargs)[0])/(2*h)
        self.assertAlmostEqual(fd,float(np.sum(gc*dc)+np.sum(gu*du)),places=7)

    def test_gap_audit_is_not_physical_overlap_tolerance(self):
        f=np.array([[2.,2.,2.,1.,0.,0.,2.,1.], [2.,3.1,2.,1.,0.,0.,2.,1.]])
        a2=np.diag([1.,0.,0.])
        self.assertTrue(certificate(f,8.,a2,1e-6,1e-6)['accepted'])
        result=certificate(f,8.,a2,1e-6,1e-6,1e-6,acg_closure(a2),min_gap=.2)
        self.assertFalse(result['accepted'])
        self.assertTrue(result['audit']['accepted'])
        self.assertFalse(result['gap_audit']['accepted'])


    def test_explicit_target_requires_a4_solver(self):
        with self.assertRaisesRegex(ValueError,'cannot be ignored'):
            generate_staged(Settings(),np.eye(3)/3,target_a4=isotropic_a4())

    def test_voxel_fourth_moment_matches_dense_and_contracts(self):
        from cufhom.geometry.voxel import voxel_a4, voxel_a2
        rng=np.random.default_rng(97)
        u=rng.normal(size=(33000,3)).astype(np.float32)
        phase=rng.integers(0,2,len(u),dtype=np.uint8)
        chosen=u[phase!=0].astype(float); chosen/=np.linalg.norm(chosen,axis=1)[:,None]
        actual=voxel_a4(phase,u)
        np.testing.assert_allclose(actual,fourth_moment(chosen),atol=2e-14)
        np.testing.assert_allclose(np.einsum('ijkk->ij',actual),voxel_a2(phase,u),atol=2e-14)
        np.testing.assert_array_equal(voxel_a4(np.zeros_like(phase),u),np.zeros((3,3,3,3)))

    def test_bootstrap_grows_only_with_contacts_satisfied_and_stays_bounded(self):
        def failed_fit(trial,*args,**kwargs):
            f=trial.copy()
            f[:,:3]=np.c_[np.arange(len(f))*2+1,np.ones(len(f)),np.ones(len(f))]
            f[:,3:6]=np.array([1.,0.,0.])
            return f,dict(iterations=1,evaluations=1)
        s=Settings(box=100,length=.5,diameter=.1,solver='a4',target_count=30,
                   initial_count=6,bootstrap_max_count=12,attempts=1,candidates=1,max_stages=10)
        with patch('cufhom.geometry.contacts.relax_a4',failed_fit):
            f, summary=generate_staged(s,np.eye(3)/3)
        self.assertEqual(len(f),0)
        self.assertEqual([e['addition'] for e in summary['events'] if e['event']=='trial_start'],[6,12])
        self.assertEqual(summary['status'],'bootstrap_failed')


if __name__ == '__main__':
    unittest.main()
