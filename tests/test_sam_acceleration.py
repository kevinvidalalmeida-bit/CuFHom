"""Conservative rejection must preserve contacts, forces and angular sampling."""
import sys
from pathlib import Path
import unittest
import numpy as np
from cufhom.pipeline import sam_generator as base
from cufhom.pipeline import sam_acceleration as opt
from cufhom.pipeline import sam_acceleration_kernels as kernels
from cufhom.pipeline import sam_periodic_kernels as periodic
from cufhom.pipeline.sam_periodic_audit import segment_distance, audit_periodic_fibers


class ConservativeSAMTests(unittest.TestCase):
    def setUp(self):
        from numba import set_num_threads
        set_num_threads(2)
        rng = np.random.default_rng(873)
        self.centers = rng.random((96, 3))*16
        self.directions = rng.normal(size=(96, 3))
        self.directions /= np.linalg.norm(self.directions, axis=1)[:, None]
        # Near-parallel, antiparallel, coincident, and boundary configurations.
        self.centers[:4] = [[.001,0,0],[15.999,0,0],[8,8,8],[8,8,8]]
        self.directions[:4] = [[1,0,0],[-1,0,0],[1,0,0],[1,1e-14,0]]

    def test_candidate_contacts(self):
        for length in [3., 8., 16.]:
            args = (self.centers, self.directions, self.centers, self.directions,
                    len(self.centers), 16.,16.,16.,length,1.,(length+1.25)**2)
            # Exclude self-candidates by translating candidate centers.
            args = ((self.centers+.17)%16,)+args[1:]
            np.testing.assert_array_equal(base._check_candidates_against_existing_numba(*args),
                                          kernels._check_candidates_against_existing_numba(*args))

    def test_pair_forces_bitwise(self):
        for length in [3., 8., 16.]:
            pairs = base._build_pairs_center_cull_numba(self.centers,16.,16.,16.,(length+1.25)**2)
            results=[]
            for fn in [base._apply_pair_forces_numba, kernels._apply_pair_forces_numba]:
                updates=np.zeros_like(self.centers); rotations=np.zeros_like(updates)
                maximum=fn(pairs,self.centers,self.directions,updates,rotations,
                           16.,16.,16.,length,1.,(length+1.25)**2,.3,.001,.03,1.)
                results.append((updates,rotations,maximum))
            for a,b in zip(*results): np.testing.assert_array_equal(a,b)

    def test_per_fiber_forces_bitwise(self):
        for length in [3.,8.,16.]:
            pairs=base._build_pairs_center_cull_numba(self.centers,16.,16.,16.,(length+1.25)**2)
            offsets,neighbors=base._pairs_to_csr_numba(pairs,len(self.centers))
            results=[]
            for fn in [base._apply_forces_per_fiber_numba,kernels._apply_forces_per_fiber_numba]:
                outputs=[np.zeros_like(self.centers),np.zeros_like(self.centers),np.zeros(len(self.centers))]
                fn(self.centers,self.directions,offsets,neighbors,*outputs,
                   16.,16.,16.,length,1.,(length+1.25)**2,.3,.001,.03,1.)
                results.append(outputs)
            for a,b in zip(*results): np.testing.assert_array_equal(a,b)

    def test_angular_quadrature_and_targets(self):
        rng=np.random.default_rng(589)
        tensors=[np.eye(3)/3,np.diag([1.,0,0]),np.diag([.5,.5,0])]
        for _ in range(12):
            a=rng.normal(size=(3,3)); a=a@a.T; tensors.append(a/np.trace(a))
        for a in tensors:
            key=tuple(a.ravel())
            np.testing.assert_allclose(opt.angular_transform(key),base._angular_gaussian_transform(key),rtol=3e-13,atol=1e-14)
            np.testing.assert_allclose(opt.fused_angular_transform(key),opt.angular_transform(key),rtol=3e-13,atol=1e-14)

    def test_activation_restored_on_exception(self):
        old=base._apply_pair_forces_numba
        with self.assertRaises(RuntimeError):
            with opt.accelerated_sam():
                self.assertIs(base._apply_pair_forces_numba,kernels._apply_pair_forces_numba)
                raise RuntimeError('test')
        self.assertIs(base._apply_pair_forces_numba,old)

    def test_non_nearest_image_regression(self):
        c=np.array([[0.,0.,0.],[8.05460298, -.87213008, -.53160864]]) % 16
        u=np.array([[-.99930212,.0255531,.02724541],[-.9993822,.03226892,.0139264]])
        u/=np.linalg.norm(u,axis=1)[:,None]
        args=(*c[0],*u[0],c[1:],u[1:],1,16.,16.,16.,12.,1.,13.25**2)
        self.assertFalse(base._check_candidate_serial_numba(*args))
        self.assertTrue(periodic._check_candidate_serial_numba(*args))
        f=np.c_[c,u,np.full(2,12.),np.ones(2)]
        check=audit_periodic_fibers(f,16.,1/120)
        self.assertFalse(check['accepted'])
        self.assertGreater(check['nonnearest_violating_image_contacts'],0)

    def test_periodic_candidates_exhaustive_oracle(self):
        import itertools
        rng=np.random.default_rng(124)
        for length in [3.,8.,12.,25.]:
            c=rng.random((12,3))*16; u=rng.normal(size=(12,3));u/=np.linalg.norm(u,axis=1)[:,None]
            cc=rng.random((16,3))*16; uu=rng.normal(size=(16,3));uu/=np.linalg.norm(uu,axis=1)[:,None]
            actual=periodic._check_candidates_against_existing_numba(cc,uu,c,u,len(c),16.,16.,16.,length,1.,(length+1.25)**2)
            expected=np.zeros(len(cc),dtype=np.uint8)
            for shift in itertools.product(range(-3,4),repeat=3):
                delta=c[None,:,:]-cc[:,None,:]+16*np.array(shift)
                distance=segment_distance(delta,uu[:,None,:]*length,u[None,:,:]*length)
                expected |= np.any(distance<1.,axis=1).astype(np.uint8)
            np.testing.assert_array_equal(actual,expected)

    def test_periodic_force_matches_explicit_images(self):
        import itertools
        c=self.centers[4:20].copy();u=self.directions[4:20].copy();length=12.
        ii,jj=np.triu_indices(len(c),1);pairs=np.c_[ii,jj].astype(np.int32)
        updates=np.zeros_like(c);rotations=np.zeros_like(c)
        actual=periodic._apply_pair_forces_numba(pairs,c,u,updates,rotations,16.,16.,16.,length,1.,13.25**2,.3,.001,.03,1.)
        force=np.zeros_like(c);torque=np.zeros_like(c);maximum=0.
        for i,j in pairs:
            for shift in itertools.product([-1,0,1],repeat=3):
                centers=np.stack([c[i],c[j]+16*np.array(shift)])
                direct=np.stack([u[i],u[j]])
                f=np.zeros((2,3));t=np.zeros((2,3))
                value=base._apply_pair_forces_numba(np.array([[0,1]],dtype=np.int32),centers,direct,f,t,
                       1e6,1e6,1e6,length,1.,13.25**2,.3,.001,.03,1.)
                force[i]+=f[0];force[j]+=f[1];torque[i]+=t[0];torque[j]+=t[1];maximum=max(maximum,value)
        np.testing.assert_allclose(updates,force,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(rotations,torque,atol=1e-12,rtol=1e-12)
        self.assertAlmostEqual(actual,maximum,places=12)

    def test_audit_self_image(self):
        f=np.array([[0.,0,0,1,0,0,16.,1.]])
        self.assertFalse(audit_periodic_fibers(f,16.,1/120)['accepted'])

    def test_periodic_csr_matches_pair_forces(self):
        c=self.centers[4:].copy(); u=self.directions[4:].copy()
        for length in [3.,12.,25.]:
            pairs=base._build_pairs_center_cull_numba(c,16.,16.,16.,(length+1.25)**2)
            offsets,neighbors=base._pairs_to_csr_numba(pairs,len(c))
            f=np.zeros_like(c);t=np.zeros_like(c)
            maximum=periodic._apply_pair_forces_numba(
                pairs,c,u,f,t,16.,16.,16.,length,1.,(length+1.25)**2,.3,.001,.03,1.)
            ff=np.zeros_like(c);tt=np.zeros_like(c);mm=np.zeros(len(c))
            periodic._apply_forces_per_fiber_numba(
                c,u,offsets,neighbors,ff,tt,mm,16.,16.,16.,length,1.,
                (length+1.25)**2,.3,.001,.03,1.)
            np.testing.assert_allclose(f,ff,rtol=1e-10,atol=1e-10)
            np.testing.assert_allclose(t,tt,rtol=1e-10,atol=1e-10)
            self.assertAlmostEqual(maximum,float(mm.max()),places=10)

    def test_degenerate_contacts_have_opposite_pushes(self):
        c=np.zeros((2,3)); pairs=np.array([[0,1]],dtype=np.int32)
        offsets,neighbors=base._pairs_to_csr_numba(pairs,2)
        for second in [[1.,0,0],[-1.,0,0],[0.,1,0],[1.,1e-14,0]]:
            u=np.array([[1.,0,0],second])
            f=np.zeros_like(c);t=np.zeros_like(c)
            periodic._apply_pair_forces_numba(pairs,c,u,f,t,16.,16.,16.,3.,1.,4.25**2,.3,.001,.03,1.)
            ff=np.zeros_like(c);tt=np.zeros_like(c);mm=np.zeros(2)
            periodic._apply_forces_per_fiber_numba(c,u,offsets,neighbors,ff,tt,mm,
                16.,16.,16.,3.,1.,4.25**2,.3,.001,.03,1.)
            np.testing.assert_allclose(ff.sum(axis=0),0.,atol=1e-12)
            np.testing.assert_allclose(f,ff,atol=1e-12)
            self.assertGreater(float(np.linalg.norm(f)),0.)

    def test_public_default_and_explicit_legacy(self):
        from unittest.mock import patch
        from cufhom.pipeline import rve_generator as rve
        original=base._apply_pair_forces_numba
        params={'sam_verbose':False,'sam_geometry_backend':'cupy','sam_vf_tolerance':.001}
        def inner(p):
            self.assertEqual(p['sam_vf_tolerance'],.001)
            self.assertEqual(p['sam_geometry_backend'],'numba')
            self.assertIs(base._apply_pair_forces_numba,periodic._apply_pair_forces_numba)
            return {'sam_geometry_backend_effective':'numba'}
        with patch.object(rve,'_generate_rve_main',side_effect=inner):
            result=rve.generate_rve_main(params)
        self.assertEqual(result['sam_implementation'],'periodic_v2')
        self.assertEqual(result['sam_geometry_backend_requested'],'cupy')
        self.assertEqual(params['sam_geometry_backend'],'cupy')
        self.assertNotIn('sam_implementation',params)
        self.assertIs(base._apply_pair_forces_numba,original)
        with patch.object(rve,'_generate_rve_main',return_value={}) as mock:
            result=rve.generate_rve_main(dict(params,sam_implementation='legacy'))
            self.assertEqual(mock.call_args.args[0]['sam_geometry_backend'],'cupy')
        self.assertEqual(result['sam_implementation'],'legacy')
        with self.assertRaises(ValueError):
            rve.generate_rve_main({'sam_implementation':'typo'})

    def test_public_exception_restores_kernels(self):
        from unittest.mock import patch
        from cufhom.pipeline import rve_generator as rve
        original=base._apply_pair_forces_numba
        with patch.object(rve,'_generate_rve_main',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                rve.generate_rve_main({'sam_verbose':False})
        self.assertIs(base._apply_pair_forces_numba,original)

    def test_rebuild_interval_one_refreshes_every_iteration(self):
        from unittest.mock import patch
        g=base.SAMLiteGenerator(np.eye(3)/3,3.,1.,.1,(16.,16.,16.))
        g._n_fibers=2
        g._centers[:2]=[[0.,0,0],[0.,.1,0]]
        g._directions[:2]=[[1.,0,0],[1.,0,0]]
        with patch.object(g,'build_neighbor_pairs_vec',wraps=g.build_neighbor_pairs_vec) as build:
            with patch.object(g,'_apply_pair_forces_vec',return_value=1.):
                with patch.object(g,'measure_max_overlap',return_value=1.):
                    result=g.relax(max_iter=3,apply_orientation=False,pair_rebuild_interval=1)
        self.assertEqual(result['iters'],3)
        self.assertEqual(build.call_count,3)


if __name__=='__main__': unittest.main()
