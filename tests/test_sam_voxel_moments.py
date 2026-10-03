"""Fused A2 must preserve the original normalization and filtering semantics."""
import unittest
import numpy as np
from cufhom.pipeline.sam_voxel_moments import voxel_a2 as _voxelized_a2
from tests.helpers.voxel_reference import _voxelized_a2_reference
from numba import set_num_threads


class VoxelMomentTests(unittest.TestCase):
    def setUp(self): set_num_threads(2)

    def test_reference_random_and_invalid_directions(self):
        rng=np.random.default_rng(761)
        for dtype in [np.float32,np.float64]:
            p=rng.integers(0,2,(34,32,31),dtype=np.uint8)
            o=rng.normal(size=p.shape+(3,)).astype(dtype)*3
            p[0,0,:5]=1
            o[0,0,:3]=[[0,0,0],[1e-13,0,0],[np.nan,0,0]]
            np.testing.assert_allclose(_voxelized_a2(p,o),_voxelized_a2_reference(p,o),atol=2e-14,rtol=2e-13)

    def test_empty_phase_and_zero_directions(self):
        for phase in [np.zeros((3,3,3),dtype=np.uint8),np.ones((3,3,3),dtype=np.uint8)]:
            np.testing.assert_array_equal(_voxelized_a2(phase,np.zeros((3,3,3,3))),np.zeros((3,3)))

    def test_thread_count_determinism_and_full_grid(self):
        rng=np.random.default_rng(52);p=rng.integers(0,2,(96,96,96),dtype=np.uint8)
        o=rng.normal(size=p.shape+(3,)).astype(np.float32)
        set_num_threads(1); a=_voxelized_a2(p,o)
        set_num_threads(2); b=_voxelized_a2(p,o)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(a,_voxelized_a2_reference(p,o),rtol=1e-12,atol=1e-14)


if __name__=='__main__': unittest.main()
