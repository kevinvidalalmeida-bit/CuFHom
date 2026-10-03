"""Fused voxel orientation moments: float64, deterministic blocks, no fastmath."""
import numpy as np
from numba import njit, prange
from itertools import combinations_with_replacement, permutations


_A4_COMPONENTS = np.array(list(combinations_with_replacement(range(3), 4)), dtype=np.int64)


@njit(cache=True, parallel=True)
def _voxel_a4_components(phase, orientations):
    """15 independent components, O(blocks) scratch rather than O(voxels*81)."""
    block_size = 16384
    blocks = (len(phase)+block_size-1)//block_size
    partial = np.zeros((blocks, 16), dtype=np.float64)
    for block in prange(blocks):
        for at in range(block*block_size, min((block+1)*block_size, len(phase))):
            if phase[at] == 0:
                continue
            norm2 = 0.
            for axis in range(3):
                norm2 += np.float64(orientations[at, axis])**2
            if not np.isfinite(norm2) or norm2 <= 1e-24:
                continue
            for component in range(15):
                value = 1.
                for axis in range(4):
                    value *= np.float64(orientations[at, _A4_COMPONENTS[component, axis]])
                partial[block, component] += value/(norm2*norm2)
            partial[block, 15] += 1.
    total = np.zeros(16)
    for block in range(blocks):
        for component in range(16):
            total[component] += partial[block, component]
    return total


def voxel_a4(phase, orientations):
    """Full symmetric fourth moment of occupied voxels, unit-axis convention."""
    total = _voxel_a4_components(np.asarray(phase).reshape(-1),
                                 np.asarray(orientations).reshape(-1, 3))
    result = np.zeros((3, 3, 3, 3))
    if total[15]:
        for component, indices in enumerate(_A4_COMPONENTS):
            for perm in set(permutations(indices)):
                result[perm] = total[component]/total[15]
    return result


@njit(cache=True, parallel=True)
def _voxel_a2(phase, orientations):
    block_size=16384
    blocks=(len(phase)+block_size-1)//block_size
    partial=np.zeros((blocks,7),dtype=np.float64)
    for block in prange(blocks):
        xx=0.; yy=0.; zz=0.; xy=0.; xz=0.; yz=0.; count=0.
        for at in range(block*block_size,min((block+1)*block_size,len(phase))):
            if phase[at]==0:
                continue
            x=np.float64(orientations[at,0]);y=np.float64(orientations[at,1]);z=np.float64(orientations[at,2])
            norm=np.sqrt(x*x+y*y+z*z)
            if norm<=1e-12 or np.isnan(norm):
                continue
            x/=norm;y/=norm;z/=norm
            xx+=x*x;yy+=y*y;zz+=z*z;xy+=x*y;xz+=x*z;yz+=y*z;count+=1.
        partial[block,0]=xx;partial[block,1]=yy;partial[block,2]=zz
        partial[block,3]=xy;partial[block,4]=xz;partial[block,5]=yz;partial[block,6]=count
    totals=np.zeros(7,dtype=np.float64)
    for block in range(blocks):
        for component in range(7):
            totals[component]+=partial[block,component]
    result=np.zeros((3,3),dtype=np.float64)
    if totals[6]>0:
        result[0,0]=totals[0]/totals[6];result[1,1]=totals[1]/totals[6];result[2,2]=totals[2]/totals[6]
        result[0,1]=result[1,0]=totals[3]/totals[6]
        result[0,2]=result[2,0]=totals[4]/totals[6]
        result[1,2]=result[2,1]=totals[5]/totals[6]
    return result


def voxel_a2(phase, orientations):
    """Second moment of occupied voxels, normalizing each valid direction."""
    return _voxel_a2(np.asarray(phase).reshape(-1), np.asarray(orientations).reshape(-1, 3))
