"""Conservative periodic AABB neighbor list with translation/rotation skin."""
import numpy as np
from numba import njit


@njit(cache=True)
def build_aabb_pairs(centers, directions, box, length, diameter, skin):
    # Two passes avoid allocating an n(n-1)/2 buffer when the graph is sparse.
    n=len(centers); count=0
    pairs=np.empty((0,2),dtype=np.int32)
    reach=diameter+skin+1e-12*max(1.,length,diameter,np.max(box))
    for phase in range(2):
        cursor=0
        for i in range(n):
            for j in range(i+1,n):
                lower2=0.
                for axis in range(3):
                    delta=centers[j,axis]-centers[i,axis]
                    delta-=round(delta/box[axis])*box[axis]
                    gap=max(0.,abs(delta)-.5*length*(abs(directions[i,axis])+abs(directions[j,axis])))
                    lower2+=gap*gap
                if lower2<=reach*reach:
                    if phase==1:
                        pairs[cursor,0]=i; pairs[cursor,1]=j
                    cursor+=1
        if phase==0:
            count=cursor
            pairs=np.empty((count,2),dtype=np.int32)
    return pairs


@njit(cache=True)
def endpoint_motion_bound(centers, directions, ref_centers, ref_directions, box, length):
    """Hausdorff-motion upper bound of each rigid segment on a periodic torus."""
    maximum=0.
    for i in range(len(centers)):
        translation2=0.; rotation2=0.
        for axis in range(3):
            delta=centers[i,axis]-ref_centers[i,axis]
            delta-=round(delta/box[axis])*box[axis]
            translation2+=delta*delta
            rotation2+=(directions[i,axis]-ref_directions[i,axis])**2
        maximum=max(maximum,np.sqrt(translation2)+.5*length*np.sqrt(rotation2))
    return maximum
