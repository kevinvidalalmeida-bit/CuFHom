"""Experimental collective axes for nearly parallel contacting rigid fibers.

This is an optimization experiment, not the published SAM algorithm. All
certificates still inspect individual physical fibers and the original targets.
"""
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from cufhom.pipeline.sam_paper_reference import (
    ContactNeighbors, objective, rotational_inertia, relax_paper_reference,
)
from cufhom.pipeline.sam_staged_insertion import certificate


HERE = Path(__file__).resolve().parent


def contact_groups(f, box, threshold, gap):
    """Join near-parallel axes only when their finite segments are neighbors."""
    n = len(f)
    cache = ContactNeighbors(f[0,6],np.full(3,box),skin=.05,contact_diameter=f[0,7]+gap)
    offsets, j = cache.get(f[:,:3],f[:,3:6])
    i = np.repeat(np.arange(n),np.diff(offsets))
    keep = (i < j) & (np.abs(np.sum(f[i,3:6]*f[j,3:6],axis=1)) >= np.cos(threshold))
    parent = np.arange(n)
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a,b in zip(i[keep],j[keep]):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[rb] = ra
    _, labels = np.unique([root(i) for i in range(n)],return_inverse=True)
    counts = np.bincount(labels)
    axes = np.empty((len(counts),3))
    for label in range(len(counts)):
        members = f[labels==label,3:6]
        sign = np.where(members@members[0] >= 0,1.,-1.)
        axes[label] = np.sum(sign[:,None]*members,axis=0)
        axes[label] /= np.linalg.norm(axes[label])
    signs = np.where(np.sum(f[:,3:6]*axes[labels],axis=1) >= 0,1.,-1.)
    return labels, counts, axes, signs


def grouped_relax(f, box, a2, a4, *, gap=.001, threshold=1e-4,maxiter=10000):
    labels, counts, axes, signs = contact_groups(f,box,threshold,gap)
    n = len(f)
    eps = rotational_inertia(f[0,6],f[0,7])
    x0 = np.r_[f[:,:3].ravel(),(np.sqrt(eps*counts)[:,None]*axes).ravel()]
    neighbors = ContactNeighbors(f[0,6],np.full(3,box),skin=.5,contact_diameter=f[0,7]+gap)
    best_score, best_f = np.inf, f.copy()
    steps, evaluations, history = 0,0,[]
    last_x = last = None
    def unpack(x):
        centers = np.ascontiguousarray(x[:3*n].reshape(n,3))
        raw = x[3*n:].reshape(-1,3)
        norms = np.linalg.norm(raw,axis=1)
        axis = raw/norms[:,None]
        u = np.ascontiguousarray(signs[:,None]*axis[labels])
        return centers, axis, u, norms
    def fun(x):
        nonlocal evaluations,last_x,last
        if last_x is not None and np.array_equal(x,last_x):
            return last[0],last[1]
        c,axis,u,norms = unpack(x)
        e,gc,gu,overlap,a4err = objective(c,u,f[0,6],np.full(3,box),
            neighbors=neighbors,target_a4=a4,contact_diameter=f[0,7]+gap)
        group_gu = np.zeros_like(axis)
        np.add.at(group_gu,labels,signs[:,None]*gu)
        tangent = group_gu-axis*np.sum(group_gu*axis,axis=1)[:,None]
        gradient = np.r_[gc.ravel(),(tangent/norms[:,None]).ravel()]
        last_x = x.copy()
        last = e,gradient,overlap,a4err
        evaluations += 1
        return e,gradient
    def record(x):
        nonlocal best_score,best_f
        fun(x)
        c,axis,u,norms = unpack(x)
        e,g,overlap,a4err = last
        a2err = np.linalg.norm(u.T@u/n-a2)/np.linalg.norm(a2)
        score = max(overlap/1e-6,a2err/.035,a4err/1e-7)
        if score < best_score:
            best_score = score
            best_f = np.column_stack((c%box,u,f[:,6:]))
        if steps%25==0 or score<=1:
            history.append(dict(iteration=steps,energy=e,overlap_relative=overlap,
                                A2_error=a2err,A4_error=a4err))
        return score
    def callback(x):
        nonlocal steps
        steps += 1
        if record(x)<=1:
            raise StopIteration
    started = time.perf_counter()
    record(x0)
    result = minimize(fun,x0,jac=True,method='L-BFGS-B',callback=callback,
                      options=dict(maxiter=maxiter,maxfun=32*(maxiter+1),maxls=30,
                                   maxcor=12,ftol=0.,gtol=1e-12))
    record(result.x)
    check = certificate(best_f,box,a2,1e-6,.035,1e-7,a4,gap)
    return best_f,dict(certificate=check,iterations=steps,evaluations=evaluations,
                      message=str(result.message),wall_seconds=time.perf_counter()-started,
                      group_count=len(counts),grouped_fibers=int(counts[counts>1].sum()),
                      largest_group=int(counts.max()),angular_threshold_radians=threshold,
                      history=history)


def main():
    with np.load(HERE/'diagnostic_gradient_paper_trial.npz',allow_pickle=False) as saved:
        f = saved['relaxed_fibers'].copy()
        a2,a4 = saved['A2'].copy(),saved['A4'].copy()
    results = {}
    started = time.perf_counter()
    ordinary, info = relax_paper_reference(f,70.,a2,target_a4=a4,min_gap=.001,
        optimizer='lbfgs',maxiter=10000,overlap_tolerance=1e-6,a4_tolerance=1e-7)
    results['ordinary_lbfgs'] = dict(info,wall_seconds=time.perf_counter()-started)
    np.savez_compressed(HERE/'ordinary_from_gradient_trial.npz',fibers=ordinary,
                        diagnostic_only=True)
    grouped, info = grouped_relax(f,70.,a2,a4,maxiter=10000)
    results['grouped_lbfgs'] = info
    np.savez_compressed(HERE/'grouped_from_gradient_trial.npz',fibers=grouped,
                        diagnostic_only=True)
    (HERE/'grouped_axes_probe.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in value.items() if k!='history'}
                      for name,value in results.items()},indent=2))


if __name__=='__main__':
    main()
