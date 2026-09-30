"""Research-only compact Fourier coarse correction; no microstructure labels."""
from contextlib import contextmanager
from itertools import product
import time
from unittest.mock import patch
import numpy as np
import cupy as cp
from cufhom.pipeline import linear_coordinates as lc


class FourierBasis:
    def __init__(self, shape, radius, xp):
        self.xp = xp
        self.shape = tuple(shape)
        hshape = self.shape[:-1]+(self.shape[-1]//2+1,)
        self.count = int(np.prod(hshape))
        self.size = 3*self.count
        ia, ib, wa, wb = [], [], [], []
        for k in product(range(-radius, radius+1), repeat=3):
            if not 0 < sum(v*v for v in k) <= radius*radius:
                continue
            if any(abs(v) > (n-1)//2 for v,n in zip(k,shape)):
                continue
            if not (k[2]>0 or (k[2]==0 and (k[1]>0 or (k[1]==0 and k[0]>0)))):
                continue
            plus = np.ravel_multi_index((k[0]%shape[0],k[1]%shape[1],k[2]),hshape)
            paired = k[2]==0
            minus = np.ravel_multi_index((-k[0]%shape[0],-k[1]%shape[1],0),hshape) if paired else plus
            for component in range(3):
                for part in (0,1):
                    ia.append(2*(component*self.count+plus)+part)
                    ib.append(2*(component*self.count+minus)+part)
                    wa.append(1/np.sqrt(2) if paired else 1.)
                    wb.append((1 if part==0 else -1)/np.sqrt(2) if paired else 0.)
        self.ia, self.ib = xp.asarray(ia), xp.asarray(ib)
        self.wa, self.wb = xp.asarray(wa), xp.asarray(wb)
        self.rank = len(ia)

    def restrict(self, x):
        real = x.view(x.real.dtype)
        return self.wa*real[self.ia]+self.wb*real[self.ib]

    def add(self, x, coefficients):
        real = x.view(x.real.dtype)
        real[self.ia] += self.wa*coefficients
        real[self.ib] += self.wb*coefficients
        return x

    def column(self, j, dtype):
        c = self.xp.zeros(self.rank)
        c[j]=1
        return self.add(self.xp.zeros(self.size,dtype),c)


class CoarseCorrection:
    def __init__(self, action, rhs, shape, radius, kind):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        self.U = FourierBasis(shape, radius, cp)
        if rhs.size != self.U.size:
            raise ValueError('Expected fixed-mean compatible displacement coordinates')
        self.kind = kind
        columns, actions = [], []
        for j in range(self.U.rank):
            v = self.U.column(j, rhs.dtype)
            av = action(v)
            columns.append(self.U.restrict(av))
            if kind == 'balanced':
                actions.append(av)
        matrix = cp.asnumpy(cp.stack(columns,axis=1))
        self.symmetry = float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix))
        if self.symmetry > 1e-5:
            raise ValueError(f'Coarse operator lost symmetry: {self.symmetry}')
        matrix = (matrix+matrix.T)/2
        eigenvalues = np.linalg.eigvalsh(matrix)
        np.linalg.cholesky(matrix)
        self.eigenvalues = [float(eigenvalues[0]),float(eigenvalues[-1])]
        self.inverse = cp.asarray(np.linalg.solve(matrix,np.eye(self.U.rank)))
        self.HU = None
        if kind == 'balanced':
            self.HU = cp.stack([v.view(rhs.real.dtype) for v in actions],axis=1)
        cp.cuda.get_current_stream().synchronize()
        self.setup_s=time.perf_counter()-start
        self.bytes=int(self.inverse.nbytes+(self.HU.nbytes if self.HU is not None else 0))

    def __call__(self, r):
        coarse = self.U.restrict(r)
        a = self.inverse@coarse
        if self.kind == 'block':
            # Identity on the orthogonal complement, exact Galerkin inverse on U.
            return self.U.add(r.copy(),a-coarse)
        q = r.copy()
        q.view(r.real.dtype)[:] -= self.HU@a.astype(r.real.dtype)
        t = self.inverse@(self.HU.T@q.view(r.real.dtype)).astype(cp.float64)
        return self.U.add(q,a-t)


@contextmanager
def trial(shape, radius, kind, records):
    original = lc.fused_cg
    coarse = None
    def solve(action,rhs,dot,tolerance,maxiter,xp,**kwargs):
        nonlocal coarse
        if coarse is None:
            coarse = CoarseCorrection(action,rhs,shape,radius,kind)
            records.append(dict(kind=kind,radius=radius,rank=coarse.U.rank,
                setup_s=coarse.setup_s,coarse_symmetry=coarse.symmetry,
                coarse_eigenvalues=coarse.eigenvalues,payload_bytes=coarse.bytes))
        if kwargs.get('precondition') is not None:
            raise ValueError('Expected already symmetrically whitened coordinates')
        kwargs['precondition'] = coarse
        return original(action,rhs,dot,tolerance,maxiter,xp,**kwargs)
    with patch.object(lc,'fused_cg',solve):
        yield
