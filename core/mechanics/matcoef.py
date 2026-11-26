import numpy as np


class ElasticTensor():
    """Isotropic 3-D stiffness 3K P_vol + 2mu P_dev as a fourth-order tensor
    (val) and in Mandel notation (mandel, order 11, 22, 33, 23, 13, 12)."""

    def __init__(self, bulk, mu):
        self.bulk = float(bulk)
        self.mu = float(mu)
        ids = np.eye(3)
        volumetric = 1./3*np.einsum('ij,kl', ids, ids)
        idsym = 0.5*(np.einsum('ik,jl', ids, ids) + np.einsum('il,jk', ids, ids))
        deviatoric = idsym - volumetric
        self.val = 3.*bulk*volumetric + 2.*mu*deviatoric
        self.mandel = self.create_mandel(self.val)

    @staticmethod
    def create_mandel(mat):
        """Symmetric (3, 3, 3, 3) tensor to its (6, 6) Mandel matrix."""
        res = np.zeros((6, 6), dtype=mat.dtype)
        for ii, jj in np.ndindex(3, 3):
            kk = list(range(3))
            kk.remove(ii)
            ll = list(range(3))
            ll.remove(jj)
            res[ii, jj] = mat[ii, ii, jj, jj]
            res[ii, jj+3] = 2**.5*mat[ii, ii, ll[0], ll[1]]
            res[jj+3, ii] = res[ii, jj+3]
            res[ii+3, jj+3] = 2*mat[kk[0], kk[1], ll[0], ll[1]]
        return res
