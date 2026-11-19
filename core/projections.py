"""Fourier projections onto compatible, zero-mean strain fields (3-D, Mandel).

The primal GaNi operator only needs hG1 = hG1h + hG1s. Even-grid Nyquist
indices differentiate the real trigonometric interpolant: their wavenumber is
zero on that axis; a vanishing wavevector has no compatible fluctuation.
"""
import numpy as np
from cufhom_core.trigpol import Grid, mean_index, fft_form_default
from cufhom_core.tensors import Tensor
from cufhom_core.tensors.fft import get_fft_backend


def elasticity(N, Y, fft_form=fft_form_default, dtype=float):
    """Stored projection hG1 as a 6x6 field over the frequencies (fft_form 0 or 'r')."""
    if fft_form in ['r']:
        fft_form_r=True
        fft_form=0
    else:
        fft_form_r=False

    dtype = np.dtype(dtype)
    N = np.array(N, dtype=int)
    d = N.size
    if d != 3:
        raise NotImplementedError("The elasticity projection is implemented for 3-D grids.")
    D = 6
    Nred = N
    xi = Grid.get_xil(N, Y, fft_form=fft_form)
    for m in range(d):
        if N[m] % 2 == 0:
            xi[m] = np.where(np.isclose(xi[m]*Y[m], -N[m]/2), 0., xi[m])

    xi2 = []
    for ii in range(d):
        xi2.append(xi[ii]**2)

    num = np.zeros(np.hstack([d, d, Nred]), dtype=dtype)
    norm2_xi = np.zeros(Nred, dtype=dtype)
    for mm in np.arange(d): # diagonal components
        Nshape = np.ones(d, dtype=int)
        Nshape[mm] = Nred[mm]
        Nrep = np.copy(Nred)
        Nrep[mm] = 1
        num[mm][mm] = np.tile(np.reshape(xi2[mm], Nshape), Nrep) # numerator
        norm2_xi += num[mm][mm]

    norm4_xi = norm2_xi**2
    ind_center = mean_index(Nred, fft_form=fft_form)
    vanishing = norm2_xi == 0
    # avoid division by zero
    norm2_xi[vanishing] = 1
    norm4_xi[vanishing] = 1

    for m in np.arange(d): # upper diagonal components
        for n in np.arange(m+1, d):
            NshapeM = np.ones(d, dtype=int)
            NshapeM[m] = Nred[m]
            NrepM = np.copy(Nred)
            NrepM[m] = 1
            NshapeN = np.ones(d, dtype=int)
            NshapeN[n] = Nred[n]
            NrepN = np.copy(Nred)
            NrepN[n] = 1
            num[m][n] = np.tile(np.reshape(xi[m], NshapeM), NrepM) \
                * np.tile(np.reshape(xi[n], NshapeN), NrepN)

    G1h = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    S = np.zeros(np.hstack([D, D, Nred]), dtype=dtype)
    for m in np.arange(d):
        S[m][m] = 2*num[m][m]/norm2_xi
        for n in np.arange(d):
            G1h[m][n] = num[m][m]*num[n][n]/norm4_xi
    for m in np.arange(d):
        S[m+3][m+3] = 1 - num[m][m]/norm2_xi
        S[m+3][m+3][ind_center] = 0
    for m in np.arange(d):
        for n in np.arange(m+1, d):
            S[m+3][n+3] = num[m][n]/norm2_xi
    for m in np.arange(d):
        for n in np.arange(d):
            ind = np.setdiff1d(np.arange(d), [n])
            S[m][n+3] = (0 == (m == n))*2**.5*num[ind[0]][ind[1]]/norm2_xi
            G1h[m][n+3] = 2**.5*num[m][m]*num[ind[0]][ind[1]]/norm4_xi
    for m in np.arange(d):
        for n in np.arange(d):
            ind_m = np.setdiff1d(np.arange(d), [m])
            ind_n = np.setdiff1d(np.arange(d), [n])
            G1h[m+3][n+3] = 2*num[ind_m[0]][ind_m[1]] \
                * num[ind_n[0]][ind_n[1]] / norm4_xi
    # symmetrization
    for n in np.arange(D):
        for m in np.arange(n+1, D):
            S[m][n] = S[n][m]
            G1h[m][n] = G1h[n][m]
    G1s = S - 2*G1h
    G1h[..., vanishing] = 0
    G1s[..., vanishing] = 0

    G1h = Tensor(name='hG1h', val=G1h, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form).enlarge(N)
    G1s = Tensor(name='hG1s', val=G1s, order=2, N=N, Fourier=True, multype=21, fft_form=fft_form).enlarge(N)
    if fft_form_r:
        for tensor in [G1h, G1s]:
            tensor.set_fft_form(fft_form='r')
            tensor.val=1./np.prod(tensor.N)*tensor.val
    return G1h + G1s


def elasticity_direct(N, Y, fft_form='r', dtype=np.float32):
    """Closed-form hG1 evaluated inside the CUDA projection kernel (no stored field)."""
    N = np.array(N, dtype=int)
    if N.size != 3 or fft_form not in [0, 'r'] or get_fft_backend() != 'cupy':
        raise NotImplementedError("The direct projection needs a 3-D grid, fft_form 0 or 'r' and CuPy.")
    if fft_form == 'r':
        grid_shape = tuple(int(v) for v in N[:-1].tolist()) + (int(N[-1] // 2 + 1),)
    else:
        grid_shape = tuple(int(v) for v in N.tolist())
    return Tensor(name='hG1_direct', val=np.zeros((1,) + grid_shape, dtype=np.dtype(dtype)), order=1,
                  N=N, Y=Y, Fourier=True, multype='elasticity_hg1_direct', fft_form=fft_form)
