"""Basic methods for trigonometric polynomials."""


import numpy as np

fft_form_default='r' # real input data


class Grid():
    @staticmethod
    def get_xil(N, Y, fft_form=fft_form_default):
        """
        it produces discrete frequencies of Fourier series
        xil[i] = ZNl[i]/Y[i]
        """
        xil = []
        for m in np.arange(np.size(N)):
            xil.append(np.arange(np.trunc(-N[m]/2.), np.trunc(N[m]/2.+0.5))/Y[m])
        if fft_form in ['r',]:
            xil=[np.fft.ifftshift(xi) for xi in xil]
            xil[-1] = xil[-1][:int(np.trunc(N[-1]/2)+1)]
        elif fft_form in [0]:
            xil = [np.fft.ifftshift(xi) for xi in xil]
        return xil

def get_Nodd(N):
    Nodd = N - ((N + 1) % 2)
    return Nodd

def mean_index(N, fft_form=fft_form_default):
    if fft_form in [0, 'r']:
        return tuple(np.zeros_like(N, dtype=int))
    elif fft_form in ['c']:
        return tuple(np.array(np.trunc(np.array(N)/2), dtype=int))
