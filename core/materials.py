"""Material coefficients on the FFT grid.

Two layouts reach the solver: a dense stiffness field ('Cfield', 6x6 or packed
sym21 per voxel) or an indexed table ('Cfield_index' per voxel, one sym21
row per orientation group in 'Cfield_table'). Both live at the grid points
(origin 0), the GaNi quadrature.
"""
from cufhom_core.tensors import Tensor


class Material():

    def __init__(self, material_conf):
        if 'Y' not in material_conf:
            raise ValueError("The definition of PUC size (Y) is missing!")
        indexed = 'Cfield_index' in material_conf and 'Cfield_table' in material_conf
        if not indexed and 'Cfield' not in material_conf:
            raise NotImplementedError("Material requires Cfield or Cfield_index and Cfield_table.")
        self.conf = material_conf
        self.Y = material_conf['Y']

    def get_A_GaNi(self, N, fft_form='c'):
        """Stiffness at the grid points (trapezoidal quadrature)."""
        if 'Cfield_index' in self.conf:
            A = Tensor(name='A_GaNi', val=self.conf['Cfield_index'][None, ...], order=1, N=N, Y=self.Y,
                       multype='sym21_indexed', Fourier=False, origin=0, fft_form=fft_form)
            A.material_table = self.conf['Cfield_table']
            return A
        A_val = self.conf['Cfield']
        order = A_val.ndim-len(N)
        multype = 'sym21' if self.conf.get('cfield_storage') == 'sym21' else 21
        return Tensor(name='A_GaNi', val=A_val, order=order, N=N, Y=self.Y, multype=multype,
                      Fourier=False, origin=0, fft_form=fft_form)
