import numpy as np
import cufhom_core.applications
from cufhom_core.general.base import Timer


class Problem(object):
    """Linear elasticity homogenization: material, discretization and solver
    configuration ('material', 'solve', 'solver'), results in output."""

    def __init__(self, conf_problem):
        self.__dict__.update(conf_problem)
        self.Y = np.array(self.material['Y'], dtype=np.float64)
        self.dim = self.Y.size
        self.shape = (int(self.dim*(self.dim+1)/2),)
        self.output = {}

    def calculate(self):
        print('\n==============================')
        tim = Timer(name='application')
        cufhom_core.applications.elasticity(self)
        tim.measure()

    def postprocessing(self):
        print('\nHomogenized matrices')
        for key, val in self.output['mat_primal'].items():
            print(key)
            print(val)

    def __repr__(self):
        def format_value(val):
            shape = getattr(val, 'shape', None)
            dtype = getattr(val, 'dtype', None)
            size = getattr(val, 'size', None)
            if shape is not None and dtype is not None:
                if size is None or int(size) > 32:
                    return '{}(shape={}, dtype={})'.format(type(val).__name__, tuple(shape), dtype)
                return str(val)
            return str(val)

        ss = "Class : {}\n".format(self.__class__.__name__)
        ss += '    name : {}\n'.format(self.name)
        ss += '    dim = {} (dimension)\n'.format(self.dim)
        ss += '    Y = {} (PUC size)\n'.format(self.Y)
        ss += '    material:\n'
        for key, val in list(self.material.items()):
            ss += '        {0} : {1}\n'.format(key, format_value(val))
        ss += '    solve:\n'
        for key, val in list(self.solve.items()):
            ss += '        {0} : {1}\n'.format(key, format_value(val))
        return ss
