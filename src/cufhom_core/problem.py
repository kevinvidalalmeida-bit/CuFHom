import numpy as np
import cufhom_core.applications
from cufhom_core.general.base import Timer
import os


class Problem(object):
    """
    Class that parse input file, calculates the physical problem,
    and post-process the results (calculates the homogenized properties.)
    """
    def __init__(self, conf_problem=None, conf=None):
        """
        Parameters
        ----------
        conf_problem : dictionary
            particular problem from problems in input file; dictionary that
            usually contains following keywords: 'name', 'material', 'solve',
            'solver', 'postprocess'
        conf : module
            configuration data from input file
        """
        self.__dict__.update(conf_problem)
        if isinstance(self.material, str):
            conf_material = conf.materials[self.material]
        else:
            conf_material = self.material

        self.material = self.parse_material(conf_material)

        self.Y = np.array(self.material['Y'], dtype=np.float64)
        self.dim = self.Y.size

        if self.physics == 'scalar':
            self.shape = (self.dim,)
        elif self.physics == 'elasticity':
            self.shape = (int(self.dim*(self.dim+1)/2),)

        self.output = {}

    @staticmethod
    def parse_material(conf_material):
        """
        Parse material from input file.
        """
        if (
            'fun' in conf_material
            or 'Cfield_path' in conf_material
            or 'Cfield' in conf_material
            or ('Cfield_index' in conf_material and 'Cfield_table' in conf_material)
        ):
            material = conf_material
        else:
            material = conf_material
            for incl in material['inclusions']:
                if incl in ['all', 'otherwise']:
                    n_incl = material['inclusions'].count(incl)
                    if n_incl == 0:
                        continue
                    elif n_incl == 1:
                        ind = material['inclusions'].index(incl)
                        if ind == material['inclusions'].__len__() - 1:
                            continue
                        else:
                            msg='inclusion ("all" or "otherwise") has to be at the end of the list'
                            raise ValueError(msg)
                    else:
                        msg = "Maximal one occurrence of inclusion \
                            'otherwise' or 'all' is allowed!"
                        raise ValueError(msg)
        return material

    def calculate(self):
        """
        Calculates the problem according to physical model.
        """
        print('\n==============================')
        tim = Timer(name='application')
        if hasattr(cufhom_core.applications, self.physics):
            getattr(cufhom_core.applications, self.physics)(self)
        else:
            msg = 'Not implemented physics ({0}).\n' \
                'Hint: Implement function ({1}) into module' \
                ' cufhom_core!'.format(self.physics, self.physics)
            raise NotImplementedError(msg)
        tim.measure()

    def postprocessing(self):
        """
        Post-process the results. Usually consists of plotting of homogenized
        properties.
        """
        output = self.output
        if self.physics in ['scalar', 'elasticity']:
            print('\nHomogenized matrices')
            for primaldual in self.solve['primaldual']:
                for key, val in list(output['mat_'+primaldual].items()):
                    print(key)
                    print(val)

        if hasattr(self, 'save'):
            if 'data' not in self.save:
                self.save['data'] = 'all'

            if self.save['data'] == 'all':
                data = self.output
                data.update({'physics': self.physics,
                             'solve': self.solve,
                             'solver': self.solver,
                             'postprocess': self.postprocess,
                             'save': self.save,
                             'material': self.material})

            filename = self.save['filename']
            dirs = os.path.dirname(filename)
            if not os.path.exists(dirs) and dirs != '':
                os.makedirs(dirs)

            import pickle
            with open(filename, 'wb') as fop:
                pickle.dump(self.output, fop, protocol=3)

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
        ss += '    physics = {}\n'.format(self.physics)
        ss += '    dim = {} (dimension)\n'.format(self.dim)
        ss += '    Y = {} (PUC size)\n'.format(self.Y)
        ss += '    material:\n'
        for key, val in list(self.material.items()):
            ss += '        {0} : {1}\n'.format(key, format_value(val))
        ss += '    solve:\n'
        for key, val in list(self.solve.items()):
            ss += '        {0} : {1}\n'.format(key, format_value(val))
        return ss
