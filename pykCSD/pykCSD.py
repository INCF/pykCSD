# -*- coding: utf-8 -*-
from KCSD1D import KCSD1D
from KCSD2D import KCSD2D


class KCSD(object):
    """
    Main class for instantiating a Kernel Current Source Density Solver.
    """

    def __init__(self, elec_pos, sampled_pots, params=[]):
        """
        Optional parameters:
        'source_radius' -- radius of a base element,
        'n_sources' -- number of sources,
        'conductance' -- space conductance of the medium,
        'lambda' -- regularization parameter for ridge regression
        'h',
        'x_min',
        'x_max',
        'y_min',
        'y_max',
        'z_min',
        'z_max'
        """
        dim = len(elec_pos.shape)
        print("Initializing kCSD %dD" % (dim))

        if dim == 1:
            self.solver = KCSD1D(elec_pos, sampled_pots, params)
        elif dim == 2:
           self.solver = KCSD2D(elec_pos, sampled_pots, params)
        elif dim == 3:
            pass
        #   self.solver = KCSD3D()
        else:
            raise Exception("Incorrect electrode format.")
        self.solver.calculate_matrices()

    def estimate_pots(self):
        """
        Calculates Local Field Potentials using the instantiated solver.
        """
        self.solver.estimate_pots()

    def estimate_csd(self):
        """
        Calculates Current Source Density using the instantiated solver.
        """
        self.solver.estimate_csd()

    def save(self, filename='result'):
        """
        Save results to file.
        """
        pass

    def plot_all(self):
        self.solver.plot_all()

if __name__ == '__main__':
    pass