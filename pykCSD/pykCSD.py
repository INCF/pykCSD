# -*- coding: utf-8 -*-

class KCSD:
    """
    Main class for instantiating a Kernel Current Source Density Solver.
    """

    def __init__(self, electrode_positions, sampled_pots, **kwargs):
        """
        Optional parameters: 
        'source_radius' -- radius of a base element,
        'n_sources' -- number of sources,
        'conductance' -- space conductance of the medium,
        'lambda' -- tikhonov regularization parameter for ridge regression
        'h',
        'x_min', 
        'x_max',
        'y_min',
        'y_max',
        'z_min',
        'z_max'
        """
        dim = len(electrode_positions.shape)
        if dim == 1:
            pass
        #   self.solver = KCSD1D()
        elif dim == 2:
            pass
        #   self.solver = KCSD2D()
        elif dim == 3:
            pass
        #   self.solver = KCSD3D()
        else:
            pass

    def estimate_potentials(self):
        """
        Calculates Local Field Potentials using the instantiated solver.
        """
        #self.solver.estimate_potentials()
        pass

    def estimate_csd(self):
        """
        Calculates Current Source Density using the instantiated solver.
        """
        #self.solver.estimate_csd()
        pass

    def save(self, filename = 'result'):
        """
        Save results to file.
        """
        pass

if __name__ == '__main__':
    pass