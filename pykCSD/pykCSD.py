# -*- coding: utf-8 -*-
from KCSD1D import KCSD1D
from KCSD2D import KCSD2D
from KCSD3D import KCSD3D


class KCSD(object):
    """
    Main class for instantiating a Kernel Current Source Density Solver.
    """

    def __init__(self, elec_pos, sampled_pots, params={}):
        """
        **Parameters**

        elec_pos : numpy array
            positions of electrodes
        
        sampled_pots : numpy array
            potentials measured by electrodes
        
        params : set, optional
            configuration parameters, that may contain the following keys:
            'sigma' : float
                space conductance of the medium
            
            'n_sources' : int
                number of sources
            
            'source_type' : str
                basis function type ('gauss', 'step', 'gauss_lim')
            
            'R_init' : float
                demanded thickness of the basis element
            
            'h' : float
                cylinder radius (KCSD1D) or slice thickness (KCSD2D)
            
            'dist_density' : int
                resolution of the dist_table
            
            'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max' : floats
                boundaries for CSD estimation space
            
            'ext' : float
                length of space extension: x_min-ext ... x_max+ext
            
            'gdX', 'gdY', 'gdZ': floats
                space increments (granularity) of the estimation space

            'lambd' : float
                regularization parameter for ridge regression

        **Methods**

        estimate_pots()
            Calculate Local Field Potentials using kCSD method.
        
        estimate_csd()
            Calculate Current Source Density using kCSD method.
        
        plot_all()
            Show a quick plot to investigate the data.
        """
        dim = len(elec_pos[0])
        print("Initializing kCSD %dD" % (dim))

        if dim == 1:
            self.solver = KCSD1D(elec_pos, sampled_pots, params)
        elif dim == 2:
            self.solver = KCSD2D(elec_pos, sampled_pots, params)
        elif dim == 3:
            self.solver = KCSD3D(elec_pos, sampled_pots, params)
        else:
            raise Exception("Incorrect electrode format.")
        self.solver.init_model()

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
        #HDF5?
        #.mat?
        pass

    def plot_all(self):
        """
        Quick plot of input and output data.
        """
        self.solver.plot_all()

if __name__ == '__main__':
    pass