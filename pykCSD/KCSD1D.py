# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi, uint16
from numpy import dot, transpose, identity
from matplotlib import pylab as plt

import cross_validation as cv
import basis_functions as bf
import potentials as pt

class KCSD1D(object):
    """
    1D variant of the kCSD method.
    It assumes constant distribution of sources in a cylinder around the electrodes.
    """

    def __init__(self, elec_pos, sampled_pots, params={}):
        """
        Required parameters:
            elec_pos (list-like) -- positions of electrodes
            sampled_pots (list-like) -- potentials measured by electrodes
        Optional parameters (keys in params dictionary):
            'sigma' -- space conductance of the medium 
            'n_sources' -- number of sources
            'source_type' -- basis function type ('gaussian', 'step')
            'h' -- thickness of the basis element
            'R' -- cylinder radius
            'dist_density' -- resolution of the output
            'x_min', 'x_max' -- boundaries for CSD estimation space
            'cross_validation' -- type of index generator 
            'lambda' -- regularization parameter for ridge regression
        """
        self.validate_parameters(elec_pos, sampled_pots)
        self.elec_pos = elec_pos
        self.sampled_pots = sampled_pots
        self.set_parameters(params)
 
    def validate_parameters(self, elec_pos, sampled_pots):
        if len(elec_pos) != len(sampled_pots):
            raise Exception("Number of measured potentials is not equal to electrode number!")
        if len(elec_pos) < 2:
            raise Exception("Number of electrodes must be at least 2!")
        elec_set = set(elec_pos)
        if len(elec_pos) != len(elec_set):
            raise Exception("Error! Duplicate electrode!")

    def set_parameters(self, params):
        self.sigma = params.get('sigma', 1.0)
        self.n_sources = params.get('n_sources', 300)
        self.xmax = params.get('x_max', max(self.elec_pos) )
        self.xmin = params.get('x_min', min(self.elec_pos))
        self.dist_density = params.get('dist_density', 200)
        self.lambd = params.get('lambda', 0.0)
        self.R = params.get('R', 1.0)
        self.h = params.get('h', abs(self.elec_pos[1] - self.elec_pos[0]))

        self.source_type = params.get('source_type', 'gaussian')
        if self.source_type not in ["gaussian", "step", "gauss_lim"]:
            raise Exception("Incorrect source type!")

        self.lambdas = np.array([1.0 / 2**n for n in xrange(0, 20)])
        self.source_positions = np.linspace(self.xmin, self.xmax, self.n_sources)
        self.estimation_area = np.linspace(self.xmin, self.xmax, self.dist_density)
        self.dist_max = self.xmax - self.xmin

    def estimate_pots(self):
        """Calculates Local Field Potentials."""
        estimation_table = self.interp_pot
    
        k_inv = np.linalg.inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)
        self.estimated_pots = dot(estimation_table, beta)

        return self.estimated_pots

    def estimate_csd(self):
        """Calculates Current Source Density."""
        estimation_table = self.k_interp_cross

        k_inv = np.linalg.inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)
        self.estimated_csd = dot(estimation_table, beta)

        return self.estimated_csd

    def save(self, filename='result'):
        """Save results to file."""
        pass

    def __repr__(self):
        info = ''.join(self.__class__.__name__)
        for key in vars(self).keys():
            if not key.startswith('_'):
                info += '%s : %s\n' %(key, vars(self)[key])
        return info

    def plot_all(self):
        fig, (ax11, ax21, ax22) = plt.subplots(1, 3, sharex=True)
        
        ax11.scatter(self.elec_pos, self.sampled_pots)
        ax11.set_title('Measured potentials')

        ax21.plot(self.estimation_area, self.estimated_pots)
        ax21.set_title('Calculated potentials')

        ax22.plot(self.estimation_area, self.estimated_csd)
        ax22.set_title('Calculated CSD')
        plt.show()

    #
    # subfunctions
    #

    def calculate_matrices(self):
        """
        Prepares all the required matrices to calculate kCSD.
        """
        self.create_dist_table()

        self.calculate_b_pot_matrix()
        self.k_pot = dot(self.b_pot_matrix.T, self.b_pot_matrix)

        self.calculate_b_src_matrix()
        self.k_interp_cross = np.dot(self.b_src_matrix, self.b_pot_matrix)

        self.calculate_b_interp_pot_matrix()
        self.interp_pot = dot(self.b_interp_pot_matrix.T, self.b_pot_matrix)


    def create_dist_table(self):
        """
        Create table of a single source contribution to overall potential 
        as a function of distance.
        """
        self.dist_table = np.zeros(self.dist_density + 1)

        for i in xrange(0, self.dist_density + 1):
            arg = (float(i)/self.dist_density) * self.dist_max
            self.dist_table[i] = pt.b_pot_quad(0, arg, self.h, self.R,
                                                   self.sigma, self.source_type)
        #return dist_table

    def calculate_b_pot_matrix(self):
        """ 
        Compute the matrix of potentials generated by every source basis function
        at every electrode position. 
        """
        n_elec = len(self.elec_pos)
    
        self.b_pot_matrix = np.zeros((self.n_sources, n_elec))
    
        for src_ind, src_pos in enumerate(self.source_positions):
            for elec_ind, elec_pos in enumerate(self.elec_pos):
                r = abs(src_pos - elec_pos)
            
                dt_ind = uint16(np.round(r * float(self.dist_density)/self.dist_max))
                #print dt_ind
                self.b_pot_matrix[src_ind, elec_ind] = self.dist_table[dt_ind]   

  

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        n_gx = len(self.estimation_area)

        self.b_src_matrix = np.zeros((n_gx, self.n_sources))

        for src_ind, curr_src in enumerate(self.source_positions):
            if self.source_type == "gaussian":
                self.b_src_matrix[:, src_ind] = bf.gauss_rescale_1D(self.estimation_area, curr_src, self.h)
            elif self.source_type == "step":
                self.b_src_matrix[:, src_ind] = bf.step_rescale_1D(self.estimation_area, curr_src, self.h)
            elif self.source_type == "gauss_lim":
                self.b_src_matrix[:, src_ind] = bf.gauss_rescale_lim_1D(self.estimation_area, curr_src, self.h)
        #return b_src_matrix    

    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source basis function
        at every position in the interpolated space.
        """
        n_points = len(self.estimation_area)
        dist_max = max(self.estimation_area) - min(self.estimation_area)

        self.b_interp_pot_matrix = np.zeros((self.n_sources, n_points))
    
        for src_ind, current_src in enumerate(self.source_positions):
            for arg_ind, arg in enumerate(self.estimation_area):
                r = abs(current_src - arg)
        
                dist_table_ind = uint16(r * float(self.dist_density)/dist_max)
                self.b_interp_pot_matrix[src_ind, arg_ind] = self.dist_table[dist_table_ind]
        #return b_interp_pot_matrix


    def choose_lambda(self, lambdas, n_folds=1, n_iter=1):
        """
        Finds the optimal regularization parameter lambda for Tikhonov regularization using cross validation.
        """
        n = len(lambdas)
        errors = np.zeros(n)
        errors_iter = np.zeros(n_iter)
        for i, lambd in enumerate(lambdas):
            for j in xrange(n_iter):
                errors_iter[j] = cv.cross_validation(lambd, self.sampled_pots, self.k_pot, 
                                                     self.elec_pos.shape[0], n_folds)
            errors[i] = np.mean(errors_iter)
        return lambdas[errors == min(errors)][0]


if __name__ == '__main__':
    """Example"""
    elec_pos = np.array([0.0, 0.1, 0.4, 0.7, 0.8, 1.0, 1.2, 1.7])
    pots = 0.8 * np.exp(-(elec_pos - 0.1)**2/0.2) + 0.8 * np.exp(-(elec_pos - 0.7)**2/0.1)
    params = {'x_min': -1.0, 'x_max': 2.5, 'R': 1.0}

    k = KCSD1D(elec_pos, pots, params=params)
    k.calculate_matrices()
    print "lambda=", k.choose_lambda(k.lambdas)
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()