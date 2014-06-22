# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi, uint16
from numpy import dot, transpose, identity
from sklearn.cross_validation import KFold, LeaveOneOut, ShuffleSplit
from matplotlib import pylab as plt


class KCSD2D(object):
    """
    2D variant of the kCSD method.
    It assumes constant distribution of sources in a slice around the estimation area.
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
        if len(elec_pos) < 3:
            raise Exception("Number of electrodes must be at least 3!")
        elec_set = set(elec_pos)
        if len(elec_pos) != len(elec_set):
            raise Exception("Error! Duplicate electrode!")

    def set_parameters(self, params):
        if 'sigma' in params.keys():
            self.sigma = params.get('sigma', 1.0)
        else:
            self.sigma = 1.0 
        if 'n_sources' in params.keys():
            self.n_sources = params['n_sources']
        else:
            self.n_sources = 400
        if 'x_max' in params.keys():
            self.xmax = params['x_max']
        else:
            self.xmax = max(self.elec_pos)
        if 'x_min' in params.keys():
            self.xmin = params['x_min']
        else:
            self.xmin = min(self.elec_pos)
        if 'dist_density' in params.keys():
            self.dist_density = params['dist_density']
        else:
            self.dist_density = 200
        if 'lambda' in params.keys():
            self.lambd = params['lambda']
        else:
            self.lambd = 0.0
        if 'source_type' in params.keys():
            if params['source_type'] in ["gaussian", "step"]:
                self.source_type = params['source_type']
            else:
                raise Exception("Incorrect source type!")
        else:    
            self.source_type = "gaussian"
        if 'h' in params.keys():
            self.h = params['h']
        else:
            self.h = abs(self.elec_pos[1] - self.elec_pos[0])
        if 'R' in params.keys():
            self.R = params['R']
        else:
            self.R = 1.0

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
        pass

    #
    # subfunctions
    #

    def calculate_matrices(self):
        """
        Prepares all the required matrices to calculate kCSD.
        """
        self.create_dist_table()

        self.calculate_b_pot_matrix()
        self.k_pot = dot(transpose(self.b_pot_matrix), self.b_pot_matrix)

        self.calculate_b_src_matrix()
        self.k_interp_cross = np.dot(self.b_src_matrix, self.b_pot_matrix)

        self.calculate_b_interp_pot_matrix()
        self.interp_pot = dot(transpose(self.b_interp_pot_matrix), self.b_pot_matrix)
        self.lambdas = np.array([10./j**2 for j in xrange(1, 50)])

    @staticmethod
    def pot_intarg(src, arg, current_pos, h, R, sigma, src_type):
        """
        Returns contribution of a single source as a function of distance
        """
        pass

    @staticmethod
    def gauss_rescale(x, mu, three_stdev):
        """
        Returns normalized gaussian scale function 

        mu -- center of the distribution, 
        three_stdev -- cut off distance from the center
        """
        pass

    @staticmethod
    def step_rescale(x, x0, width):
        """
        Returns normalized step function
        """
        pass

    @staticmethod
    def b_pot_quad(src, arg, h, R, sigma, src_type):
        """
        Returns potential as a function of distance from the source.
        """
        pass

    def create_dist_table(self):
        """
        Create table of a single source contribution to overall potential 
        as a function of distance.
        """
        pass

    def calculate_b_pot_matrix(self):
        """ 
        Compute the matrix of potentials generated by every source basis function
        at every electrode position. 
        """
        pass
  

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        pass

    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source basis function
        at every position in the interpolated space.
        """
        pass

    def calc_CV_error(self, lambd, pot, k_pot, ind_test, ind_train):
        k_train = k_pot[ind_train, ind_train]
    
        pot_train = pot[ind_train]
        pot_test = pot[ind_test]
    
        beta = dot(np.linalg.inv(k_train + lambd * identity(k_train.shape[0])), pot_train)
    
        #k_cross = k_pot[np.array([ind_test, ind_train])]
        k_cross = k_pot[ind_test][:, ind_train]
    
        pot_est = dot(k_cross, beta)

        err = np.linalg.norm(pot_test - pot_est)
        return err

    def cross_validation(self, lambd, pot, k_pot, n_folds):
        """
        Calculate error using LeaveOneOut or KFold cross validation.
        """
        n = len(self.elec_pos)
        errors = []

        #ind_generator = KFold(n, n_folds=n_folds, indices=True)
        ind_generator = LeaveOneOut(n, indices=True)
        #ind_generator = ShuffleSplit(5, n_iter=15, test_size=0.25, indices=True)
        #ind_generator = LeavePOut(len(Y), 2)
        for ind_train, ind_test in ind_generator:
            err = self.calc_CV_error(lambd, pot, k_pot, ind_test, ind_train)
            errors.append(err)

        error = np.mean(errors)
        #print "l=", lambd, ", err=", error
        return error

    def choose_lambda(self, lambdas, n_folds=1, n_iter=1):
        """
        Finds the optimal regularization parameter lambda for Tikhonov regularization using cross validation.
        """
        n = len(lambdas)
        errors = np.zeros(n)
        errors_iter = np.zeros(n_iter)
        for i, lambd in enumerate(lambdas):
            for j in xrange(n_iter):
                errors_iter[j] = self.cross_validation(lambd, self.sampled_pots, self.k_pot, n_folds)
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