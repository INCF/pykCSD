# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi, uint16
from numpy import dot, transpose, identity
from sklearn.cross_validation import KFold, LeaveOneOut, ShuffleSplit
from pylab import *

class KCSD1D(object):
    """
    1D variant of the kCSD method. It assumes constant distribution of sources in a cylinder around the electrodes.
    """

    def __init__(self, elec_pos, sampled_pots, params = {}):
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
        if len(elec_pos) != len(sampled_pots):
            raise Exception("Number of measured potentials is not equal to electrode number!")
        if elec_pos < 2:
            raise Exception("Number of electrodes must be at least 2!")

        self.elec_pos = elec_pos
        self.sampled_pots = sampled_pots
        self.set_parameters(params)
 
    def set_parameters(self, params):
        if 'sigma' in params.keys():
            self.sigma = params['sigma']
        else:
            self.sigma = 1.0 
        if 'n_sources' in params.keys():
            self.n_sources = params['n_sources']
        else:
            self.n_sources = 200
        if 'x_max' in params.keys():
            self.xmax = params['x_max']
        else:
            self.xmax = 1.0
        if 'x_min' in params.keys():
            self.xmin = params['x_min']
        else:
            self.xmin = 0.0
        if 'dist_density' in params.keys():
            self.dist_density = params['dist_density']
        else:
            self.dist_density = 300
        if 'lambda' in params.keys():
            self.lambd = params['lambda']
        else:
            self.lambd = 0.0
        if 'source_type' in params.keys():
            self.source_type = params['source_type']
        else:    
            self.source_type = "gaussian"
        if 'h' in params.keys():
            self.h = params['h']
        else:
            self.h = self.elec_pos[1] - self.elec_pos[0]
        if 'R' in params.keys():
            self.R = params['R']
        else:
            self.R = 0.2*(self.elec_pos[1] - self.elec_pos[0])

        self.__source_space = np.linspace(min(self.elec_pos), max(self.elec_pos), self.n_sources)
        self.__estimation_area = np.linspace(self.xmin, self.xmax, self.n_sources)
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


    def save(self, filename = 'result'):
        """Save results to file."""
        pass

    def __repr__(self):
        print self.__class__.__name__
        for key in vars(self).keys():
            if not key.startswith('_'):
                print key, ': ', vars(self)[key]
        return ''

    def plot_all(self):
        fig, (ax11, ax21, ax22) = plt.subplots(1, 3, sharex=True)
        
        ax11.scatter(self.elec_pos, self.sampled_pots)
        ax11.set_title('Measured potentials')

        ax21.plot(self.__estimation_area, self.estimated_pots)
        ax21.set_title('Calculated potentials')

        ax22.plot(self.__estimation_area, self.estimated_csd)
        ax22.set_title('Calculated CSD')
        show()
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
        self.__lambdas = np.array([10./j**2 for j in xrange(1,50) ])


    @staticmethod
    def pot_intarg(src, arg, current_pos, h, R, sigma, src_type):
        """

        """
        if src_type == "gaussian":
            y = (1./(2*sigma))*(np.sqrt((arg-current_pos)**2 + R**2) - abs(arg - current_pos)) * KCSD1D.gauss_rescale(src, current_pos, h)
            # for this formula look at formula (8) from Pettersen et al., 2006
        if src_type == "step":
            #TODO check/re-evaluate
            y = (1./(2*sigma))*(np.sqrt((arg-current_pos)**2+R**2) - abs(arg-current_pos)) * (abs(src-current_pos) < h)   
        return y


    @staticmethod
    def gauss_rescale(x, mu, three_stdev):
        """
        Returns normalized gaussian scale function 

        mu -- center of the distribution, 
        three_stdev -- cut off distance from the center
        """
        variance = (three_stdev/3.0)**2
        g = 1./np.sqrt(2. * pi * variance) * np.exp( -(1./(2.*variance)) * (x-mu)**2 ) * (abs(x-mu)<three_stdev)
        return g

    @staticmethod
    def b_pot_quad(src, arg, h, R, sigma, src_type):
        """
        Returns potential as a function of distance from the source.
        """
        if src_type in ["gaussian", "step"]:
            x = np.linspace(src - 4*h, src + 4*h, 150) #manipulate the resolution
            y = [KCSD1D.pot_intarg(src, arg, current_pos, h, R, sigma, src_type) for current_pos in x]
            #plot(x,y)
            z = np.trapz(y, x)
        else:
            raise Exception("Source type not implemented!")

        return z


    def create_dist_table(self):
        """
        Create table of a single source contribution to overall potential as a function of distance
        """
        self.dist_table = np.zeros(self.dist_density+1)

        for i in xrange(0, self.dist_density+1):
            self.dist_table[i] = KCSD1D.b_pot_quad(0,(float(i)/self.dist_density) * self.dist_max, 
                                            self.h, self.R, self.sigma, self.source_type)
        #return dist_table


    def calculate_b_pot_matrix(self):
        """ 
        Compute the matrix of potentials generated by every source basis function
        at every electrode position. 
        """
        n_elec = len(self.elec_pos)
    
        self.b_pot_matrix = np.zeros((self.n_sources, n_elec))
    
        for src_ind, current_src in enumerate(self.__source_space):
            for arg_ind, arg in enumerate(self.elec_pos):
                r = abs(current_src - arg)
            
                #dist_table_ind = uint16(0.85 * r * float(dist_dens)/max_dist)
                dist_table_ind = uint16(r * float(self.dist_density)/self.dist_max)
                #print dist_table_ind
                self.b_pot_matrix[src_ind, arg_ind] = self.dist_table[dist_table_ind]   
        #return b_pot_matrix
  

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        n_gx = len(self.__estimation_area)

        self.b_src_matrix = np.zeros((n_gx, self.n_sources))

        for src_ind, curr_src in enumerate(self.__source_space):
            self.b_src_matrix[:, src_ind] = KCSD1D.gauss_rescale(self.__estimation_area, curr_src, self.h)

        #return b_src_matrix
    

    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source basis function
        at every position in the interpolated space.
        """
        n_points = len(self.__estimation_area)
        dist_max = max(self.__estimation_area) - min(self.__estimation_area)

        self.b_interp_pot_matrix = np.zeros((self.n_sources, n_points))
    
        for src_ind, current_src in enumerate(self.__source_space):
            for arg_ind, arg in enumerate(self.__estimation_area):
                r = abs(current_src - arg)
        
                #dist_table_ind = uint16(0.85*r*float(dist_dens)/dist_max)   
                dist_table_ind = uint16(r*float(self.dist_density)/dist_max)
                self.b_interp_pot_matrix[src_ind, arg_ind] = self.dist_table[dist_table_ind]
    
        #return b_interp_pot_matrix

    def calc_CV_error(self, lambd, pot, k_pot, ind_test, ind_train):
        k_train = k_pot[ind_train, ind_train]
    
        pot_train = pot[ind_train]
        pot_test = pot[ind_test]
    
        beta = dot(np.linalg.inv(k_train + lambd*identity(k_train.shape[0])) , pot_train)
    
        #k_cross = k_pot[np.array([ind_test, ind_train])]
        k_cross = k_pot[ind_test][:,ind_train]
    
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

    def choose_lambda(self, __lambdas, n_folds=1, n_iter=1):
        """
        Finds the optimal regularization parameter lambda for Tikhonov regularization by cross validation.
        """
        n = len(__lambdas)
        errors = np.zeros(n)
        errors_iter = np.zeros(n_iter)
        for i, lambd in enumerate(__lambdas):
            for j in xrange(n_iter):
                errors_iter[j] = self.cross_validation(lambd, self.sampled_pots, self.k_pot, n_folds)
            errors[i] = np.mean(errors_iter)
        return __lambdas[errors == min(errors)][0]


if __name__ == '__main__':
    """Example"""
    elec_pos = np.array([0.1, 0.7])
    pots = 0.8 * np.exp(-(elec_pos - 0.1)**2) - 0.8 * np.exp(-(elec_pos - 0.7)**2)
        
    k = KCSD1D(elec_pos, pots)
    k.calculate_matrices()
    print "lambda=", k.choose_lambda(k.__lambdas)
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()