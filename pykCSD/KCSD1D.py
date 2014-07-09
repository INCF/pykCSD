# -*- coding: utf-8 -*-
import numpy as np
from numpy import uint16
from numpy import dot, identity

from numpy.linalg import norm, inv

import cross_validation as cv
import basis_functions as bf
import source_distribution as sd
import potentials as pt
import plotting_utils as plut


class KCSD1D(object):
    """
    1D variant of the kCSD method.
    It assumes constant distribution of sources in a cylinder
    around the electrodes.
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
            'R' -- thickness of the basis element
            'h' -- cylinder radius
            'dist_density' -- resolution of the dist_table
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
            raise Exception("Number of measured potentials is not equal\
                             to electrode number!")
        if len(elec_pos) < 2:
            raise Exception("Number of electrodes must be at least 2!")
        elec_set = set(elec_pos)
        if len(elec_pos) != len(elec_set):
            raise Exception("Error! Duplicate electrode!")

    def set_parameters(self, params):
        # assign value from params or default value
        self.sigma = params.get('sigma', 1.0)
        self.n_sources = params.get('n_sources', 300)
        self.xmax = params.get('x_max', np.max(self.elec_pos))
        self.xmin = params.get('x_min', np.min(self.elec_pos))
        self.dist_density = params.get('dist_density', 100)
        self.lambd = params.get('lambda', 0.0)
        self.R_init = params.get('R_init',  2*abs(self.elec_pos[1] - self.elec_pos[0]))
        self.ext = params.get('ext', 0.0)
        self.h = params.get('h', 1.0)
        self.gdX = params.get('gdX', 0.01*(self.xmax - self.xmin))

        self.source_type = params.get('source_type', 'gaussian')
        if self.source_type not in ["gaussian", "step", "gauss_lim"]:
            raise Exception("Incorrect source type!")

        if self.source_type == "step":
            self.basis = bf.step_rescale_1D
        elif self.source_type == "gaussian":
            self.basis = bf.gauss_rescale_1D
        elif self.source_type == "gauss_lim":
            self.basis = bf.gauss_rescale_lim_1D

        self.lambdas = np.array([1.0 / 2**n for n in xrange(0, 20)])
        # space_X is the estimation area
        nx = np.ceil((self.xmax - self.xmin)/self.gdX)
        self.space_X = np.linspace(self.xmin - self.ext, self.xmax + self.ext,
                                   nx)
        (self.X_src, self.R) = sd.make_src_1D(self.space_X, self.ext,
                                              self.n_sources, self.R_init)

        self.X_src += np.min(self.space_X)
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        self.dist_max = Lx

    def estimate_pots(self):
        """Calculates Local Field Potentials."""
        estimation_table = self.interp_pot

        k_inv = inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)
        self.estimated_pots = dot(estimation_table, beta)

        return self.estimated_pots

    def estimate_csd(self):
        """Calculates Current Source Density."""
        estimation_table = self.k_interp_cross

        k_inv = inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
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
                info += '%s : %s\n' % (key, vars(self)[key])
        return info

    def plot_all(self):
        extent = self.space_X
        plut.plot_1D(self.elec_pos, self.sampled_pots, self.estimated_pots,
                     self.estimated_csd, extent)

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
        self.k_interp_cross = dot(self.b_src_matrix, self.b_pot_matrix)

        self.calculate_b_interp_pot_matrix()
        self.interp_pot = dot(self.b_interp_pot_matrix, self.b_pot_matrix)

    def create_dist_table(self):
        """
        Create table of a single source contribution to overall potential
        as a function of distance.
        """
        self.dist_table = np.zeros(self.dist_density)

        for i in xrange(0, self.dist_density):
            pos = (float(i)/self.dist_density) * self.dist_max
            self.dist_table[i] = pt.b_pot_quad(0, pos, self.R, self.h,
                                               self.sigma, self.basis)

    def calculate_b_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every
        source basis function at every electrode position.
        """
        n_obs = self.elec_pos.shape[0]
        nx, = self.X_src.shape
        n = nx

        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        dist_max = Lx

        dt_len = len(self.dist_table)

        self.b_pot_matrix = np.zeros((n, n_obs))

        for i in xrange(0, n):
            # finding the coordinates of the i-th source
            src = self.X_src[i]

            for j in xrange(0, n_obs):
                # for all the observation points
                # checking the distance between the observation point
                # and the source, and calculating the base value
                dist = norm(self.elec_pos[j] - src)
                ind = np.minimum(uint16(np.round(dt_len * dist/dist_max)), dt_len - 1)

                self.b_pot_matrix[i, j] = self.dist_table[ind]

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        ngx = len(self.space_X)
        n = self.n_sources

        self.b_src_matrix = np.zeros((ngx, n))

        for i in xrange(n):
            x_src = self.X_src[i]
            self.b_src_matrix[:, i] = self.basis(self.space_X, x_src, self.R)

    def generated_potential(self, x_src, dist_max, dt_len):
        """
        """
        norms = np.sqrt((self.space_X - x_src)**2)
        ind = np.maximum(0, np.minimum(uint16(np.round(dt_len * norms/dist_max)), dt_len - 1))

        pot = self.dist_table[ind]
        return pot

    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every
        source basis function at every position in the interpolated space.
        """
        self.make_b_interp_pot_matrix_1D()

    def make_b_interp_pot_matrix_1D(self):
        """
        Calculate b_interp_pot_matrix
        """
        dt_len = len(self.dist_table)
        dist_max = self.dist_max

        ngx, = self.space_X.shape
        ng = ngx

        nsx, = self.X_src.shape
        n_src = nsx

        self.b_interp_pot_matrix = np.zeros((ngx, n_src))

        for src in xrange(0, n_src):
            # getting the coordinates of the i-th source
            x_src = self.X_src[src]

            self.b_interp_pot_matrix[:, src] = self.generated_potential(x_src, dist_max, dt_len)

        self.b_interp_pot_matrix = self.b_interp_pot_matrix.reshape(ng, n_src)

    def choose_lambda(self, lambdas, n_folds=1, n_iter=1):
        """
        Finds the optimal regularization parameter lambda
        for Tikhonov regularization using cross validation.
        """
        n = len(lambdas)
        errors = np.zeros(n)
        errors_iter = np.zeros(n_iter)
        for i, lambd in enumerate(lambdas):
            for j in xrange(n_iter):
                errors_iter[j] = cv.cross_validation(lambd, self.sampled_pots,
                                                     self.k_pot,
                                                     self.elec_pos.shape[0],
                                                     n_folds)
            errors[i] = np.mean(errors_iter)
        return lambdas[errors == min(errors)][0]


if __name__ == '__main__':
    """Example"""
    elec_pos = np.array([0.0, 0.1, 0.4, 0.7, 0.8, 1.0, 1.2, 1.7])
    pots = 0.8 * np.exp(-(elec_pos - 0.1)**2/0.2)
    pots += 0.8 * np.exp(-(elec_pos - 0.7)**2/0.1)
    params = {'x_min': -1.0, 'x_max': 2.5}

    k = KCSD1D(elec_pos, pots, params=params)
    k.calculate_matrices()
    print "lambda=", k.choose_lambda(k.lambdas)
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()