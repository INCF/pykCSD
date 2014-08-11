# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from numpy import dot, identity
from numpy.linalg import norm, inv

import basis_functions as bf
import source_distribution as sd
import potentials as pt
import dist_table_utils as dt
import plotting_utils as plut
import parameters_utils as parut


class KCSD2D(object):
    """
    2D variant of solver for the Kernel Current Source Density method.

    It assumes constant distribution of sources in a slice around
    the estimation area.

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
            cylinder radius

        'dist_density' : int
            resolution of the dist_table

        'x_min', 'x_max', 'y_min', 'y_max' : floats
            boundaries for CSD estimation space

        'ext' : float
            length of space extension: x_min-ext ... x_max+ext

        'gdX', 'gdY' : float
            space increments in the estimation space

        'lambd' : float
            regularization parameter for ridge regression
    """

    def __init__(self, elec_pos, sampled_pots, params={}):
        self.validate_parameters(elec_pos, sampled_pots)
        self.elec_pos = elec_pos
        self.sampled_pots = sampled_pots
        self.set_parameters(params)

    def validate_parameters(self, elec_pos, sampled_pots):
        if elec_pos.shape[0] != sampled_pots.shape[0]:
            raise Exception("Number of measured potentials is not equal\
                             to electrode number!")
        if elec_pos.shape[0] < 3:
            raise Exception("Number of electrodes must be at least 3!")
        if parut.check_for_duplicated_electrodes(elec_pos) is False:
            raise Exception("Error! Duplicated electrode!")

    def set_parameters(self, params):
        default_params = {
            'sigma': 1.0,
            'n_sources': 300,
            'xmin': np.min(self.elec_pos[:, 0]),
            'xmax': np.max(self.elec_pos[:, 0]),
            'ymin': np.min(self.elec_pos[:, 1]),
            'ymax': np.max(self.elec_pos[:, 1]),
            'dist_table_density': 100,
            'lambd': 0.0,
            'R_init': 2 * parut.min_dist(self.elec_pos),
            'ext_x': 0.0,
            'ext_y': 0.0,
            'h': 1.0,
            'source_type': 'gauss'
        }
        for (prop, default) in default_params.iteritems():
            setattr(self, prop, params.get(prop, default))

        self.gdX = params.get('gdX', 0.01 * (self.xmax - self.xmin))
        self.gdY = params.get('gdY', 0.01 * (self.ymax - self.ymin))

        if self.source_type not in ["gauss", "step"]:
            raise Exception("Incorrect source type!")

        basis_types = {
            "step": bf.step_rescale_2D,
            "gauss": bf.gauss_rescale_2D,
            "gauss_lim": bf.gauss_rescale_lim_2D,
        }
        if self.source_type not in basis_types.keys():
            raise Exception("Incorrect source type!")
        else:
            self.basis = basis_types.get(self.source_type)

        nx = (self.xmax - self.xmin)/self.gdX + 1
        ny = (self.ymax - self.ymin)/self.gdY + 1

        lin_x = np.linspace(self.xmin, self.xmax, nx)
        lin_y = np.linspace(self.ymin, self.ymax, ny)
        self.space_X, self.space_Y = np.meshgrid(lin_x, lin_y)

        (self.X_src, self.Y_src, self.R) = sd.make_src_2D(
            self.space_X,
            self.space_Y,
            self.n_sources,
            self.ext_x,
            self.ext_y,
            self.R_init
        )

        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        self.dist_max = (Lx**2 + Ly**2)**0.5

    def estimate_pots(self):
        """Calculates Local Field Potentials."""
        estimation_table = self.interp_pot
        self.estimated_pots = self.estimate(estimation_table)
        return self.estimated_pots

    def estimate_csd(self):
        """Calculates Current Source Density."""
        estimation_table = self.k_interp_cross
        self.estimated_csd = self.estimate(estimation_table)
        return self.estimated_csd

    def estimate(self, estimation_table):
        k_inv = inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        nt = self.sampled_pots.shape[1]
        (nx, ny) = self.space_X.shape
        estimation = np.zeros((nx * ny, nt))

        for t in xrange(nt):
            beta = dot(k_inv, self.sampled_pots[:, t])
            for i in xrange(self.elec_pos.shape[0]):
                estimation[:, t] += beta[i] * estimation_table[:, i]

        estimation = estimation.reshape(nx, ny, nt)
        return estimation


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
        extent = [self.xmin, self.xmax, self.ymin, self.ymax]
        plut.plot_2D(self.elec_pos, self.sampled_pots, self.estimated_pots,
                     self.estimated_csd, extent)

    #
    # subfunctions
    #

    def init_model(self):
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
        Create table of a single source base element contribution
        to overall potential as a function of distance.
        The last record corresponds to the distance equal to the
        diagonal of the grid.
        """
        self.dist_table = dt.create_dist_table(self.basis,
                                               pt.b_pot_2d_cont,
                                               self.R,
                                               self.h,
                                               self.sigma,
                                               self.dist_max,
                                               self.dist_table_density
                                               )

    def calculate_b_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every
        source basis function at every electrode position.
        """
        self.calculate_b_pot_matrix_2D()

    def calculate_b_pot_matrix_2D(self):
        """
        Calculates b_pot_matrix - matrix containing the values of all
        the potential basis functions in all the electrode positions
        (essential for calculating the cross_matrix).
        """
        n_obs = self.elec_pos.shape[0]
        (nx, ny) = self.X_src.shape
        n = nx * ny

        self.b_pot_matrix = np.zeros((n, n_obs))

        for i in xrange(0, n):
            # finding the coordinates of the i-th source
            i_x, i_y = np.unravel_index(i, (nx, ny))
            src = [self.X_src[i_x, i_y], self.Y_src[i_x, i_y]]

            for j in xrange(0, n_obs):
                # for all the observation points
                # checking the distance between the observation point and
                # the source,
                # calculating the base value
                dist = norm(self.elec_pos[j] - src)

                self.b_pot_matrix[i, j] = dt.generated_potential(
                    dist,
                    self.dist_max,
                    self.dist_table
                )

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        self.make_b_src_matrix_2D()

    def make_b_src_matrix_2D(self):
        """
        Calculate b_src_matrix - matrix containing containing the values of
        all the source basis functions in all the points at which we want to
        calculate the solution (essential for calculating the cross_matrix)
        """
        (nsx, nsy) = self.X_src.shape
        n = nsy * nsx  # total number of sources
        (ngx, ngy) = self.space_X.shape
        ng = ngx * ngy

        self.b_src_matrix = np.zeros((ngx, ngy, n))

        for i in xrange(n):
            # getting the coordinates of the i-th source
            (i_x, i_y) = np.unravel_index(i, (nsx, nsy), order='F')
            y_src = self.Y_src[i_x, i_y]
            x_src = self.X_src[i_x, i_y]

            self.b_src_matrix[:, :, i] = self.basis(self.space_X,
                                                    self.space_Y,
                                                    [x_src, y_src],
                                                    self.R)

        self.b_src_matrix = self.b_src_matrix.reshape(ng, n)

    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source
        basis function at every position in the interpolated space.
        """
        self.make_b_interp_pot_matrix_2D()

    def make_b_interp_pot_matrix_2D(self):
        """
        Calculate b_interp_pot_matrix
        """
        (ngx, ngy) = self.space_X.shape
        ng = ngx * ngy
        (nsx, nsy) = self.X_src.shape
        n_src = nsy * nsx

        self.b_interp_pot_matrix = np.zeros((ngx, ngy, n_src))

        for i in xrange(0, n_src):
            # getting the coordinates of the i-th source
            (i_x, i_y) = np.unravel_index(i, (nsx, nsy), order='F')
            y_src = self.Y_src[i_x, i_y]
            x_src = self.X_src[i_x, i_y]
            norms = np.sqrt((self.space_X - x_src)**2
                            + (self.space_Y - y_src)**2)

            self.b_interp_pot_matrix[:, :, i] = dt.generated_potential(
                norms,
                self.dist_max,
                self.dist_table
            )

        self.b_interp_pot_matrix = self.b_interp_pot_matrix.reshape(ng, n_src)


def main():
    elec_pos = np.array([[0, 0], [0, 1], [1, 1]])
    pots = np.array([[0], [1], [2]])
    k = KCSD2D(elec_pos, pots)
    k.init_model()
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()


if __name__ == '__main__':
    main()