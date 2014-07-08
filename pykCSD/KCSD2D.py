# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi, uint16
from numpy import dot, transpose, identity
from numpy.linalg import norm
from matplotlib import pylab as plt
from scipy.interpolate import interp1d
from scipy import integrate
import scipy.spatial.distance as distance 

import cross_validation as cv
import basis_functions as bf
import source_distribution as sd

class KCSD2D(object):
    """
    2D variant of the kCSD method.
    It assumes constant distribution of sources in a slice around 
    the estimation area.
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
            'R' -- 
            'x_min', 'x_max', 'y_min', 'y_max' -- boundaries for CSD estimation space
            'cross_validation' -- type of index generator 
            'lambda' -- regularization parameter for ridge regression
        """
        self.validate_parameters(elec_pos, sampled_pots)
        self.elec_pos = elec_pos
        self.sampled_pots = sampled_pots
        self.set_parameters(params)
 
    def validate_parameters(self, elec_pos, sampled_pots):
        if elec_pos.shape[0] != sampled_pots.shape[0]:
            raise Exception("Number of measured potentials is not equal to electrode number!")
        if elec_pos.shape[0] < 3:
            raise Exception("Number of electrodes must be at least 3!")
        #elec_set = set(elec_pos)
        #if len(elec_pos) != len(elec_set):
        #    raise Exception("Error! Duplicate electrode!")

    def set_parameters(self, params):
        self.sigma = params.get('sigma', 1.0)
        self.n_sources = params.get('n_sources', 300)
        self.xmax = params.get('x_max', np.max(self.elec_pos[:,0]))
        self.xmin = params.get('x_min', np.min(self.elec_pos[:,0]))
        self.ymax = params.get('y_max', np.max(self.elec_pos[:,1]))
        self.ymin = params.get('y_min', np.min(self.elec_pos[:,1]))
        self.lambd = params.get('lambda', 0.0)
        self.R_init = params.get('R_init', 2*distance.pdist(self.elec_pos).min())
        self.h = params.get('h', 1.0)
        self.ext_x = params.get('ext_x', 0.0)
        self.ext_y = params.get('ext_y', 0.0)

        self.dx = params.get('dx', 0.01*(self.xmax - self.xmin))
        self.dy = params.get('dy', 0.01*(self.ymax - self.ymin))
        self.__dist_table_density = 100

        self.source_type = params.get('source_type', 'gaussian')
        if self.source_type not in ["gaussian", "step"]:
            raise Exception("Incorrect source type!")

        self.lambdas = np.array([1.0 / 2**n for n in xrange(0, 20)])

        lin_x = np.linspace(self.xmin, self.xmax, (self.xmax - self.xmin)/self.dx +1 )
        lin_y = np.linspace(self.ymin, self.ymax, (self.ymax - self.ymin)/self.dy +1 )
        self.space_X, self.space_Y = np.meshgrid(lin_x, lin_y)

        (self.X_src, self.Y_src, self.R) = sd.make_src_2D(self.space_X, self.space_Y, self.n_sources, 
                                                          self.ext_x, self.ext_x, self.R_init)
        
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        self.dist_max = (Lx**2 + Ly**2)**0.5

    def estimate_pots(self):
        """Calculates Local Field Potentials."""
        estimation_table = self.interp_pot
    
        k_inv = np.linalg.inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)

        (nx,ny) = self.space_X.shape
        output = np.zeros(nx*ny)

        for i in xrange(self.elec_pos.shape[0]):
            output[:] += beta[i]*estimation_table[:,i]

        self.estimated_pots = output.reshape(nx, ny)
        return self.estimated_pots

    def estimate_csd(self):
        """Calculates Current Source Density."""
        estimation_table = self.k_interp_cross

        k_inv = np.linalg.inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)
        
        (nx,ny) = self.space_X.shape
        output = np.zeros(nx*ny)

        for i in xrange(self.elec_pos.shape[0]):
            output[:] += beta[i]*estimation_table[:,i]

        self.estimated_csd = output.reshape(nx, ny)
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
        extent = [self.xmin, self.xmax, self.ymin, self.ymax]

        fig, (ax11, ax12, ax13) = plt.subplots(1, 3)
        
        ax11.set_xlim([extent[0], extent[1]])
        ax11.set_ylim([extent[2], extent[3]])
        ax11.scatter(x=self.elec_pos[:,0], y=self.elec_pos[:,1],  
                     c=self.sampled_pots, s=200, marker='s')
        ax11.set_title('Measured potentials')

        ax12.imshow(self.estimated_pots.T, interpolation='none', 
                    extent=extent, aspect="auto", origin='lower')
        ax12.set_title('Calculated potentials')
        ax12.autoscale_view(True,True,True)

        ax13.imshow(self.estimated_csd.T, interpolation='none', 
                    extent=extent, aspect="auto", origin='lower')
        ax13.set_title('Calculated CSD')
        ax13.autoscale_view(True,True,True)
        
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
        self.k_pot = dot(transpose(self.b_pot_matrix), self.b_pot_matrix)

        self.calculate_b_src_matrix()
        self.k_interp_cross = np.dot(self.b_src_matrix, self.b_pot_matrix)
        
        self.calculate_b_interp_pot_matrix()
        self.interp_pot = dot(self.b_interp_pot_matrix, self.b_pot_matrix)


    @staticmethod
    def int_pot(xp, yp, x, R, h, src_type):
        """INPUT
        xp,yp    - coordinates of some point laying in the support of a 
               - basis element centered at (0,0)
        x        - x - coordinate of a point (x,0) at which we calculate the
             - potential
        R        - radius of the basis element
        h
        src_type - type of basis function in the source space
               (step/gauss/gauss_lim)
        OUTPUT
        int_pot - contribution of a point xp,yp, belonging to a basis source
                - support centered at (0,0) to the potential measured at (x,0)
                - , integrated over xp,yp gives the potential generated by a 
                - basis source element centered at (0,0) at point (x,0)  
        """
        y = ((x-xp)**2 + yp**2)**(0.5)
        if y < 0.00001:
            y = 0.00001
        y = np.arcsinh(h/y)
        if src_type == 'step':
            y *= bf.step_rescale_2D(xp, yp, R)
        elif src_type == 'gaussian':
            y *= bf.gauss_rescale_2D(xp, yp, [0,0], R)
        elif src_type == 'gauss_lim':
            y *= bf.gauss_rescale_2D_lim(xp, yp, [0,0], R)
        return y

    @staticmethod
    def b_pot_2d_cont(x, R, h, sigma, src_type):
        """
        Returns the value of the potential at point (x,0) generated
        by a basis source located at (0,0)
        """
        pot, err = integrate.dblquad(KCSD2D.int_pot, -R, R, lambda x:-R, lambda x:R, 
                                     args=(x,R,h,src_type), epsrel=1e0, epsabs=1e0)
        pot *= 1./(2.0*pi*sigma)
        return pot

    def create_dist_table(self):
        """
        Create table of a single source base element contribution 
        to overall potential as a function of distance.
        The last record corresponds to the distance equal to the
        diagonal of the grid.
        """

        dense_step = 3
        denser_step = 1
        sparse_step = 9
        border1 = 0.9*self.R/self.dist_max * self.__dist_table_density
        border2 = 1.3*self.R/self.dist_max * self.__dist_table_density
    
        xs = np.arange( 0,  border1, dense_step )
        xs = np.append( xs, border1 )
        zz = np.arange( (border1 + denser_step), border2, dense_step )

        xs = np.concatenate( (xs,zz) )
        xs = np.append( xs, [border2, (border2+denser_step)] )
        xs = np.concatenate( (xs, np.arange((border2 + denser_step + sparse_step/2.), 
                              self.__dist_table_density, sparse_step)) )
        xs = np.append( xs, self.__dist_table_density + 1)
    
        xs = np.unique(np.array(xs))

        dist_table = np.zeros(len(xs))

        for i, x in enumerate(xs):
            dist_table[i] = KCSD2D.b_pot_2d_cont((x/self.__dist_table_density) * self.dist_max,
                                                self.R, self.h, self.sigma, self.source_type)
        #print "dt: ", dist_table
        inter = interp1d(x=xs, y=dist_table, kind='cubic', fill_value=0.0)
        dt_int = np.array([inter(xx) for xx in xrange(self.__dist_table_density)])
        dt_int.flatten()

        self.dist_table = dt_int.copy()
        #return dt_int

    def calculate_b_pot_matrix(self):
        """ 
        Compute the matrix of potentials generated by every source basis function
        at every electrode position. 
        """
        self.calculate_b_pot_matrix_2D()

    def calculate_b_pot_matrix_2D(self):
        """ INPUT 
        X,Y        - grid of points at which we want to calculate CSD 
        Pot        - Vector of potentials containing electrode positions and values
                (calculated with 'make_pot')
        nsx,nsy    - number of base elements in the x and y direction 
        dist_table - vector calculated with 'create_dist_table'
        R -        - radius of the support of the basis functions
    
        OUTPUT
        b_pot_matrix - matrix containing containing the values of all
                   the potential basis functions in all the electrode
                    positions (essential for calculating the cross_matrix)
        """
        n_obs = self.elec_pos.shape[0]
        (nx, ny) = self.X_src.shape
        n = nx * ny
 
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        dist_max = (Lx**2 + Ly**2)**0.5
    
        dt_len = len(self.dist_table)
    
        self.b_pot_matrix = np.zeros((n, n_obs))
    
        for i in xrange(0, n):
            #finding the coordinates of the i-th source
            i_x, i_y = np.unravel_index(i, (nx,ny))
            src = [self.X_src[i_x, i_y], self.Y_src[i_x, i_y]]

            for j in xrange(0, n_obs): 
                # for all the observation points
                # checking the distance between the observation point and the source,
                # calculating the base value
                dist = norm(self.elec_pos[j] - src)
      
                ind = np.minimum( uint16(np.round(dt_len * dist/dist_max)), dt_len-1)

                self.b_pot_matrix[i,j] = self.dist_table[ind]

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        self.make_b_src_matrix_2D()

    def make_b_src_matrix_2D(self):
        """
        Calculate b_src_matrix - matrix containing containing the values of all
        the source basis functions in all the points at which we want to 
        calculate the solution (essential for calculating the cross_matrix)
        """
        R2 = self.R**2
    
        (nsx, nsy) = self.X_src.shape
        n = nsy * nsx  #total number of sources    
        (ngx, ngy) = self.space_X.shape
        ng = ngx * ngy

        self.b_src_matrix = np.zeros((self.space_X.shape[0], self.space_X.shape[1], n))
    
        for i in xrange(n): 
            #getting the coordinates of the i-th source
            (i_x, i_y) = np.unravel_index(i, (nsx,nsy), order='F')
            y_src = self.Y_src[i_x, i_y]
            x_src = self.X_src[i_x, i_y]
        
            if self.source_type == 'step':
                self.b_src_matrix[:,:,i]= bf.step_rescale_2D((self.space_X - x_src), (self.space_Y - y_src), R2)
            elif self.source_type == 'gaussian':
                self.b_src_matrix[:,:,i] = bf.gauss_rescale_2D(self.space_X, self.space_Y, [x_src,y_src], self.R)
            elif self.source_type == 'gauss_lim':
                self.b_src_matrix[:,:,i] = bf.gauss_rescale_2D_lim(self.space_X, self.space_Y, [x_src,y_src], self.R)            

        self.b_src_matrix = self.b_src_matrix.reshape(ng,n)


    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source basis function
        at every position in the interpolated space.
        """
        self.make_b_interp_pot_matrix_2D()

    def generated_potential(self, x_src, y_src,  dist_max, dt_len):
        """
        """
        norms = np.sqrt((self.space_X - x_src)**2 + (self.space_Y - y_src)**2)
        ind = np.maximum(0, np.minimum( uint16(np.round(dt_len * norms/dist_max)), dt_len-1))

        pot = self.dist_table[ind]
        return pot


    def make_b_interp_pot_matrix_2D(self):
        """
        Calculate b_interp_pot_matrix
        """
        dt_len = len(self.dist_table)
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        dist_max = np.sqrt(Lx**2 + Ly**2)

        (ngx, ngy) = self.space_X.shape
        ng = ngx * ngy
        (nsx, nsy) = self.X_src.shape
        n_src = nsy * nsx

        self.b_interp_pot_matrix = np.zeros((ngx, ngy, n_src))
    
        for src in xrange(0, n_src): 
            #getting the coordinates of the i-th source
            (i_x, i_y) = np.unravel_index(src, (nsx,nsy), order='F')
            y_src = self.Y_src[i_x, i_y]
            x_src = self.X_src[i_x, i_y]       
        
            self.b_interp_pot_matrix[:, :, src] = self.generated_potential(x_src, y_src, dist_max, dt_len)
   
        self.b_interp_pot_matrix = self.b_interp_pot_matrix.reshape(ng, n_src)


    def choose_lambda(self, lambdas, n_folds=1, n_iter=1):
        """
        Finds the optimal regularization parameter lambda for Tikhonov regularization using cross validation.
        """
        n = len(lambdas)
        errors = np.zeros(n)
        errors_iter = np.zeros(n_iter)
        for i, lambd in enumerate(lambdas):
            for j in xrange(n_iter):
                errors_iter[j] = cv.cross_validation(lambd, self.sampled_pots, 
                                                     self.k_pot,
                                                     elec_pos.shape[0], n_folds)
            errors[i] = np.mean(errors_iter)
        return lambdas[errors == min(errors)][0]


if __name__ == '__main__':
    elec_pos = np.array([[0,0], [0,1], [1,1]])
    pots = np.array([0,1,2])
    k = KCSD2D(elec_pos, pots)
    k.calculate_matrices()
    k.estimate_pots()
    k.estimate_csd()
    k.plot_all()