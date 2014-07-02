# -*- coding: utf-8 -*-
import numpy as np
from numpy import pi, uint16
from numpy import dot, transpose, identity
from matplotlib import pylab as plt
from scipy.interpolate import interp1d
from scipy import integrate
from numpy.linalg import norm
import cross_validation as cv


class KCSD3D(object):
    """
    3D variant of the kCSD method.
    It assumes sources are distributed in 3D space.
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
        if elec_pos.shape[0] < 4:
            raise Exception("Number of electrodes must be at least 4!")

    def set_parameters(self, params):
        self.sigma = params.get('sigma', 1.0)
        self.n_sources = params.get('n_sources', 300)
        self.xmax = params.get('x_max', np.max(self.elec_pos[:,0]))
        self.xmin = params.get('x_min', np.min(self.elec_pos[:,0]))
        self.ymax = params.get('y_max', np.max(self.elec_pos[:,1]))
        self.ymin = params.get('y_min', np.min(self.elec_pos[:,1]))
        self.zmax = params.get('z_max', np.max(self.elec_pos[:,2]))
        self.zmin = params.get('z_min', np.min(self.elec_pos[:,2]))

        self.lambd = params.get('lambda', 0.0)
        self.R_init = params.get('R_init', 2*KCSD3D.calc_min_dist(self.elec_pos))
        self.h = params.get('h', 1.0)
        self.ext_X = params.get('ext_X', 0.0)
        self.ext_Y = params.get('ext_Y', 0.0)
        self.ext_Z = params.get('ext_Z', 0.0)

        self.gdX = params.get('gdX', 0.01*(self.xmax - self.xmin))
        self.gdY = params.get('gdY', 0.01*(self.ymax - self.ymin))
        self.gdZ = params.get('gdZ', 0.01*(self.zmax - self.zmin))
        self.__dist_table_density = 100

        if 'source_type' in params:
            if params['source_type'] in ["gaussian", "step"]:
                self.source_type = params['source_type']
            else:
                raise Exception("Incorrect source type!")
        else:    
            self.source_type = "gaussian"

        self.lambdas = np.array([1.0 / 2**n for n in xrange(0, 20)])

        lin_x = np.linspace(self.xmin, self.xmax, (self.xmax - self.xmin)/self.gdX +1 )
        lin_y = np.linspace(self.ymin, self.ymax, (self.ymax - self.ymin)/self.gdY +1 )
        lin_z = np.linspace(self.zmin, self.ymax, (self.zmax - self.zmin)/self.gdZ +1 )
        self.space_X, self.space_Y, self.space_Z = np.meshgrid(lin_x, lin_y, lin_z)

        [self.X_src, self.Y_src, self.Z_src, _, _, _, self.R] = KCSD3D.make_src_3D(self.space_X, self.space_Y, self.space_Z,
                                                                            self.n_sources, self.ext_X, self.ext_Y, self.ext_Z, self.R_init)
        
        Lx = np.max(self.X_src) - np.min(self.X_src) + self.R
        Ly = np.max(self.Y_src) - np.min(self.Y_src) + self.R
        Lz = np.max(self.Z_src) - np.min(self.Z_src) + self.R
        self.dist_max = np.sqrt(Lx**2 + Ly**2 + Lz**2)

    def estimate_pots(self):
        """Calculates Local Field Potentials."""
        estimation_table = self.interp_pot
    
        k_inv = np.linalg.inv(self.k_pot + self.lambd * identity(self.k_pot.shape[0]))
        beta = dot(k_inv, self.sampled_pots)

        nx,ny,nz = self.space_X.shape
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
        
        nx,ny,nz = self.space_X.shape
        output = np.zeros(nx*ny)

        for i in xrange(self.elec_pos.shape[0]):
            output[:] += beta[i]*estimation_table[:,i]

        self.estimated_csd = output.reshape(nx, ny, nz)
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
        self.interp_pot = dot(self.b_interp_pot_matrix, self.b_pot_matrix)
        

    @staticmethod
    def calc_min_dist(elec_pos):
        n = elec_pos.shape[0]
        min_dist = norm(elec_pos[0, :] - elec_pos[1, :])
        for i in xrange(0, n):
            for j in xrange(0, n):
                dist = norm(elec_pos[i, :] - elec_pos[j, :])
                if dist < min_dist and i!=j:
                    min_dist = norm(elec_pos[i, :] - elec_pos[j, :])
        return min_dist

    @staticmethod
    def get_src_params_3D(Lx, Ly, Lz, n_src):
        """
        helps uniformally distribute n_src sources among an area of size Lx x Ly x Lz

        INPUT
        Lx,Ly,Lz    - lengths in the directions x,y of the area, ...
                 the sources should be placed
        n_src     - number of sources

        OUTPUT
        nx,ny,nz     - number of sources in directions x,y
        ds        - spacing between the sources
        """
        V = Lx*Ly*Lz
        #print V
        V_unit = V/n_src
        nx = np.floor(Lx*V_unit**(-1./3)) + 1
        ny = np.floor(Ly*V_unit**(-1./3)) + 1
        nz = np.floor(Lz*V_unit**(-1./3)) + 1
    
        ds = Lx/(nx-1)
    
        Lx_n = (nx-1)*ds
        Ly_n = (ny-1)*ds
        Lz_n = (nz-1)*ds
    
        return (nx, ny, nz,  Lx_n, Ly_n, Lz_n, ds)

    @staticmethod
    def make_src_3D(X, Y, Z, n_src, ext_x, ext_y, ext_z, R_init):
        """                               
        INPUT
        X,Y,Z                 - Points at which CSD will be estimated
        n_src               - number of sources we want to include in the model
        ext_x,ext_y,ext_z        - how should the sources extend the area X,Y,Z
        R_init              - demanded radius of the basis element

        OUTPUT
        X_src, Y_src, Z_src       - Positions of the sources
        nx,ny,nz             - number of sources in directions x,y,z
        R                 - effective radius of the basis element 
        """
        Lx = np.max(X) - np.min(X)
        Ly = np.max(Y) - np.min(Y)
        Lz = np.max(Z) - np.min(Z)

        Lx_n = Lx + 2*ext_x 
        Ly_n = Ly + 2*ext_y
        Lz_n = Lz + 2*ext_z

        (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n, Ly_n, Lz_n, n_src)
        # print (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds)

        ext_x_n = (Lx_nn - Lx)/2
        ext_y_n = (Ly_nn - Ly)/2
        ext_z_n = (Lz_nn - Lz)/2

        X_src, Y_src, Z_src = np.meshgrid(np.linspace(-ext_x_n, Lx+ext_x_n, (Lx+2*ext_x_n)/ds + 1), 
                                          np.linspace(-ext_y_n, Ly+ext_y_n, (Ly+2*ext_y_n)/ds + 1),
                                          np.linspace(-ext_z_n, Lz+ext_z_n, (Lz+2*ext_z_n)/ds + 1))

        d = np.round(R_init/ds)
        R = d * ds

        return X_src, Y_src, Z_src, nx, ny, nz, R


    @staticmethod
    def int_pot(xp, yp, zp, x, R, h, src_type):
        """INPUT
        xp,yp,zp    - coordinates of some point laying in the support of a 
               - basis element centered at (0,0,0)
        x,y        - coordinates of a point (x,0,0) at which we calculate the
             - potential
        R        - radius of the basis element
        h
        src_type - type of basis function in the source space
               (step/gauss/gauss_lim)
        OUTPUT
        int_pot - contribution of a point xp,yp,zp belonging to a basis source
                - support centered at (0,0,0) to the potential measured at (x,0,0)
                - integrated over xp, yp, zp gives the potential generated by a 
                - basis source element centered at (0,0,0) at point (x,0,0)  
        """
        y = np.sqrt((x-xp)**2 + yp**2 + zp**2)
        if y < 0.00001:
            y = 0.00001
        y = 1.0/y
        if src_type == 'step':
            y *= step_rescale(xp, yp, zp, R)
        elif src_type == 'gaussian':
            y *= KCSD3D.gauss_rescale_3D(xp, yp, zp, [0,0,0], R);
        elif src_type == 'gauss_lim':
            y *= gauss_rescale_lim(xp, yp, [0,0,0], R);
        return y

    @staticmethod
    def gauss_rescale_3D(x, y, z, mu, three_stdev):
        """
        Returns normalized gaussian 2D scale function 

        x, y        -- coordinates a point at which we calculate the density 
        mu          -- distribution mean vector
        three_stdev -- 3 * standard deviation of the distribution
        """
        h = 1./(2*pi)
        stdev = three_stdev/3.0
        inv_std = 1.0/stdev
        h_n = h * stdev/1
        Z = h_n * np.exp ( -inv_std**2 * 0.5 * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2 ) )
        return Z


    @staticmethod
    def step_rescale(xp, yp, zp, R):
        """
        Returns normalized 3D step function
        """
        s = (xp**2 + yp**2 + zp**2 <= R**2)
        return s

    @staticmethod
    def b_pot_3d_cont(x, R, h, sigma, src_type):
        """
        Returns the value of the potential at point (x,y,0) generated
        by a basis source located at (0,0,0)
        """
        #pot, err = integrate.nquad(int_pot, [[-R, R],[-R,R], [-R, R]], args=(x,R,h,src_type))
        pot, err = integrate.tplquad(int_pot, -R, R, 
                                     lambda x:-R, lambda x:R, 
                                     lambda x,y:-R, lambda x,y:R,
                                     args=(x,R,h,src_type))
        #print err
        pot *= 1./(4.0*pi*sigma)
        return pot

    def create_dist_table(self):
        """
        Create table of a single source base element contribution 
        to overall potential as a function of distance.
        The last record corresponds to the distance equal to the
        diagonal of the cuboid.
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
            dist_table[i] = KCSD3D.b_pot_3d_cont((x/self.__dist_table_density) * self.dist_max,
                                                self.R, self.h, self.sigma, self.source_type)
        #print "dt: ", dist_table
        inter = interp1d(x=xs, y=dist_table, kind='cubic', fill_value=0.0)
        dt_int = np.array([inter(xx) for xx in xrange(self.__dist_table_density)])
        dt_int.flatten()

        self.dist_table = dt_int.copy()

    def calculate_b_pot_matrix(self):
        """ 
        Compute the matrix of potentials generated by every source basis function
        at every electrode position. 
        """
        self.calculate_b_pot_matrix_3D()

    def calculate_b_pot_matrix_3D(self):
        """ INPUT 
        X,Y,Z        - grid of points at which we want to calculate CSD 
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
        # TODO
        pass

    def calculate_b_src_matrix(self):
        """
        Compute the matrix of basis sources.
        """
        self.make_b_src_matrix_3D()

    def make_b_src_matrix_3D(self):
        """
        Calculate b_src_matrix - matrix containing containing the values of all
        the source basis functions in all the points at which we want to 
        calculate the solution (essential for calculating the cross_matrix)
        """
        pass


    def calculate_b_interp_pot_matrix(self):
        """
        Compute the matrix of potentials generated by every source basis function
        at every position in the interpolated space.
        """
        self.make_b_interp_pot_matrix_3D()

    def generated_potential(self, x_src, y_src, z_src,  dist_max, dt_len):
        """
        """
        # TODO!
        pass


    def make_b_interp_pot_matrix_3D(self):
        """
        Calculate b_interp_pot_matrix
        """
        #TODO!
        pass


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
    pass