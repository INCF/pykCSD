import numpy as np
from numpy import pi



def gauss_rescale_1D(x, mu, three_stdev):
    """
    Returns normalized gaussian scale function 

    mu -- center of the distribution, 
    three_stdev -- cut off distance from the center
    """
    variance = (three_stdev/3.0)**2
    g = 1./np.sqrt(2. * pi * variance) * np.exp(-(1./(2.*variance)) * (x-mu)**2)
    return g

def step_rescale_1D(x, x0, width):
    """
    Returns normalized step function
    """
    s = 1.0/width * (np.abs(x - x0) < width)
    return s

def gauss_rescale_lim_1D(x, mu, three_stdev):
    variance = (three_stdev/3.0)**2
    g = 1./np.sqrt(2. * pi * variance) * np.exp(-(1./(2.*variance)) * (x-mu)**2) * (abs(x - mu) < three_stdev)
    return g





def gauss_rescale_2D(x, y, mu, three_stdev):
    """
    Returns normalized gaussian 2D scale function 

    x, y        -- coordinates a point at which we calculate the density 
    mu          -- distribution mean vector
    three_stdev -- 3 * standard deviation of the distribution
    """
    h = 1./(2*pi)
    stdev = three_stdev/3.0
    h_n = h * stdev
    Z = h_n * np.exp ( - stdev**(-2) * 0.5 * ((x - mu[0])**2 + (y - mu[1])**2) )
    return Z

def step_rescale_2D(xp, yp, R):
    """
    Returns normalized 2D step function
    """
    s = (xp**2 + yp**2 <= R**2)
    return s

def gauss_rescale_2D_lim(x, y, mu, three_stdev):
    h = 1./(2*pi)
    stdev = three_stdev/3.0
    h_n = h * stdev/1
    Z = h_n * np.exp ( - stdev**(-2) * 0.5 * ((x - mu[0])**2 + (y - mu[1])**2) ) 
    Z *= ((x - mu[0])**2 + (y - mu[1])**2 < three_stdev**2)
    return Z





def gauss_rescale_3D(x, y, z, mu, three_stdev):
    """
    Returns normalized gaussian 3D scale function 

    x, y, z     -- coordinates a point at which we calculate the density 
    mu          -- distribution mean vector
    three_stdev -- 3 * standard deviation of the distribution
    """
    h = 1./(2*pi)
    stdev = three_stdev/3.0
    h_n = h * stdev
    Z = h_n * np.exp( - stdev**(-2) * 0.5 * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2 ) )
    return Z

def step_rescale_3D(xp, yp, zp, R):
    """
    Returns normalized 3D step function
    """
    s = (xp**2 + yp**2 + zp**2 <= R**2)
    return s

def gauss_rescale_lim_3D(x, y, z, mu, three_stdev):
    h = 1./(2*pi)
    stdev = three_stdev/3.0
    h_n = h * stdev
    Z = h_n * np.exp( - stdev**(-2) * 0.5 * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2 ) )
    Z *= ((x - mu[0])**2 + (y - mu[1])**2 + (z-mu[2])**2 < three_stdev**2)
    return Z