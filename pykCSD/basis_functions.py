from __future__ import division

import numpy as np
from numpy import pi


"""
This module contains basis functions for KCSD in 1D, 2D, 3D.
"""


def gauss_rescale_1D(x, mu, three_stdev):
    """
    Returns normalized gaussian scale function value

    Key arguments:
    x -- point or set of points where function should be calculated
    mu -- center of the distribution
    """
    variance = (three_stdev/3.0)**2
    c = 1./np.sqrt(2 * pi * variance)
    g = c * np.exp(-(1./(2.*variance)) * (x - mu)**2)
    return g


def gauss_rescale_lim_1D(x, mu, three_stdev):
    """
    Returns gaussian scale function value cut off after 3 standard deviations.

    Key arguments:
    x -- point or set of points where function should be calculated
    mu -- center of the distribution,
    three_stdev -- cut off distance from the center
    """
    g = gauss_rescale_1D(x, mu, three_stdev)
    g *= (np.abs(x - mu) < three_stdev)
    return g


def step_rescale_1D(x, x0, width):
    """
    Returns normalized step function.

    Key arguments:
    x -- point or set of points where function should be calculated
    x0 -- center
    width -- cutoff range
    """
    s = 1.0/width * (np.abs(x - x0) < width)
    return s


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
    Z = h_n * np.exp(-0.5 * stdev**(-2) * ((x - mu[0])**2 + (y - mu[1])**2))
    return Z


def gauss_rescale_lim_2D(x, y, mu, three_stdev):
    """
    """
    Z = gauss_rescale_2D(x, y, mu, three_stdev)
    Z *= ((x - mu[0])**2 + (y - mu[1])**2 < three_stdev**2)
    return Z


def step_rescale_2D(xp, yp, mu, R):
    """
    Returns normalized 2D step function
    """
    s = ((xp-mu[0])**2 + (yp-mu[1])**2 <= R**2)
    return s


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
    c = 0.5 * stdev**(-2)
    Z = np.exp(-c * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2))
    Z = Z * h_n
    return Z


def gauss_rescale_lim_3D(x, y, z, mu, three_stdev):
    Z = gauss_rescale_3D(x, y, z, mu, three_stdev)
    Z = Z * ((x - mu[0])**2 + (y - mu[1])**2 + (z - mu[2])**2 < three_stdev**2)
    return Z


def step_rescale_3D(xp, yp, zp, mu, R):
    """
    Returns normalized 3D step function
    """
    s = (xp**2 + yp**2 + zp**2 <= R**2)
    return s
