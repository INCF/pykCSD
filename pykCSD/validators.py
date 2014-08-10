import numpy as np

from pykCSD import KCSD
import plotting_utils as plut
import potentials as pt

"""
This module contains routines to perform a visual check of pots and CSD
created with a forward calculation scheme.
"""


def calculate_potential_1D(csd, boundary_x, h):
    """
    csd : np.array
        True csd
    boundary : list
        min and max coordinate
    h : float
        cylinder radius
    """
    sigma = 1.0
    x = np.linspace(boundary_x[0], boundary_x[1], csd.shape[0])
    true_pots = [integrate_1D(csd, x, x0, h, sigma) for x0 in x]
    return true_pots


def integrate_1D(csd, z, z0, h, sigma):
    pot = np.trapz((np.sqrt((z0 - z)**2 + h**2) - np.abs(z0 - z)) * csd, z)
    pot *= 1.0/(2 * sigma)
    return pot


def calculate_potential_2D(csd, boundary_x, boundary_y, h):
    sigma = 1.0
    nx, ny = csd.shape[0], csd.shape[1]
    x = np.linspace(boundary_x[0], boundary_x[1], nx)
    y = np.linspace(boundary_y[0], boundary_y[1], ny)
    true_pots = np.zeros((nx, ny))
    for i, y0 in enumerate(y):
        for j, x0 in enumerate(x):
            true_pots[i, j] = integrate_2D(x0, y0, x, y, csd, h)
    return true_pots


def integrate_2D(x, y, xlin, ylin, csd, h):
    """
    X,Y - parts of meshgrid
    """
    X, Y = np.meshgrid(xlin, ylin)
    # Nx = xlin.shape[0]
    Ny = ylin.shape[0]

    # construct 2-D integrand
    m = np.sqrt((x - X)**2 + (y - Y)**2)
    m[m < 0.00001] = 0.00001
    y = 2*h / np.arcsinh(m) * csd

    # do a 1-D integral over every row
    I = np.zeros(Ny)
    for i in xrange(Ny):
        I[i] = np.trapz(y[:, i], ylin)

    # then an integral over the result
    F = np.trapz(I, xlin)

    return F


def calculate_potential_3D(csd, boundary_x, boundary_y, boundary_z):
    nx, ny, nz = csd.shape[0], csd.shape[1], csd.shape[2]
    x = np.linspace(boundary_x[0], boundary_x[1], nx)
    y = np.linspace(boundary_y[0], boundary_y[1], ny)
    z = np.linspace(boundary_z[0], boundary_z[1], nz)
    true_pots = np.zeros((nx, ny, nz))
    for i, z0 in enumerate(z):
        for j, y0 in enumerate(y):
            for k, x0 in enumerate(x):
                true_pots[k, j, i] = integrate_3D(x0, y0, z0, x, y, z, csd)
    return true_pots


def integrate_3D(x, y, z, xlin, ylin, zlin, csd):
    # TODO: NOT YET WORKING AS EXPECTED
    X, Y, Z = np.meshgrid(xlin, ylin, zlin)
    Nz = zlin.shape[0]
    Ny = ylin.shape[0]

    m = np.sqrt((x - xlin)**2 + (y - ylin)**2 + (z - zlin)**2)
    m[m < 0.00001] = 0.00001
    csd = csd / m
    #print 'CSD:'
    #print csd

    J = np.zeros((Ny, Nz))
    for i in xrange(Nz):
        J[:, i] = np.trapz(csd[:, :, i], zlin)

    #print '1st integration'
    #print J

    Ny = ylin.shape[0]
    I = np.zeros(Ny)
    for i in xrange(Ny):
        I[i] = np.trapz(J[:, i], ylin)

    #print '2nd integration'
    #print I

    norm = np.trapz(I, xlin)
    
    #print '3rd integration'
    #print norm
    
    return norm


def main1D():
    x = np.linspace(-10, 10, 100)

    true_csd = 1.0 * np.exp(-(x + 2.)**2/(2 * np.pi * 0.5))
    true_csd += 0.5 * np.exp(-(x - 7)**2/(2 * np.pi * 1.0))

    elec_pos = np.array([-9.0, -6.9, -3.1, -0.3, 2.5, 5.7, 9.3])
    params = {'xmin': -10, 'xmax': 10, 'gdX': 0.20}
    indx = [5, 15, 25, 45, 51, 73, 89]

    compare_with_model_1D(x, true_csd, indx, params)


def compare_with_model_1D(X, true_csd, indx, params):
    true_pot = calculate_potential_1D(true_csd, [np.min(X), np.max(X)], 0.5)

    elec_pos = np.array([[X[i]] for i in indx])
    pots = np.array([[true_pot[i]] for i in indx])

    k = KCSD(elec_pos, pots, params)
    k.estimate_pots()
    k.estimate_csd()

    true_csd = np.atleast_2d(true_csd)
    true_pot = np.atleast_2d(true_pot)
    rec_csd = k.solver.estimated_csd
    rec_pot = k.solver.estimated_pots

    csd_err = true_csd - rec_csd
    csd_err = get_relative_error(rec_csd[:,0], true_csd[0,:])
    pot_err = true_pot - rec_pot
    pot_err = get_relative_error(rec_pot[:,0], true_pot[0,:])
    print 'true_csd.shape: ', true_csd.shape
    print 'recstr_csd.shape: ', rec_csd.shape
    print 'csd_err.shape: ', csd_err.shape

    plut.plot_comparison_1D(X, elec_pos, true_csd[0, :], true_pot[0, :],
                            rec_csd, rec_pot, csd_err, pot_err)


def main2D():
    xlin = np.linspace(0, 10, 101)
    ylin = np.linspace(0, 10, 101)
    X, Y = np.meshgrid(xlin, ylin)
    true_csd = 1.0 * np.exp(-((X - 8.)**2 + (Y - 8)**2)/(2 * np.pi * 1.5))
    true_csd -= 0.5 * np.exp(-((X - 1)**2 + (Y - 9)**2)/(2 * np.pi * 2.0))
    true_csd += 1.5 * np.exp(-((X - 2)**2 + (Y - 2)**2)/(2 * np.pi * 2.0))
    kcsd_params = {'xmin': 0, 'xmax': 10,
                   'ymin': 0, 'ymax': 10,
                   'gdX': 0.10, 'gdY': 0.10,
                   'h': 0.5}
    indx = [[5, 5], [15, 10], [25, 50], [45, 70],
            [51, 30], [73, 89], [5, 80], [90, 15], [60,90]]
    #indx = []
    #for x in xrange(5, 100, 15):
    #    for y in xrange(5 ,100, 15):
    #        indx.append([x,y])
    compare_with_model_2D(X, Y, true_csd, indx, kcsd_params)


def compare_with_model_2D(X, Y, true_csd, indx, params):
    boundary_x = [np.min(X), np.max(X)]
    boundary_y = [np.min(Y), np.max(Y)]
    true_pots = calculate_potential_2D(true_csd, boundary_x, boundary_y, 
                                       params['h'])

    elec_pos = np.array([[X[i, j], Y[i, j]] for i, j in indx])
    pots = np.array([[true_pots[i, j]] for i, j in indx])

    print elec_pos
    true_pots = np.atleast_2d(true_pots)
    k = KCSD(elec_pos, pots, params)
    k.estimate_pots()
    k.estimate_csd()
    rec_csd = k.solver.estimated_csd
    rec_pot = k.solver.estimated_pots
    csd_err = get_relative_error(true_csd[:100, :100], rec_csd[:100, :100, 0].T)
    pot_err = get_relative_error(true_pots[:100, :100], rec_pot[:100, :100, 0].T)
    print 'true_csd.shape: ', true_csd.shape
    print 'recstr_csd.shape: ', rec_csd.shape
    print 'csd_err.shape: ', csd_err.shape

    plut.plot_comparison_2D(X[1:-1, 1:-1], Y[1:-1, 1:-1], elec_pos,
                            true_csd[1:-1, 1:-1], true_pots[1:-1, 1:-1],
                            rec_csd[1:-1, 1:-1, 0].T, rec_pot[1:-1, 1:-1, 0].T,
                            csd_err[1:-1, 1:-1], pot_err[1:-1, 1:-1])


def main3D():
    # TODO: NOT YET READY
    xlin = np.linspace(0, 10, 20)
    ylin = np.linspace(0, 10, 20)
    zlin = np.linspace(0, 10, 20)
    X, Y, Z = np.meshgrid(xlin, ylin, zlin)
    true_csd = 1.0 * np.exp(-((X - 8.)**2 + (Y - 8)**2 + (Z - 8)**2)/(2 * np.pi * 1.5))
    true_csd -= 0.5 * np.exp(-((X - 1)**2 + (Y - 9)**2 + (Z - 9)**2)/(2 * np.pi * 2.0))
    true_csd += 1.5 * np.exp(-((X - 2)**2 + (Y - 2)**2 + (Z - 2)**2)/(2 * np.pi * 2.0))
    true_pots = compare_with_model_3D(X, Y, Z, true_csd, None, None)
    return true_pots, true_csd

def compare_with_model_3D(X, Y, Z, true_csd, indx, params):
    # TODO: NOT YET READY
    boundary_x = [np.min(X), np.max(X)]
    boundary_y = [np.min(Y), np.max(Y)]
    boundary_z = [np.min(Z), np.max(Z)]
    true_pots = calculate_potential_3D(true_csd, boundary_x, boundary_y, boundary_z)
    return true_pots


def get_relative_error(orig, rec):
    norm_orig = (orig-np.min(orig))/(np.max(orig)-np.min(orig))
    norm_rec = (rec-np.min(rec))/(np.max(rec)-np.min(rec))
    return np.abs(norm_rec - norm_orig) * 100


if __name__ == '__main__':
    #main3D()
    main2D()
    #main1D()
