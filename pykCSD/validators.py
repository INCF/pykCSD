import numpy as np

# from scipy import integrate

from pykCSD import KCSD
import plotting_utils as plut


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
    for i, y0 in enumerate(x):
        for j, x0 in enumerate(y):
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
    #-----------------------------------
    m = np.sqrt((x - X)**2 + (y - Y)**2)
    m[m < 0.00001] = 0.00001
    y = 2*h / np.arcsinh(m) * csd

    # do a 1-D integral over every row
    #-----------------------------------
    I = np.zeros(Ny)
    for i in range(Ny):
        I[i] = np.trapz(y[i, :], ylin)

    # then an integral over the result
    #-----------------------------------
    F = np.trapz(I, xlin)

    return F


def calculate_potential_3D(csd, boundary):
    pass


def main1D():
    x = np.linspace(-10, 10, 100)

    true_csd = 1.0 * np.exp(-(x + 2.)**2/(2 * np.pi * 0.5))
    true_csd += 0.5 * np.exp(-(x - 7)**2/(2 * np.pi * 1.0))

    true_pot = calculate_potential_1D(true_csd, [np.min(x), np.max(x)], 0.5)

    indx = [5, 15, 25, 45, 51, 73, 89]
    elec_pos = np.array([[x[i]] for i in indx])
    pots = np.array([[true_pot[i]] for i in indx])

    params = {'x_min': -10, 'x_max': 10, 'gdX': 0.20}
    k = KCSD(elec_pos, pots, params)
    k.estimate_pots()
    k.estimate_csd()

    true_csd = np.atleast_2d(true_csd)
    true_pot = np.atleast_2d(true_pot)
    rec_csd = k.solver.estimated_csd
    rec_pot = k.solver.estimated_pots

    csd_err = true_csd - rec_csd
    csd_err = csd_err * max(rec_csd)/max(true_csd)
    pot_err = true_pot - rec_pot
    pot_err = pot_err * max(rec_pot)/max(true_pot)
    print true_csd.shape
    print rec_csd.shape
    print csd_err.shape

    plut.plot_comparison_1D(x, elec_pos, true_csd, true_pot,
                            rec_csd, rec_pot, csd_err, pot_err)


def main2D():
    xlin = np.linspace(0, 10, 100)
    ylin = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(xlin, ylin)
    true_csd = 1.0 * np.exp(-((X - 8.)**2 + (Y - 8)**2)/(2 * np.pi * 1.5))
    true_csd -= 0.5 * np.exp(-((X - 1)**2 + (Y - 9)**2)/(2 * np.pi * 2.0))
    true_csd += 1.5 * np.exp(-((X - 2)**2 + (Y - 2)**2)/(2 * np.pi * 2.0))

    boundary_x = [np.min(X), np.max(X)]
    boundary_y = [np.min(Y), np.max(Y)]
    true_pots = calculate_potential_2D(true_csd, boundary_x, boundary_y, 1.5)

    indx = [[5, 5], [15, 10], [25, 50], [45, 70], [51, 30], [73, 89]]

    elec_pos = np.array([[X[i, j], Y[i, j]] for i, j in indx])
    pots = np.array([[true_pots[i, j]] for i, j in indx])

    print elec_pos
    true_pots = np.atleast_2d(true_pots)
    params = {'xmin': 0, 'xmax': 10,
              'ymin': 0, 'ymax': 10,
              'gdX': 0.10, 'gdY': 0.10}
    k = KCSD(elec_pos, pots, params)
    k.estimate_pots()
    k.estimate_csd()
    rec_csd = k.solver.estimated_csd
    rec_pot = k.solver.estimated_pots
    csd_err = 0.0
    pot_err = 0.0
    print true_csd.shape
    print rec_csd.shape

    plut.plot_comparison_2D(X, Y, elec_pos, true_csd, true_pots,
                            rec_csd[:100, :100], rec_pot[:100, :100],
                            csd_err, pot_err)

if __name__ == '__main__':
    main2D()
    #main1D()
