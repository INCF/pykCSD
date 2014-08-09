# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np

"""
This module contains routines for distributing sources uniformly
in 1D, 2D, 3D space.
"""


def make_src_1D(X, ext_x, n_src, R_init):
    """
    **Parameters**

    X : np.array
        points at which CSD will be estimated
    
    ext_x : float
        how should the sources extend the area X, Y
    
    n_src : int
        demanded number of sources to be included in the model
    
    R_init : float
        demanded radius of the basis element

    **Returns**
    
    X_src : np.arrays
        positions of the sources
    
    nx : ints
        number of sources over the line
    
    R : float
        effective radius of the basis element
    """
    Lx = np.max(X) - np.min(X)

    Lx_n = Lx + 2*ext_x

    (nx, Lx_nn, ds) = get_src_params_1D(Lx_n, n_src)

    ext_x_n = (Lx_nn - Lx)/2.0

    X_src = np.linspace(np.min(X)-ext_x_n, np.max(X)+ext_x_n, nx)

    d = round(R_init/ds)
    R = d * ds

    return (X_src, R)


def get_src_params_1D(Lx, n_src):
    """
    Helps to distribute n_src sources evenly over a line of length Lx

    **Parameters**

    Lx : float
        length of the line over which the sources should be placed
    
    n_src : int
        demanded number of sources to be included in the model

    **Returns**

    nx : int
        number of sources in direction x
    
    Lx_n : float
        updated length
    
    ds : float
        spacing between the sources
    """
    V = Lx
    V_unit = V/n_src
    L_unit = V_unit

    nx = round(Lx/L_unit)
    ds = Lx/(nx-1)

    Lx_n = (nx-1)*ds

    return (nx, Lx_n, ds)


# TODO: replace with the hypervolume approach?
def make_src_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """
    **Parameters**
    
    X, Y : np.arrays
        points at which CSD will be estimated
    
    n_src : int
        demanded number of sources to be included in the model
    
    ext_x, ext_y : floats
        how should the sources extend the area X, Y
    
    R_init : float
        demanded radius of the basis element

    **Returns**

    X_src, Y_src : np.arrays
        positions of the sources
    
    nx, ny : ints
        number of sources in directions x,y
        new n_src = nx * ny may not be equal to the demanded number of sources
    
    R : float
        effective radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)

    Lx_n = Lx + 2*ext_x
    Ly_n = Ly + 2*ext_y

    [nx, ny, Lx_nn, Ly_nn, ds] = get_src_params_2D(Lx_n, Ly_n, n_src)

    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2

    lin_x = np.linspace(np.min(X) - ext_x_n, np.max(X) + ext_x_n, nx)
    lin_y = np.linspace(np.min(Y) - ext_y_n, np.max(Y) + ext_y_n, ny)

    X_src, Y_src = np.meshgrid(lin_x, lin_y)

    d = round(R_init/ds)
    R = d * ds

    return X_src, Y_src, R


def get_src_params_2D_new(Lx, Ly, n_src):
    V = Lx*Ly
    V_unit = V/n_src
    L_unit = V_unit**(0.5)

    nx = np.ceil(Lx/L_unit)
    ny = np.ceil(Ly/L_unit)

    ds = Lx/(nx-1)

    Lx_n = (nx-1)*ds
    Ly_n = (ny-1)*ds

    return (nx, ny, Lx_n, Ly_n, ds)


def get_src_params_2D(Lx, Ly, n_src):
    """
    Helps to distribute n_src sources evenly in a rectangle of size Lx * Ly

    **Parameters**

    Lx, Ly : floats
        lengths in the directions x, y of the area,
        the sources should be placed
    
    n_src : int
        demanded number of sources

    **Returns**
    
    nx, ny : ints
        number of sources in directions x, y
        new n_src = nx * ny may not be equal to the demanded number of sources
    
    Lx_n, Ly_n : floats
        updated lengths in the directions x, y
    
    ds : float
        spacing between the sources
    """
    coeff = [Ly, Lx - Ly, -Lx * n_src]

    rts = np.roots(coeff)
    r = [r for r in rts if type(r) is not complex and r > 0]
    nx = r[0]
    ny = n_src/nx

    ds = Lx/(nx-1)

    nx = np.floor(nx) + 1
    ny = np.floor(ny) + 1

    Lx_n = (nx - 1) * ds
    Ly_n = (ny - 1) * ds

    return (nx, ny, Lx_n, Ly_n, ds)


def make_src_3D(X, Y, Z, n_src, ext_x, ext_y, ext_z, R_init):
    """
    **Parameters**

    X, Y, Z : np.arrays
        points at which CSD will be estimated

    n_src : int
        desired number of sources we want to include in the model

    ext_x, ext_y, ext_z : floats
        how should the sources extend over the area X,Y,Z

    R_init : float
        demanded radius of the basis element

    **Returns**

    X_src, Y_src, Z_src : np.arrays
        positions of the sources in 3D space

    nx, ny, nz : ints
        number of sources in directions x,y,z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources
        
    R : float
        updated radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)
    Lz = np.max(Z) - np.min(Z)

    Lx_n = Lx + 2*ext_x
    Ly_n = Ly + 2*ext_y
    Lz_n = Lz + 2*ext_z

    (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n, Ly_n, Lz_n,
                                                              n_src)

    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2
    ext_z_n = (Lz_nn - Lz)/2

    lin_x = np.linspace(np.min(X) - ext_x_n, np.max(X) + ext_x_n, nx)
    lin_y = np.linspace(np.min(Y) - ext_y_n, np.max(Y) + ext_y_n, ny)
    lin_z = np.linspace(np.min(Z) - ext_z_n, np.max(Z) + ext_z_n, nz)

    (X_src, Y_src, Z_src) = np.meshgrid(lin_x, lin_y, lin_z)

    d = np.round(R_init/ds)
    R = d * ds

    return (X_src, Y_src, Z_src, R)


def get_src_params_3D(Lx, Ly, Lz, n_src):
    """
    Helps to evenly distribute n_src sources in a cuboid of size Lx * Ly * Lz

    **Parameters**

    Lx, Ly, Lz : floats
        lengths in the directions x, y, z of the area,
        the sources should be placed

    n_src : int
        demanded number of sources to be included in the model

    **Returns**

    nx, ny, nz : ints
        number of sources in directions x, y, z
        new n_src = nx * ny * nz may not be equal to the demanded number of
        sources

    Lx_n, Ly_n, Lz_n : floats
        updated lengths in the directions x, y, z

    ds : float
        spacing between the sources (grid nodes)
    """
    V = Lx*Ly*Lz
    V_unit = V/n_src
    L_unit = V_unit**(1./3.)

    nx = np.ceil(Lx/L_unit)
    ny = np.ceil(Ly/L_unit)
    nz = np.ceil(Lz/L_unit)

    ds = Lx/(nx-1)

    Lx_n = (nx-1)*ds
    Ly_n = (ny-1)*ds
    Lz_n = (nz-1)*ds

    return (nx, ny, nz,  Lx_n, Ly_n, Lz_n, ds)
