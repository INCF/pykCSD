# -*- coding: utf-8 -*-
import numpy as np


def get_src_params_2D(Lx, Ly, n_src):
    """
    helps uniformally distribute n_src sources among an area of size Lx x Ly

    INPUT
    Lx,Ly     - lengths in the directions x,y of the area, ...
             the sources should be placed
    n_src     - number of sources

    OUTPUT
    nx,ny     - number of sources in directions x,y
    ds        - spacing between the sources
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


def make_src_2D(X, Y, n_src, ext_x, ext_y, R_init):
    """
    INPUT
    X,Y                 - Points at which CSD will be estimated
    n_src               - number of sources we want to include in the model
    ext_x,ext_y,        - how should the sources extend the area X,Y,Z
    R_init              - demanded radius of the basis element

    OUTPUT
    X_src,Y_src       - Positions of the sources
    nx,ny             - number of sources in directions x,y,z
    R                 - effective radius of the basis element
    """
    Lx = np.max(X) - np.min(X)
    Ly = np.max(Y) - np.min(Y)

    Lx_n = Lx + 2*ext_x
    Ly_n = Ly + 2*ext_y

    [nx, ny, Lx_nn, Ly_nn, ds] = get_src_params_2D(Lx_n, Ly_n, n_src)

    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2

    X_src, Y_src = np.meshgrid(np.linspace(-ext_x_n, Lx+ext_x_n, nx),
                               np.linspace(-ext_y_n, Ly+ext_y_n, ny))

    d = np.round(R_init/ds)
    R = d * ds

    return X_src, Y_src, R


def get_src_params_3D(Lx, Ly, Lz, n_src):
    """
    helps uniformally distribute n_src sources in a cuboid of size Lx x Ly x Lz

    INPUT
    Lx,Ly,Lz    - lengths in the directions x,y,z of the area, ...
             the sources should be placed
    n_src     - number of sources

    OUTPUT
    nx,ny,nz     - number of sources in directions x,y
    ds        - spacing between the sources
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

    (nx, ny, nz, Lx_nn, Ly_nn, Lz_nn, ds) = get_src_params_3D(Lx_n, Ly_n, Lz_n,
                                                              n_src)

    ext_x_n = (Lx_nn - Lx)/2
    ext_y_n = (Ly_nn - Ly)/2
    ext_z_n = (Lz_nn - Lz)/2

    (X_src, Y_src, Z_src) = np.meshgrid(np.linspace(-ext_x_n, Lx+ext_x_n, nx),
                                        np.linspace(-ext_y_n, Ly+ext_y_n, ny),
                                        np.linspace(-ext_z_n, Lz+ext_z_n, nz))

    d = np.round(R_init/ds)
    R = d * ds

    return (X_src, Y_src, Z_src, R)
