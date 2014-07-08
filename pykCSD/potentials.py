# -*- coding: utf-8 -*-
import numpy as np
import basis_functions as bf

def int_pot(src, arg, current_pos, h, R, sigma, src_type):
    """
    Returns contribution of a single source as a function of distance
    """
    y = 1./(2 * sigma) * (((arg - current_pos)**2 + R**2)**0.5 - abs(arg - current_pos))
    if src_type == "gaussian":
        # for this formula look at formula (8) from Pettersen et al., 2006
        y *= bf.gauss_rescale_1D(src, current_pos, h)
    if src_type == "step":
        y *= bf.step_rescale_1D(src, current_pos, h)
    if src_type == 'gauss_lim':
        y *= bf.gauss_rescale_lim_1D(src, current_pos, h)

    return y

def b_pot_quad(src, arg, h, R, sigma, src_type):
    """
    Returns potential as a function of distance from the source.
    """
    x = np.linspace(src - 4*h, src + 4*h, 51)  # manipulate the resolution, 
    # TODO: smarter choice of integration resolution
    pot = np.array([int_pot(src, arg, current_pos, h, R, sigma, src_type) for current_pos in x])

    z = np.trapz(pot, x)

    return z