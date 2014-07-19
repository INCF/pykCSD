# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.interpolate import interp1d


def probe_dist_table_points(R, dist_max, dt_len):
    """
    Helps to choose important points in the distance table to probe
    the distance function of potential. The points should be probed
    denser in the place, where the function changes more rapidly.

    Parameters
    -----------
    R : float
        basis function radius
    dist_max : float
        distance between two most distant points in estimation space
    dt_len : int
        total number of samples in distance table

    Returns
    -----------
    xs : np.array
        sparsely probed indices from the distance table
    """
    dense_step = 3
    denser_step = 1
    sparse_step = 9
    border1 = 0.9 * R/dist_max * dt_len
    border2 = 1.3 * R/dist_max * dt_len

    xs = np.arange(0,  border1, dense_step)
    xs = np.append(xs, border1)
    zz = np.arange((border1 + denser_step), border2, dense_step)

    xs = np.concatenate((xs, zz))
    xs = np.append(xs, [border2, (border2 + denser_step)])
    xs = np.concatenate((xs, np.arange((border2 + denser_step + sparse_step/2),
                                       dt_len, sparse_step)))
    xs = np.append(xs, dt_len + 1)

    xs = np.unique(np.array(xs))

    return xs


def interpolate_dist_table(xs, probed_dist_table, dt_len):
    inter = interp1d(
        x=xs,
        y=probed_dist_table,
        kind='cubic',
        fill_value=0.0
    )
    dt_int = np.array([inter(i) for i in xrange(dt_len)])
    dt_int.flatten()
    return dt_int


def generated_potential(dist, dist_max, dist_table):
    """
    Parameters
    -----------
    dist : float
        distance at which we want to obtain the potential value
    dist_max : float
        distance between two most distant points in estimation space
    dist_table : np.array
        potential as a probed function of distance

    Returns
    -----------
    pot : float
        value of potential at specified distance from the source
    """
    dt_len = len(dist_table)
    indices = np.uint16(np.round(dt_len * dist/dist_max))
    ind = np.maximum(0, np.minimum(indices, dt_len-1))

    pot = dist_table[ind]
    return pot
