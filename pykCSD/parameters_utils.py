# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
import scipy.spatial.distance as distance


"""
This module contains routines for managing and validating kCSD parameters.
"""


def check_for_duplicated_electrodes(elec_pos):
    """
    **Parameters**

    elec_pos : np.array

    **Returns**

    has_duplicated_elec : Boolean
    """
    unique_elec_pos = np.vstack({tuple(row) for row in elec_pos})
    has_duplicated_elec = unique_elec_pos.shape == elec_pos.shape
    return has_duplicated_elec


def min_dist(elec_pos):
    """
    Returns minimal distance between any electrodes
    """
    min_distance = 0.0
    dim = len(elec_pos[0])
    if dim > 1:
        min_distance = distance.pdist(elec_pos).min()
    else:
        flat_elec = elec_pos.flatten()
        min_distance = distance.pdist(flat_elec[:, None], 'cityblock').min()
    return min_distance