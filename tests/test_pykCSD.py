#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pykCSD
----------------------------------

Tests for `pykCSD` module.
"""

import unittest
import numpy as np

#from pylab import *
from numpy.linalg import norm
from pykCSD.pykCSD import KCSD1D

class TestKCSD1D(unittest.TestCase):

    def setUp(self):
        params = {
                'x_min': 0.0, 
                'x_max': 1.0, 
                'dist_density': 11,
                }
        self.k = KCSD1D(elec_pos = np.array([0.2, 0.7]), 
                        sampled_pots = np.array([1.0, 0.5]), 
                        params = params)
        self.k.calculate_matrices()
        self.k.estimate_pots()
        self.k.estimate_csd()

        self.reference_pots = np.loadtxt('tests/test_datasets/2_elec_pot.dat', skiprows=5)
        self.reference_csd = np.loadtxt('tests/test_datasets/2_elec_csd.dat', skiprows=5)

    def test_KCSD1D_pot_estimation_two_electrodes(self):
        #figure()
        #plot(self.k.estimated_pots)
        #plot(self.reference_pots)
        #show()
        pot_difference = norm(self.k.estimated_pots - self.reference_pots)
        self.assertAlmostEqual(pot_difference, 0.05, places=1)

    def test_KCSD1D_csd_estimation_two_electrodes(self):
        #print ''
        #print self.reference_csd
        #print self.k.estimated_csd
        #print self.k.estimation_area
        #print self.k.source_positions
        #figure()
        #plot(self.k.estimated_csd)
        #plot(self.reference_csd)
        #show()
        csd_difference = norm(self.k.estimated_csd - self.reference_csd)
        self.assertAlmostEqual(csd_difference , 0.0, places=0)

    def test_KCSD1D_cross_validation_two_electrodes(self):
        lambdas = np.array([0.1, 0.5, 1.0])
        self.k.lambd = self.k.choose_lambda(lambdas)
        self.assertEquals(self.k.lambd, 1.0)
        self.k.lambd = 0.0


    def test_KCSD1D_zero_pot(self):
        params = {'n_sources':20, 'dist_density':20}
        k_zero = KCSD1D(elec_pos=[1.0, 2.0, 3.0, 4.0, 5.0], 
                        sampled_pots=[0.0, 0.0, 0.0, 0.0, 0.0],
                        params = params)
        k_zero.calculate_matrices()
        k_zero.estimate_pots()
        self.assertAlmostEqual(norm(k_zero.estimated_pots), 0.0, places=10)
        k_zero.estimate_csd()
        self.assertAlmostEqual(norm(k_zero.estimated_csd), 0.0, places=10)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
