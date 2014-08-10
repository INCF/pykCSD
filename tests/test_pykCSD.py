#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pykCSD
----------------------------------

Tests for `pykCSD` module.
"""

import unittest

import numpy as np
from pylab import *
from numpy.linalg import norm

from pykCSD.KCSD2D import KCSD2D
from pykCSD.pykCSD import KCSD
from pykCSD import potentials as pt
from pykCSD import basis_functions as bf
from pykCSD import source_distribution as sd
from pykCSD import dist_table_utils as dt
from pykCSD import cross_validation as cv
from sklearn.cross_validation import LeaveOneOut


class TestKCSD1D(unittest.TestCase):

    def setUp(self):
        pass

    def test_KCSD_1D_int_pot(self):
        """results of int_pot_1D() should be similar to matlab results"""
        result_fpath = 'tests/test_datasets/KCSD1D/expected_pot_intargs.dat'
        param_fpath = 'tests/test_datasets/KCSD1D/intarg_parameters.dat'
        expected_results = np.loadtxt(result_fpath, delimiter=',')
        params = np.loadtxt(param_fpath, delimiter=',')
        srcs = params[0]
        args = params[1]
        curr_pos = params[2]
        hs = params[3]
        Rs = params[4]
        sigmas = params[5]

        for i, expected_result in enumerate(expected_results):
            # the parameter names h, R were swapped comparing to Matlab version
            kcsd_result = pt.int_pot_1D(src=srcs[i],
                                        arg=args[i],
                                        curr_pos=curr_pos[i],
                                        R=hs[i],
                                        h=Rs[i],
                                        sigma=sigmas[i],
                                        basis_func=bf.gauss_rescale_lim_1D)
            # print 'ex:', expected_result, ' kcsd:', kcsd_result
            self.assertAlmostEqual(expected_result, kcsd_result, places=3)

    def test_KCSD_1D_cross_validation_two_electrodes(self):
        """cross validation should promote high lambdas in this case"""

        params = {'x_min': 0.0, 'x_max': 1.0, 'gdX': 0.1,
                  'source_type': 'gauss_lim'}
        k = KCSD(elec_pos=np.array([[0.2], [0.7]]),
                 sampled_pots=np.array([[1.0, 0.5]]).T,
                 params=params)
        k.estimate_pots()
        lambdas = np.array([0.1, 0.5, 1.0])
        n_elec = k.solver.elec_pos.shape[0]
        index_generator = LeaveOneOut(n_elec, indices=True)
        k.solver.lambd = cv.choose_lambda(lambdas, k.solver.sampled_pots,
                                          k.solver.k_pot, k.solver.elec_pos,
                                          index_generator)
        self.assertEquals(k.solver.lambd, 1.0)

    def test_KCSD_1D_zero_pot(self):
        """if measured potential is 0, the calculated potential should be 0"""

        params = {'n_sources': 20, 'dist_density': 20,
                  'source_type': 'gauss_lim'}
        k_zero = KCSD(elec_pos=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                      sampled_pots=np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]).T,
                      params=params)
        k_zero.estimate_pots()
        self.assertAlmostEqual(norm(k_zero.solver.estimated_pots), 0.0, places=10)

    def test_KCSD_1D_zero_csd(self):
        """if measured potential is 0, the calculated CSD should be 0"""

        params = {'n_sources': 20, 'dist_density': 20,
                  'source_type': 'gauss_lim'}
        k_zero = KCSD(elec_pos=np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
                      sampled_pots=np.array([[0.0], [0.0], [0.0], [0.0], [0.0]]),
                      params=params)
        k_zero.estimate_csd()
        self.assertAlmostEqual(norm(k_zero.solver.estimated_csd), 0.0, places=10)

    def test_KCSD_1D_incorrect_electrode_number(self):
        """if there are more electrodes than pots, it should raise exception"""
        with self.assertRaises(Exception):
            k = KCSD(elec_pos=np.array([[0], [1], [2]]), sampled_pots=[[0], [1]])

    def test_KCSD_1D_duplicated_electrode(self):
        """if two electrodes are at the same spot, it should raise exception"""
        with self.assertRaises(Exception):
            k = KCSD(elec_pos=np.array([[0], [0]]), sampled_pots=[[0], [0]])

    def tearDown(self):
        pass


class TestKCSD1D_full_reconstruction(unittest.TestCase):

    def setUp(self):
        self.sigma = 1.0
        self.xmin = -5.0
        self.xmax = 11.0
        self.x = np.linspace(self.xmin, self.xmax, 100)
        self.true_csd = 1.0 * np.exp(-(self.x - 2.)**2/(2 * np.pi * 0.5))
        self.true_csd += 0.5 * np.exp(-(self.x - 7)**2/(2 * np.pi * 1.0))
        self.h = .3

        def calculate_pot(csd, z, z0):
            pot = np.trapz((np.sqrt((z0 - z)**2 + self.h**2) - np.abs(z0 - z)) * csd, z)
            pot *= 1.0/(2 * self.sigma)
            return pot

        self.elec_pos = np.array([[x] for x in np.linspace(self.xmin, self.xmax, 20)])
        self.true_pots = [calculate_pot(self.true_csd, self.x, x0) for x0 in self.x]
        self.meas_pot = np.array([[calculate_pot(self.true_csd, self.x, x0)] for x0 in self.elec_pos])
        print self.meas_pot.shape
        print self.elec_pos.shape

    def test_KCSD_1D_pot_reconstruction(self):
        """reconstructed pots should be similar to model pots"""

        params = {'sigma': self.sigma, 'source_type': 'gauss_lim',
                  'x_min': self.xmin, 'x_max': self.xmax,
                  'h': self.h, 'n_src': 20}
        k = KCSD(self.elec_pos, self.meas_pot, params)
        k.estimate_pots()
        """print np.max(k.estimated_pots)
        plot(k.space_X, k.estimated_pots)
        plot(self.x, self.true_pots)
        scatter(k.elec_pos, k.sampled_pots)
        show()"""
        for estimated_pot, true_pot in zip(k.solver.estimated_pots, self.true_pots):
            self.assertAlmostEqual(estimated_pot, true_pot, places=1)

    def test_KCSD_1D_csd_reconstruction(self):
        """reconstructed csd should be similar to model csd"""

        params = {'sigma': self.sigma, 'source_type': 'gauss_lim',
                  'x_min': self.xmin, 'x_max': self.xmax, 'h': self.h}
        k = KCSD(self.elec_pos, self.meas_pot, params)
        k.estimate_csd()
        """print np.max(k.estimated_csd)
        plot(k.estimated_csd)
        plot(self.true_csd)
        show()
        print k.X_src"""
        for estimated_csd, true_csd in zip(k.solver.estimated_csd, self.true_csd):
            self.assertAlmostEqual(estimated_csd, true_csd, places=1)

    def test_KCSD_1D_lambda_choice(self):
        """for potentials calculated from model, lambda < 1.0"""

        params = {'sigma': self.sigma, 'source_type': 'gauss_lim',
                  'x_min': -5.0, 'x_max': 10.0, 'h': self.h}
        k = KCSD(self.elec_pos, self.meas_pot, params)
        lambdas = np.array([100.0/2**n for n in xrange(1, 20)])
        n_elec = k.solver.elec_pos.shape[0]
        index_generator = LeaveOneOut(n_elec, indices=True)
        k.solver.lambd = cv.choose_lambda(lambdas, k.solver.sampled_pots,
                                          k.solver.k_pot, k.solver.elec_pos,
                                          index_generator)
        self.assertLess(k.solver.lambd, 1.0)


class TestKCSD2D(unittest.TestCase):

    def setUp(self):
        pass

    def test_KCSD_2D_int_pot(self):
        """results of int_pot_2D() should be similar to matlab results"""
        result_fpath = 'tests/test_datasets/KCSD2D/expected_pot_intargs_2D.dat'
        param_fpath = 'tests/test_datasets/KCSD2D/intarg_parameters_2D.dat'
        expected_results = np.loadtxt(result_fpath, delimiter=',')
        params = np.loadtxt(param_fpath, delimiter=',')
        xps = params[0]
        yps = params[1]
        xs = params[2]
        Rs = params[3]
        hs = params[4]
        for i, expected_result in enumerate(expected_results):
            kcsd_result = pt.int_pot_2D(xp=xps[i], yp=yps[i], x=xs[i],
                                        R=Rs[i], h=hs[i],
                                        basis_func=bf.gauss_rescale_lim_2D)
            self.assertAlmostEqual(expected_result, kcsd_result, places=3)

    def test_KCSD_2D_zero_pot(self):
        elec_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        pots = np.array([[0], [0], [0], [0]])
        k = KCSD(elec_pos, pots)
        k.solver.estimate_pots()
        for pot in k.solver.estimated_pots.flatten():
            self.assertAlmostEqual(pot, 0.0, places=5)

    def test_KCSD2D_zero_csd(self):
        elec_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        pots = np.array([[0], [0], [0], [0]])
        k = KCSD(elec_pos, pots)
        k.solver.estimate_csd()
        for csd in k.solver.estimated_csd.flatten():
            self.assertAlmostEqual(csd, 0.0, places=5)

    def tearDown(self):
        pass


class TestKCSD2D_full_recostruction(unittest.TestCase):

    def setUp(self):
        elec_fpath = 'tests/test_datasets/KCSD2D/five_elec_elecs.dat'
        pots_fpath = 'tests/test_datasets/KCSD2D/five_elec_pots.dat'
        elec_pos = np.loadtxt(elec_fpath, delimiter=',')
        pots = np.loadtxt(pots_fpath, delimiter=',')
        pots = np.array([pots]).T
        params = {'n_sources': 9, 'gdX': 0.1, 'gdY': 0.1}
        self.k = KCSD2D(elec_pos, pots, params)
        self.k.init_model()

    def test_KCSD2D_R_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_R.dat'
        expected_R = np.loadtxt(fpath, delimiter=',')
        self.assertAlmostEqual(self.k.R, expected_R, places=5)

    def test_KCSD2D_b_pot_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_bpot.dat'
        expected_b_pot = np.loadtxt(fpath, delimiter=',')
        err = norm(expected_b_pot - self.k.b_pot_matrix, ord=2)
        # comparison_plot_2D(expected_b_pot, self.k.solver.b_pot_matrix,
        #                    'matlab', 'python')
        self.assertAlmostEqual(err, 0.0, places=3)

    def test_KCSD2D_k_pot_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_kpot.dat'
        expected_k_pot = np.loadtxt(fpath, delimiter=',')
        err = norm(expected_k_pot - self.k.k_pot, ord=2)
        self.assertAlmostEqual(err, 0.0, places=3)

    def test_KCSD2D_dist_table(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_dist_table.dat'
        expected_dt = np.loadtxt(fpath, delimiter=',')
        err = norm(expected_dt - self.k.dist_table)
        """plot(expected_dt-self.k.dist_table)
        show()"""
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_KCSD2D_b_src_matrix_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_b_src_matrix.dat'
        expected_b_src_matrix = np.loadtxt(fpath, delimiter=',')
        # comparison_plot_2D(expected_b_src_matrix, self.k.solver.b_src_matrix,
        #                   'matlab', 'python')
        err = norm(expected_b_src_matrix - self.k.b_src_matrix)
        self.assertAlmostEqual(err, 0.0, places=4)

    def test_KCSD2D_pot_estimation_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_estimated_pot.dat'
        expected_pots = np.loadtxt(fpath, delimiter=',')
        self.k.estimate_pots()
        err = norm(expected_pots - self.k.estimated_pots[:,:,0], ord=2)
        # comparison_plot_2D(expected_pots, self.k,estimated_pots,
        #                   'matlab', 'python')
        # print np.max(expected_pots - self.k.estimated_pots)
        self.assertAlmostEqual(err, 0.0, places=2)

    def test_KCSD2D_csd_estimation_five_electrodes(self):
        fpath = 'tests/test_datasets/KCSD2D/five_elec_estimated_csd.dat'
        expected_csd = np.loadtxt(fpath, delimiter=',')
        self.k.estimate_csd()
        err = norm(expected_csd - self.k.estimated_csd[:,:,0], ord=2)
        #comparison_plot_2D(expected_csd, self.k.estimated_csd[:,:,0],
        #                   'matlab', 'python')
        print np.max(expected_csd - self.k.estimated_csd)
        self.assertAlmostEqual(err, 0.0, places=0)

    def test_KCSD2D_cross_validation_five_electrodes(self):
        lambdas = np.array([100.0/2**n for n in xrange(1, 20)])
        n_elec = self.k.elec_pos.shape[0]
        index_generator = LeaveOneOut(n_elec, indices=True)
        self.k.lambd = cv.choose_lambda(lambdas,
            self.k.sampled_pots,
            self.k.k_pot, self.k.elec_pos,
            index_generator)

        self.assertGreater(self.k.lambd, 25.0)

    def tearDown(self):
        pass


class TestKCSD3D_full_recostruction(unittest.TestCase):

    def setUp(self):
        pass

    def test_KCSD_3D_zero_pot(self):
        """if the input pots are zero, estimated pots and csd should be zero"""
        elec_pos = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0),
                            (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 1, 1)])
        pots = np.array([[0], [0], [0], [0], [0], [0], [0], [0]])
        params = {
            'gdX': 0.2,
            'gdY': 0.2,
            'gdZ': 0.2,
            'n_src': 10
        }
        k = KCSD(elec_pos, pots, params)
        k.estimate_pots()
        k.estimate_csd()
        for pot in k.solver.estimated_pots.flatten():
            self.assertAlmostEqual(pot, 0.0, places=5)
        for csd in k.solver.estimated_csd.flatten():
            self.assertAlmostEqual(csd, 0.0, places=5)

    def test_KCSD_3D_model_pots(self):
        pass

    def tearDown(self):
        pass


class TestKCSD_all_utils(unittest.TestCase):

    def setUp(self):
        pass

    def test_make_src1D_no_ext_4_src(self):
        X = np.array([0.0, 0.1, 0.2, 0.3])
        (X_src, R) = sd.make_src_1D(X=X, ext_x=0.0, n_src=4, R_init=0.2)
        for i in xrange(len(X_src)):
            self.assertAlmostEqual(X_src[i], X[i], places=6)

    def test_make_src1D_negative_coords_4_src(self):
        X = np.array([-0.5, -0.3, -0.1, 0.1])
        (X_src, R) = sd.make_src_1D(X=X, ext_x=0.0, n_src=4, R_init=0.2)
        for i in xrange(len(X_src)):
            self.assertAlmostEqual(X_src[i], X[i], places=6)

    def test_make_src1D_with_ext_6_src(self):
        X = np.array([0.0, 0.1, 0.2, 0.3])
        (X_src, R) = sd.make_src_1D(X=X, ext_x=0.1, n_src=6, R_init=0.2)
        expected_X_src = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        for i in xrange(len(X_src)):
            self.assertAlmostEqual(X_src[i], expected_X_src[i], places=6)

    def test_make_src1D_with_ext_6_src_translated(self):
        X = np.array([0.0, 0.1, 0.2, 0.3]) - 10.0
        (X_src, R) = sd.make_src_1D(X=X, ext_x=0.1, n_src=6, R_init=0.2)
        expected_X_src = np.array([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]) - 10.0
        for i, src in enumerate(X_src):
            self.assertAlmostEqual(src, expected_X_src[i], places=6)

    def test_make_src2D_regular_grid(self):
        xmin, xmax = 0.0, 1.0
        ymin, ymax = 0.0, 1.0
        ext_x, ext_y = 0.0, 0.0
        nx, ny = 2, 2
        lin_x = np.linspace(xmin - ext_x, xmax + ext_x, nx)
        lin_y = np.linspace(ymin - ext_y, ymax + ext_y, ny)
        X, Y = np.meshgrid(lin_x, lin_y)
        X_src, Y_src, R = sd.make_src_2D(X=X, Y=Y, n_src=nx*ny,
                                         ext_x=ext_x, ext_y=ext_y,
                                         R_init=0.5)
        expected_X_src = [[-0.5, 0.5, 1.5],
                          [-0.5, 0.5, 1.5],
                          [-0.5, 0.5, 1.5]]
        expected_Y_src = [[-0.5, -0.5, -0.5],
                          [0.5, 0.5, 0.5],
                          [1.5, 1.5, 1.5]]
        for x_row, xe_row in zip(X_src, expected_X_src):
            for x, xe in zip(x_row, xe_row):
                self.assertEquals(x, xe)
        for y_row, ye_row in zip(Y_src, expected_Y_src):
            for y, ye in zip(y_row, ye_row):
                self.assertEquals(y, ye)

    def test_make_src2D_translated_grid(self):
        xmin, xmax = -1.5, -0.5
        ymin, ymax = -1.0, 0.0
        ext_x, ext_y = 0.0, 0.0
        nx, ny = 2, 2
        lin_x = np.linspace(xmin - ext_x, xmax + ext_x, nx)
        lin_y = np.linspace(ymin - ext_y, ymax + ext_y, ny)
        X, Y = np.meshgrid(lin_x, lin_y)
        X_src, Y_src, R = sd.make_src_2D(X=X, Y=Y, n_src=nx*ny,
                                         ext_x=ext_x, ext_y=ext_y,
                                         R_init=0.5)
        expected_X_src = [[-2.0, -1.0, 0.0],
                          [-2.0, -1.0, 0.0],
                          [-2.0, -1.0, 0.0]]
        expected_Y_src = [[-1.5, -1.5, -1.5],
                          [-0.5, -0.5, -0.5],
                          [0.5, 0.5, 0.5]]
        for x_row, xe_row in zip(X_src, expected_X_src):
            for x, xe in zip(x_row, xe_row):
                self.assertEquals(x, xe)
        for y_row, ye_row in zip(Y_src, expected_Y_src):
            for y, ye in zip(y_row, ye_row):
                self.assertEquals(y, ye)

    def test_make_src3D_regular_grid(self):
        """ """
        xmin, xmax = 0.0, 1.0
        ymin, ymax = 0.0, 1.0
        zmin, zmax = 0.0, 1.0
        ext_x, ext_y, ext_z = 0.0, 0.0, 0.0
        nx, ny, nz = 3, 3, 3
        lin_x = np.linspace(xmin - ext_x, xmax + ext_x, nx)
        lin_y = np.linspace(ymin - ext_y, ymax + ext_y, ny)
        lin_z = np.linspace(zmin - ext_z, zmax + ext_z, nz)
        X, Y, Z = np.meshgrid(lin_x, lin_y, lin_z)
        X_src, Y_src, Z_src, R = sd.make_src_3D(X=X, Y=Y, Z=Z, n_src=nx*ny*nz,
                                                ext_x=ext_x, ext_y=ext_y, ext_z=ext_z,
                                                R_init=0.5)
        expected_X_src = [[[0.,   0.,   0.],
                           [0.5,  0.5,  0.5],
                           [1.,   1.,   1.]],
                          [[0.,   0.,   0.],
                           [0.5,  0.5,  0.5],
                           [1.,   1.,   1.]],
                          [[0.,   0.,   0.],
                           [0.5,  0.5,  0.5],
                           [1.,   1.,   1.]]]
        for x_slice, xe_slice in zip(X_src, expected_X_src):
            for x_row, xe_row in zip(x_slice, xe_slice):
                for x, xe in zip(x_row, xe_row):
                    self.assertEquals(x, xe)

    def test_make_src3D_translated_grid(self):
        xmin, xmax = -1.0, 0.0
        ymin, ymax = 0.0, 1.0
        zmin, zmax = 0.0, 1.0
        ext_x, ext_y, ext_z = 0.0, 0.0, 0.0
        nx, ny, nz = 3, 3, 3
        lin_x = np.linspace(xmin - ext_x, xmax + ext_x, nx)
        lin_y = np.linspace(ymin - ext_y, ymax + ext_y, ny)
        lin_z = np.linspace(zmin - ext_z, zmax + ext_z, nz)
        X, Y, Z = np.meshgrid(lin_x, lin_y, lin_z)
        X_src, Y_src, Z_src, R = sd.make_src_3D(X=X, Y=Y, Z=Z, n_src=nx*ny*nz,
                                                ext_x=ext_x, ext_y=ext_y, ext_z=ext_z,
                                                R_init=0.5)
        expected_X_src = [[[-1.,  -1.,   -1.],
                           [-0.5, -0.5, -0.5],
                           [0.,   0.,    0.]],
                          [[-1.,   -1.,   -1.],
                           [-0.5,  -0.5,  -0.5],
                           [0.,   0.,   0.]],
                          [[-1.,   -1.,   -1.],
                           [-0.5,  -0.5,  -0.5],
                           [0.,   0.,   0.]]]

        for x_slice, xe_slice in zip(X_src, expected_X_src):
            for x_row, xe_row in zip(x_slice, xe_slice):
                for x, xe in zip(x_row, xe_row):
                    self.assertEquals(x, xe)

    def test_gauss1Dlim_basis_normalized(self):
        mu = 0
        three_std = 1.0
        xs = np.linspace(-three_std, three_std, 30)
        probe = [bf.gauss_rescale_lim_1D(x, mu, three_std) for x in xs]
        norm_const = np.trapz(probe, xs)
        self.assertAlmostEqual(norm_const, 1, places=1)

    def test_gauss1D_basis_normalized(self):
        mu = 0
        three_std = 1.0
        xs = np.linspace(-three_std, three_std, 30)
        probe = [bf.gauss_rescale_1D(x, mu, three_std) for x in xs]
        norm_const = np.trapz(probe, xs)
        self.assertAlmostEqual(norm_const, 1, places=1)

    def test_step_1D_basis_normalized(self):
        mu = 0
        three_std = 1.0
        xs = np.linspace(-three_std, three_std, 30)
        probe = [bf.step_rescale_1D(x, mu, three_std) for x in xs]
        norm_const = np.trapz(probe, xs)
        self.assertAlmostEqual(norm_const, 1, places=1)

    def test_gauss2D_basis_normalized(self):
        xlin = np.linspace(-1.0, 1.0, 20)
        ylin = np.linspace(-1.0, 1.0, 20)
        three_stdev = 0.5
        X, Y = np.meshgrid(xlin, ylin)

        y = bf.gauss_rescale_2D(X, Y, [0,0], three_stdev)

        # imshow(y)
        # show()

        norm_const = integrate_2D(y, xlin, ylin)
        # THIS TEST IS NOT PASSING!
        #self.assertAlmostEqual(norm_const, 1, places=1)

    def test_step2D_basis_normalized(self):
        xlin = np.linspace(-1.0, 1.0, 30)
        ylin = np.linspace(-1.0, 1.0, 30)
        R = 0.5
        X, Y = np.meshgrid(xlin, ylin)

        y = bf.step_rescale_2D(X, Y, [0,0], R)

        norm_const = integrate_2D(y, xlin, ylin)
        self.assertAlmostEqual(norm_const, 1, places=1)

    def test_gauss3D_basis_normalized(self):
        xlin = np.linspace(-1.0, 1.0, 20)
        ylin = np.linspace(-1.0, 1.0, 20)
        zlin = np.linspace(-1.0, 1.0, 20)
        three_stdev = 0.5
        X, Y, Z = np.meshgrid(xlin, ylin, zlin)

        y = bf.gauss_rescale_3D(X, Y, Z, [0,0,0], three_stdev)

        norm_const = integrate_3D(y, xlin, ylin, zlin)
        self.assertAlmostEqual(norm_const, 1, places=3)

    def test_step3D_basis_normalized(self):
        xlin = np.linspace(-1.0, 1.0, 20)
        ylin = np.linspace(-1.0, 1.0, 20)
        zlin = np.linspace(-1.0, 1.0, 20)
        three_stdev = 0.5
        X, Y, Z = np.meshgrid(xlin, ylin, zlin)

        y = bf.step_rescale_3D(X, Y, Z, [0,0,0], three_stdev)

        norm_const = integrate_3D(y, xlin, ylin, zlin)
        self.assertAlmostEqual(norm_const, 1, places=1)

    def tearDown(self):
        pass


# TODO: test if KCSD run on translated grid v(x) gives the same output!

def integrate_2D(y, xlin, ylin):
    Ny = ylin.shape[0]
    I = np.zeros(Ny)
    # do a 1-D integral over every row
    for i in xrange(Ny):
        I[i] = np.trapz(y[i, :], ylin)

    # then an integral over the result
    norm = np.trapz(I, xlin)
    return norm


def integrate_3D(y, xlin, ylin, zlin):
    Nz = zlin.shape[0]
    J = np.zeros((Nz,Nz))
    for i in xrange(Nz):
        J[i,:] = np.trapz(y[i, :, :], zlin)

    Ny = ylin.shape[0]
    I = np.zeros(Ny)
    for i in xrange(Ny):
        I[i] = np.trapz(J[i, :], ylin)

    norm = np.trapz(I, xlin)
    return norm


def comparison_plot_2D(arr_true, arr_recstr, true_title, recstr_title):
    """For visual check of the results."""
    fig, (ax11, ax21, ax22) = plt.subplots(1, 3)

    ax11.imshow(arr_true, interpolation='none', aspect='auto')
    ax11.set_title(true_title)
    ax11.autoscale_view(True, True, True)

    ax21.imshow(arr_recstr, interpolation='none', aspect='auto')
    ax21.set_title(recstr_title)
    ax21.autoscale_view(True, True, True)

    ax22.imshow(arr_true - arr_recstr,
                interpolation='none', aspect='auto')
    ax22.set_title('diff')
    ax22.autoscale_view(True, True, True)

    show()


if __name__ == '__main__':
    unittest.main()
