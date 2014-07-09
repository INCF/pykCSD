import numpy as np
from numpy import dot, identity
from numpy.linalg import norm, inv
from sklearn.cross_validation import KFold, LeaveOneOut, ShuffleSplit

"""
This module contains routines for cross validation, which is used
to find the regularization parameter in the KCSD method
"""


def calc_CV_error(lambd, pot, k_pot, ind_test, ind_train):
    k_train = k_pot[ind_train, ind_train]

    pot_train = pot[ind_train]
    pot_test = pot[ind_test]

    beta = dot(inv(k_train + lambd * identity(k_train.shape[0])), pot_train)

    # k_cross = k_pot[np.array([ind_test, ind_train])]
    k_cross = k_pot[ind_test][:, ind_train]

    pot_est = dot(k_cross, beta)

    err = norm(pot_test - pot_est)
    return err


def cross_validation(lambd, pot, k_pot, n_elec, n_folds):
    """
    Calculate error using LeaveOneOut or KFold cross validation.
    """
    errors = []

    # ind_generator = KFold(n, n_folds=n_folds, indices=True)
    ind_generator = LeaveOneOut(n_elec, indices=True)
    # ind_generator = ShuffleSplit(5, n_iter=15, test_size=0.25, indices=True)
    # ind_generator = LeavePOut(len(Y), 2)
    for ind_train, ind_test in ind_generator:
        err = calc_CV_error(lambd, pot, k_pot, ind_test, ind_train)
        errors.append(err)

    error = np.mean(errors)
    # print "l=", lambd, ", err=", error
    return error