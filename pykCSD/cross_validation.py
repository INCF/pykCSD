import numpy as np
from numpy import dot, identity
from numpy.linalg import norm, inv

"""
This module contains routines for cross validation, which is used
to find the regularization parameter in the KCSD method
"""


def choose_lambda(lambdas, sampled_pots, k_pot, elec_pos, index_generator):
    """
    Finds the optimal regularization parameter lambda
    for Tikhonov regularization using cross validation.

    **Parameters**

    lambdas: list-like
        regularization parameters set to choose from
    
    index_generator: callable
        generator of training and testing indices, for example:

        from sklearn.cross_validation import KFold, LeaveOneOut, ShuffleSplit

        index_generator = KFold(n, n_folds=10, indices=True)
        index_generator = ShuffleSplit(5, n_iter=15, test_size=0.25, indices=True)
        index_generator = LeavePOut(n, 2)
        index_generator = LeaveOneOut(n_elec, indices=True)
    """
    n = len(lambdas)
    errors = np.zeros(n)
    for i, lambd in enumerate(lambdas):
        errors[i] = cross_validation(
            lambd,
            sampled_pots,
            k_pot,
            index_generator
        )
    return lambdas[errors == min(errors)][0]


def cross_validation(lambd, pot, k_pot, index_generator):
    """
    Calculate error using LeaveOneOut or KFold cross validation.
    """
    errors = []

    for ind_train, ind_test in index_generator:
        err = calc_CV_error(lambd, pot, k_pot, ind_test, ind_train)
        errors.append(err)

    error = np.mean(errors)
    # print "l=", lambd, ", err=", error
    return error


def calc_CV_error(lambd, pot, k_pot, ind_test, ind_train):
    k_train = k_pot[ind_train, ind_train]

    pot_train = pot[ind_train]
    pot_test = pot[ind_test]

    try:
        beta = dot(inv(k_train + lambd * identity(k_train.shape[0])), pot_train)
    except Exception:
        #if the matrix is not invertible, then return a high error
        err = 100000
        return err

    # k_cross = k_pot[np.array([ind_test, ind_train])]
    k_cross = k_pot[ind_test][:, ind_train]

    pot_est = dot(k_cross, beta)

    err = norm(pot_test - pot_est)
    return err
