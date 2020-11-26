import numpy as np

import util

def load_regressand(file):
    return np.loadtxt(file).reshape(-1, 1)

def load_regressors(file, bias=True, standardize=True):
    X = np.loadtxt(file, delimiter=',')
    if standardize:
        X = util.standardization(X)
    return util.combine_with_unit_column(X) if bias else X

def load_train_data(xfile, yfile, bias=True, standardize=True):
    A = load_regressors(xfile, bias, standardize)
    Y = load_regressand(yfile)
    return (A, Y)