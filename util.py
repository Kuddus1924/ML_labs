import numpy as np

def split(A, boundary):
    edge = int(A.shape[0] * boundary)
    top = A[:edge]
    bottom = A[edge:]
    return (top, bottom)

def symmetric_permutation(A, Y):
    combination = np.hstack((Y, A))
    combination = np.random.permutation(combination)
    Y = combination[:, :1]
    A = combination[:, 1:]
    return (A, Y)

def standardization(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

def combine_with_unit_column(X):
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))

def sigmoid(X):
    return 1 / (1 + np.exp(-X))