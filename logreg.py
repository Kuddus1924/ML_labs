import numpy as np

import util

def loss(Y, sigm):
    loss = Y.T @ np.log(sigm) + (1 - Y).T @ np.log(1 - sigm)
    return -loss[0][0] / Y.shape[0]

class LogregGradientDescent:
    def __init__(self, A, Y):
        self.__A = A
        self.__Y = Y
        self.reset()

    def reset(self):
        self.__init_theta(self.__A.shape[1])
        self.__update_sigmoid()

    def __init_theta(self, size):
        self.__theta = np.random.rand(size, 1)

    def __update_sigmoid(self):
        self.__sigm = util.sigmoid(self.__A @ self.__theta)

    def __gradient(self):
        diff = (self.__sigm - self.__Y)
        return (self.__A.T @ diff) / self.__A.shape[0]

    def update_theta(self, speed):
        self.__theta -= speed * self.__gradient()
        self.__update_sigmoid()

    def predict(self, A):
        sigm = util.sigmoid(A @ self.__theta)
        return np.round(sigm)

    def predict_loss(self, A, Y):
        sigm = util.sigmoid(A @ self.__theta)
        return loss(Y, sigm)

    def theta(self):
        return self.__theta.copy()

    def loss(self):
        return loss(self.__Y, self.__sigm)
