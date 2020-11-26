import numpy as np

def theta_via_least_squares(A, Y):
    return np.linalg.pinv(A) @ Y

def loss(diff):
    sum_squares = (diff.T @ diff)[0][0]
    return sum_squares / (2 * diff.shape[0])

class LinregGradientDescent:
    def __init__(self, A, Y):
        self.__A = A
        self.__Y = Y
        self.reset()

    def reset(self):
        self.__init_theta(self.__A.shape[1])
        self.__update_difference()

    def __init_theta(self, size):
        self.__theta = np.random.rand(size, 1)

    def __update_difference(self):
        self.__diff = self.__A @ self.__theta - self.__Y

    def __gradient(self):
        return (self.__A.T @ self.__diff) / self.__A.shape[0]

    def update_theta(self, speed):
        self.__theta -= speed * self.__gradient()
        self.__update_difference()

    def theta(self):
        return self.__theta.copy()

    def loss(self):
        return loss(self.__diff)

    def predict(self, A):
        return A @ self.__theta

    def predict_loss(self, A, Y):
        return loss((self.predict(A) - Y))