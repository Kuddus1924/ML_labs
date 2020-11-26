import numpy as np
import matplotlib.pyplot as plt

import csvio
import util

def predict_via_iterations(xtrain, ytrain, xtest, ytest, regclass, speed, iterations, boundary):
    A, Y = csvio.load_train_data(xtrain, ytrain)
    A, Y = util.symmetric_permutation(A, Y)
    A_train, A_control = util.split(A, boundary)
    Y_train, Y_control = util.split(Y, boundary)

    regression = regclass(A_train, Y_train)
    losses, losses_control = train_via_iterations(regression, speed, iterations, A_control, Y_control)

    A_test = csvio.load_regressors(xtest)
    prediction = regression.predict(A_test)
    np.savetxt(ytest, prediction, fmt='%.5e', delimiter=',')

    plt.figure()
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.plot(range(1, len(losses) + 1), losses, label='losses during studies')
    plt.plot(range(1, len(losses_control) + 1), losses_control, label='loss during control')
    plt.legend(loc='upper right', framealpha=0.95)
    plt.draw()
    plt.pause(0.001)

def train_via_iterations(regression, speed, iterations, A_control, Y_control):
    losses = []
    losses_control = []

    regression.reset()
    for _ in range(iterations):
        regression.update_theta(speed)
        losses.append(regression.loss())
        losses_control.append(regression.predict_loss(A_control, Y_control))

    return (losses, losses_control)

def research_speed_via_iterations(xtrain, ytrain, regclass, speed_range, iterations):
    A, Y = csvio.load_train_data(xtrain, ytrain)
    regression = regclass(A, Y)

    losses = []
    for speed in np.arange(*speed_range):
        loss = estimate_speed_via_iterations(regression, speed, iterations)
        losses.append(loss)

    plt.figure()
    plt.ylabel('loss')
    plt.xlabel('\u03B1')
    plt.plot(np.arange(*speed_range), losses, label='dependence of losses on speed')
    plt.draw()
    plt.pause(0.001)

def estimate_speed_via_iterations(regression, speed, iterations):
    regression.reset()
    for _ in range(iterations):
        regression.update_theta(speed)
    return regression.loss()


def train_via_epsilon(regression, speed, epsilon):
    regression.reset()
    current_loss = regression.loss()
    previous_loss = -1

    while abs(current_loss - previous_loss) > epsilon:
        previous_loss = current_loss
        regression.update_theta(speed)
        current_loss = regression.loss()

    return regression