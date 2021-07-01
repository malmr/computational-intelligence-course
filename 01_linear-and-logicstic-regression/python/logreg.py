#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # Hint:
    #   - use the logistic function sig imported from the file toolbox
    #   - sums of logs of numbers close to zero might lead to numerical errors, try splitting the cost into the sum
    # over positive and negative samples to overcome the problem. If the problem remains note that low errors is not
    # necessarily a problem for gradient descent because only the gradient of the cost is used for the parameter updates.

    c = 0
    for i in range(N):
        c += (y[i]*np.log(sig(np.dot(x[i], theta))) + (1-y[i])*np.log(1 - sig(np.dot(x[i], theta))))
    c /= -N

    #print("cost is: ",c)
    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    g = np.zeros(theta.shape)

    for j in range(len(theta)):
        for i in range(N):
            g[j] += (sig(np.dot(x[i], theta)) - y[i]) * x[i][j]
    g /= N

    return g
