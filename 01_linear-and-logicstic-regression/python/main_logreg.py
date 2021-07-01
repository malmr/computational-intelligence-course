#!/usr/bin/env python
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gradient_descent as gd
import logreg as lr
import logreg_toolbox

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression

This is the main file that loads the data, compute the solution and plot the results.
"""


def main():
    # Set parameters
    degree = 15
    eta = 10
    max_iter = 500

    # Load data and expand with polynomial features
    f = open('data_logreg.json', 'r')
    data = json.load(f)
    for k, v in data.items(): data[k] = np.array(v)  # Encode list into numpy array

    # Expand with polynomial features
    X_train = logreg_toolbox.poly_2D_design_matrix(data['x1_train'], data['x2_train'], degree)
    n = X_train.shape[1]

    # Define the functions of the parameter we want to optimize
    def f(theta): return lr.cost(theta, X_train, data['y_train'])

    def df(theta): return lr.grad(theta, X_train, data['y_train'])

    # Test to verify if the computation of the gradient is correct
    logreg_toolbox.check_gradient(f, df, n)

    # Point for initialization of gradient descent
    theta0 = np.zeros(n)

    #### VARIANT 1: Optimize with gradient descent
    theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)

    #### VARIANT 2: Optimize with gradient descent
    """
    theta_opt, E_list, lr_list = gd.adaptative_gradient_descent(f, df, theta0, eta, max_iter)
    plt.plot(lr_list)
    plt.xlabel('Iterations')
    plt.ylabel('Learning rate')
    print('Adaptative gradient, final learning rate: {:.3g}'.format(lr_list[-1]))
    """

    #### VARIANT 3: Optimize with gradient descent
    #res = minimize(f, x0=theta0, jac=df, options={'disp': True ,'maxiter': max_iter})
    #theta_opt = res.x.reshape((n, 1))
    #E_list = []

    logreg_toolbox.plot_logreg(data, degree, theta_opt, E_list)
    plt.show()


if __name__ == '__main__':
    main()
