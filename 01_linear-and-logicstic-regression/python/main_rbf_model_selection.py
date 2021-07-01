#!/usr/bin/env python
import numpy as np
import json
import matplotlib.pyplot as plt
from plot_rbf import plot_rbf, plot_errors
import rbf

__author__ = 'bellec, subramoney'
"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file:
1) loads the data from 'data_linreg.json'
2) trains and test a linear regression model for K different numbers of RBF centers
3) TODO: Select the best cluster center number
3) plots the optimal results

TODO boxes are to be found here and in 'rbf.py'
"""


def main():
    # Number of possible degrees to be tested
    K = 40
    data_path = 'data_linreg.json'

    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    ######################
    #
    # TODO
    #
    # Compute the arrays containing the Means Square Errors in all cases
    #
    # Find the degree that minimizes the validation error
    # Store it in the variable i_best for plotting the results
    #
    # TIP:
    # - You are invited to adapt the code you did for the polynomial  case
    # - use the argmin function of numpy
    #

    # Init vectors storing MSE (Mean square error) values of each sets at each degrees
    mse_train = np.zeros(K)
    mse_val = np.zeros(K)
    mse_test = np.zeros(K)
    theta_list = np.zeros(K, dtype=object)
    n_centers = np.arange(K) + 1



    # Compute the MSE values
    for i in range(K):
        theta_list[i], mse_train[i], mse_val[i], mse_test[i] = rbf.train_and_test(data, n_centers[i])

    i_best = np.argmin(mse_val)
    # TODO: Change this


    #
    # TODO END
    ######################

    # Plot the training error as a function of the degrees
    plot_errors(i_best, n_centers, mse_train, mse_val, mse_test)
    plot_rbf(data, n_centers[i_best], theta_list[i_best])
    plt.show()


if __name__ == '__main__':
    main()
