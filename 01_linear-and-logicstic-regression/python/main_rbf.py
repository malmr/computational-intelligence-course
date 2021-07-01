#!/usr/bin/env python
import json
import matplotlib.pyplot as plt
from plot_rbf import plot_rbf
import rbf
import numpy as np

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with radial basis functions

This file:
1) loads the data from 'data_linreg.json'
2) trains and test a linear regression model for a given number of RBF centers
3) plots the results

TODO boxes are to be found in 'rbf.py'
"""


def main():
    # Set the n_centers of the polynomial expansion
    n_centers = 5

    data_path = 'data_linreg.json'

    # Load the data
    f = open(data_path, 'r')
    data = json.load(f)
    for k, v in data.items():
        data[k] = np.array(v).reshape((len(v), 1))

    # Print the training and testing errors
    theta, err_train, err_val, err_test = rbf.train_and_test(data, n_centers)
    print('Training error {} \t Validation error {} \t Testing error {} '.format(err_train, err_val, err_test))

    plot_rbf(data, n_centers, theta)
    plt.show()


if __name__ == '__main__':
    main()
