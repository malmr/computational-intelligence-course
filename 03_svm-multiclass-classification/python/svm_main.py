import json

import numpy as np

from svm import ex_1_c, ex_1_b, ex_1_a, ex_2_a, ex_2_b, ex_2_c, ex_3_a, ex_3_b
from svm_plot import plot_data_points, plot_mnist

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

This file loads the data and calls the functions for each section of hw4.
"""

__author__ = 'bellec,subramoney'


def load_data(filename):
    """
    Loads the data from data.json
    :return: A dictionary containing keys x_train, x_test, y_train, y_test
    """
    with open(filename, 'r') as f:
        raw_data = json.load(f)

    data = {}
    # Convert arrays in the raw_data to numpy arrays
    for key, value in raw_data.items():
        data[key] = np.array(value)
    return data


def ex_1():
    data = load_data('data.json')
    x, y = data['X'], data['Y'].ravel()

    plot_data_points(x, y)

    ex_1_a(x, y)
    ex_1_b(x, y)
    ex_1_c(x, y)


def ex_2():
    data = load_data('data_nl.json')
    x_train, y_train, x_test, y_test = \
        data['X'], data['Y'].ravel(), data['XT'], data['YT'].ravel()

    plot_data_points(x_train, y_train, x_test, y_test)

    ex_2_a(x_train, y_train, x_test, y_test)
    ex_2_b(x_train, y_train, x_test, y_test)
    ex_2_c(x_train, y_train, x_test, y_test)


def ex_3():
    data = load_data('data_mnist.json')
    # Normalize data from [0,255] to [0,1]
    x_train, y_train, x_test, y_test = \
        data['X'] / 255, data['Y'].ravel(), data['XT'] / 255, data['YT'].ravel()

    #plot_mnist(x_train, y_train)

    #ex_3_a(x_train, y_train, x_test, y_test)
    ex_3_b(x_train, y_train, x_test, y_test)


def main():
    #ex_1()
    #ex_2()
    ex_3()


if __name__ == '__main__':
    main()
