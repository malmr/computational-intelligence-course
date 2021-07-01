import json

import numpy as np
from sklearn.preprocessing import scale

from nn_regression import ex_1_1_a, ex_1_1_b, ex_1_1_c, ex_1_1_d, ex_1_2_a, ex_1_2_b, ex_1_2_c

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file loads the data and calls the functions for each section of the assignment.bak.
"""

__author__ = 'bellec,subramoney'


def load_data():
    """
    Loads the data from data.json
    :return: A dictionary containing keys x_train, x_test, y_train, y_test
    """
    with open('data.json', 'r') as f:
        raw_data = json.load(f)

    data = {}
    # Convert arrays in the raw_data to numpy arrays
    for key, value in raw_data.items():
        data[key] = scale(np.array(value))

    # Let's reduce the size
    data['x_test'] = data['x_test'][0:10000:10]
    data['y_test'] = data['y_test'][0:10000:10]

    return data


def main():
    data = load_data()
    x_train, x_test, y_train, y_test = \
        data['x_train'], data['x_test'], data['y_train'].ravel(), data['y_test'].ravel()



    ## 1.1 a)
    #ex_1_1_a(x_train, x_test, y_train, y_test)

    # 1.1 b)
    #ex_1_1_b(x_train, x_test, y_train, y_test)

    # 1.1 c)
    #ex_1_1_c(x_train, x_test, y_train, y_test)

    # 1.1 d)
    #ex_1_1_d(x_train, x_test, y_train, y_test)

    ## 1.2 a)
    #ex_1_2_a(x_train, x_test, y_train, y_test)

    # Add noise to the data:
    x_train_noisy = x_train + np.random.randn(60, 1) * .5
    y_train_noisy = y_train + np.random.randn(60) * .5

    ## 1.2 b)
    #ex_1_2_b(x_train_noisy, x_test, y_train_noisy, y_test)

    ## 1.2 c)
    ex_1_2_c(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
