#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import poly

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file defines the plotting functions used the 'main_poly.py' and 'main_poly_model_selection.py'.
There is nothing to modify here.
"""


def plot_poly(data, degree, theta_opt, n_line_precision=100):
    """
    Create of plot that shows the polynomial expansion and the fit as compared to the scattered data sets.

    :param data:
    :param degree:
    :param theta_opt:
    :param n_line_precision:
    :return:
    """

    fig, ax_list = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.4)

    # Plot the polynomials
    xx = np.linspace(-1, 1, n_line_precision).reshape((n_line_precision, 1))
    XX = poly.design_matrix(xx, degree)

    ax_list[0, 0].plot(xx, XX, linewidth=3)

    ax_list[0, 0].set_xlabel('x')
    ax_list[0, 0].set_ylabel('x^n')

    ax_list[0, 0].set_xlim([-1, 1])
    ax_list[0, 0].set_ylim([-1, 1])
    ax_list[0, 0].set_title('Polynomial up to degree {}'.format(degree))

    # Computation the predicted model
    y_pred = XX.dot(theta_opt)

    # Plot the fit on training data
    As = [(0, 1), (1, 0), (1, 1)]
    Xs = ['x_train', 'x_val', 'x_test']
    Ys = ['y_train', 'y_val', 'y_test']
    Cs = ['blue', 'red', 'purple']
    Titles = ['train', 'validation', 'test']

    for a, x, y, c, ti in zip(As, Xs, Ys, Cs, Titles):
        # Plot
        ax_list[a].plot(xx, y_pred, color='black', linewidth=3)
        ax_list[a].scatter(data[x], data[y], color=c, label=ti + ' set')

        ax_list[a].set_xlabel('x')
        ax_list[a].set_ylabel('y')

        mse = poly.compute_error(theta_opt, degree, data[x], data[y])
        mse=round(float(mse),3)

        #ax_list[a].set_title('Set {} (MSE {:.3g}) '.format(ti, mse))
        ax_list[a].set_title('Set {} (MSE {}) '.format(ti, mse))
        ax_list[a].set_xlim([-1, 1])
        ax_list[a].set_ylim([-5, 5])


def plot_errors(i_best, degrees, mse_train, mse_val, mse_test):
    """
    Display the evolution of the error when the degree is increasing

    :param i_best:
    :param degrees:
    :param mse_train:
    :param mse_val:
    :param mse_test:
    :return:
    """

    for mse, lab in zip([mse_train, mse_val, mse_test], ['train', 'val', 'test']):

        plt.plot(degrees, mse, label=lab, linewidth=3)

    plt.ylim([0, 1])
    plt.axvline(x=degrees[i_best], color='black', linestyle='--', linewidth=3,
                label='Optimal degree {}'.format(degrees[i_best]))
    plt.xlabel('Degrees')
    plt.ylabel('MSE')
    plt.legend()
