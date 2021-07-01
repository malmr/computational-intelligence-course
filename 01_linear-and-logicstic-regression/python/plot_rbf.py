#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with radial basis functions

This file defines the plotting functions used the 'main_poly.py' and 'main_poly_model_selection.py'.
TODO Fill in fn plot_errors
"""


def plot_rbf(data, n_center, theta_opt, n_line_precision=100):
    """
    Create of plot that shows the RBF expansion and the fit as compared to the scattered data sets.

    :param data:
    :param n_center:
    :param theta_opt:
    :param n_line_precision:
    :return:
    """

    fig, ax_list = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.4)
	
    # Plot the polynomials
    xx = np.linspace(-1, 1, n_line_precision).reshape((n_line_precision, 1))
    centers, sigma = rbf.get_centers_and_sigma(n_center)
    XX = rbf.design_matrix(xx, centers, sigma)

    ax_list[0, 0].plot(xx, XX, linewidth=3)

    ax_list[0, 0].set_xlabel('x')
    ax_list[0, 0].set_ylabel('y')


    ax_list[0, 0].set_xlim([-1, 1])
    ax_list[0, 0].set_ylim([0, 1])
    ax_list[0, 0].set_title('{} RBF kernels'.format(n_center))

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

        mse = rbf.compute_error(theta_opt, n_center, data[x], data[y])

        mse = round(float(mse),2)
        #ax_list[a].set_title('Set {} (MSE {:.3g}) '.format(ti, mse))
        ax_list[a].set_title('Set {} (MSE {}) '.format(ti, mse))
        ax_list[a].set_xlim([-1, 1])
        ax_list[a].set_ylim([-5, 5])


def plot_errors(i_best, n_centers, mse_train, mse_val, mse_test):
    """
    Display the evolution of the error when the center number is increasing

    :param i_best:
    :param n_centers:
    :param mse_train:
    :param mse_val:
    :param mse_test:
    :return:
    """

    for mse, lab in zip([mse_train, mse_val, mse_test], ['train', 'val', 'test']):

        plt.plot(n_centers, mse, label=lab, linewidth=3)

    plt.ylim([0, 1])
    plt.axvline(x=n_centers[i_best], color='black', linestyle='--', linewidth=3,
                label='Optimal degree {}'.format(n_centers[i_best]))
    plt.xlabel('n of centers')
    plt.ylabel('MSE')
    plt.legend()