#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rd

import logreg

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)


This file contains useful functions to be used in the assignment.
Non of the function need to be modified by the student, they are:
- sig: logistic function
- poly_2D_design_matrix: create the design matrix
- plot_logreg: plotting the results
- check_gradient: small test to verify if the gradient is correctly computed
"""


def sig(x):
    """
    Sigmoid (logistic) funtion
    :param x:
    :return:
    """
    return 1. / (1 + np.exp(-x))


def poly_2D_design_matrix(x1, x2, degree):
    """
    Create the expanded design matrix with all polynoms of x1 and x2 up to degree

    :param x1:
    :param x2:
    :param degree:
    :return:
    """
    N = x1.size

    design_matrix = np.zeros((N, 0))
    for k1 in range(degree + 1):
        for k2 in range(degree + 1):
            if k1 + k2 <= degree:
                X = x1 ** k1 * x2 ** k2
                design_matrix = np.hstack((design_matrix, X.reshape((N, 1))))

    return design_matrix


def plot_logreg(data, degree, theta, E_list):
    """
    Plot the results for the logistic regression exercice

    :param data:
    :param degree:
    :param theta:
    :param Jtrain:
    :param Jtest:
    :return:
    """

    # Plot the list of errors
    if len(E_list) > 0:
        fig, ax = plt.subplots(1)
        ax.plot(E_list, linewidth=2)
        ax.set_xlabel('Iteration number')
        ax.set_ylabel('Error')
        ax.set_title('Error monitoring')

    # Plot the solution on with training and testing set
    K = 100
    x_min = -1
    x_max = 1

    [xx1, xx2] = np.meshgrid(np.linspace(x_min, x_max, K), np.linspace(x_min, x_max, K))

    XX = poly_2D_design_matrix(xx1.reshape(K ** 2, 1), xx2.reshape(K ** 2, 1), degree)
    hh = sig(XX.dot(theta)).reshape(K, K)

    fig, ax = plt.subplots(1, 2)

    for k, st in zip([0, 1], ['train', 'test']):
        x1 = data['x1_' + st]
        x2 = data['x2_' + st]
        y = data['y_' + st]

        XX = poly_2D_design_matrix(x1, x2, degree)
        J = np.asscalar(logreg.cost(theta, XX, y))

        ax[k].pcolor(xx1, xx2, hh, cmap='cool')

        pos, neg = y, np.logical_not(y)
        ax[k].scatter(x1[pos], x2[pos], marker='o', color='black', s=20)
        ax[k].scatter(x1[neg], x2[neg], marker='x', color='white', s=40)

        ax[k].contour(xx1, xx2, hh, levels=[.5], linewidths=[3])

        ax[k].set_xlabel('x1')
        if k == 0:
            ax[k].set_ylabel('x2')

        ax[k].set_title('{} set J: {:.3f}'.format(st, J))

        # cbar = plt.colorbar()
        # cbar.ax.set_ylabel('h(x)')
        ax[k].set_xlim([x_min, x_max])
        ax[k].set_ylim([x_min, x_max])


def check_gradient(f, df, n, tries=1, deltas=(1e-2, 1e-4, 1e-6)):
    # Init around the point x0
    x0 = rd.randn(n)
    f0 = f(x0)
    g0 = df(x0)

    # For different variations tries if the gradient is well approximated with finite difference
    for k_dx in range(tries):
        dx = rd.randn(x0.size)

        approx_err = np.zeros(len(deltas))
        df_g = np.inner(dx, g0)

        for k, d in enumerate(deltas):
            f1 = f(x0 + d * dx)
            df = (f1 - f0) / d
            approx_err[k] = np.log10(np.abs(df_g - df) + 1e-20)

        if (np.diff(approx_err) < -1).all() or (approx_err < -20).all():
            print('Gradient security check OK: the gradient df is well approximated by finite difference.')

        else:
            raise ValueError(
                '''GRADIENT SECURITY CHECK ERROR:
                Detected by approximating the gradient with finite difference.
                The cost function or the gradient are not correctly computed.
                The approximation D(eta) = (f(x + eta dx) - f(x)) / eta should converge toward df=grad*dx.

                Instead \t D({:.3g}) = {:.3g} \t df = {:.3g}
                Overall for

                \t \t  eta \t \t \t {}
                log10( |D(eta) - df|) \t {} '''.format(d, df, df_g, deltas, approx_err))
