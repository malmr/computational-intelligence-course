#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Linear Regression with radial basis functions

This file contains the main work to be done.
The functions are:
- TODO get_centers_and_sigma: Compte the centers as explained in the assignement hand-out sheet
- TODO design_matrix: Create the design matrix including the rbf expansions and the constant feature
- TODO train: finds the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Square Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""


def get_centers_and_sigma(n_centers):
    """
    Create for a given center numbers the numpy array containing the centers and provide a good sigma
    :param n_centers:
    :return:
    """

    ######################
    #
    # TODO
    #
    # TIPs:
    #   - Use the linspace function from numpy
    #   - ATTENTION: You might want to write the division a / b, with a and b integers but in python 2 this will perform
    #       an euclidian division. The solution is to convert one of the two integers to float.
    #

    centers = np.linspace(-1.0, 1.0, n_centers)
    sigma = 2/n_centers

    # END TODO
    ######################

    return centers, sigma


def design_matrix(x, centers, sigma):
    """
    Creates the design matrix given the data x.
    The design matrix is built out of radial basis functions.
    Those are family of gaussians of width sigma, each of them in centered at one of the centered specified in 'centers'.
    The first row is one for all input data.


    E.g: for the data x = [0,1,2], the centers [0,1] and sigma = 1/sqrt(2)
    the function should return: [[1, exp(0), exp(-1)],
								 [1, exp(-1), exp(0)],
								 [1, exp(-4), exp(-1)]] 

    :param x: numpy array of shape (N,1)
    :param centers: List of centers
    :param sigma: parameter to control the width of the RBF
    :return: Expanded data in a numpy array of shape (N,n_centers+1)
    """

    ######################
    #
    # TODO
    #
    # Return the numpy array of shape (N,n_centers+1)
    # Storing the data of the form exp(- (x_i - c_j) ^2 / (2 sigma^2) ) at row i and column j+1
    # Look at the function description for more info
    #
    # TIP: don't forget that the first row has only ones
    #

    X = np.ones(len(x))
    X.shape = (len(x), 1)

    for i in range(len(centers)):
        X = np.concatenate((X, np.exp(-(x-centers[i])**2/(2*sigma**2))), axis=1)


    # END TODO
    ######################

    return X


def train(x, y, n_centers):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the RBF expansion of x_train with n_centers being the number of kernel centers.

    :param x: input data as numpy array
    :param y: output data as numpy array
    :param n_centers: number of cluster centers
    :return: numpy array containing the coefficients of each polynomial degree in the regression
    """

    ######################
    #
    # TODO
    #
    # Returns the analytical solution of the linear regression
    #
    # TIPs:
    #   - Don't forget to first expand the data
    #   - This should not be very different from the solution you provided in poly.py
    #

    [centers, sigma] = get_centers_and_sigma(n_centers)

    X = design_matrix(x, centers, sigma)
    theta_opt = np.dot(pinv(X), y)
    # END TODO
    ######################

    return theta_opt


def compute_error(theta, n_centers, x, y):
    """
    Predict the value of y given by the model given by theta and number of centers.
    Then compare the predicted value to y and provide the mean square error.

    :param theta: Coefficients of the linear regression
    :param n_centers: Number of RBF centers in the RBF expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean square error
    """

    ######################
    #
    # TODO
    #
    # Returns the error (i.e. the cost function)
    #
    # TIPs:
    #   - Don't forget to first expand the data
    #   - This should not be very different from the solution you provided in poly.py
    #

    centersandsigma = get_centers_and_sigma(n_centers)
    X = design_matrix(x, centersandsigma[0], centersandsigma[1])
    H = np.dot(X, theta)

    err = np.dot((H - y).transpose(), H - y)/len(y)
    # END TODO
    ######################

    return err


def train_and_test(data, n_centers):
    """
    Train the model with the number of centers 'n_centers' and provide the MSE for the training, validation and testing
     sets

    :param data:
    :param n_centers: number of centers
    :return:
    """

    theta = train(data['x_train'], data['y_train'], n_centers)

    err_train = compute_error(theta, n_centers, data['x_train'], data['y_train'])
    err_val = compute_error(theta, n_centers, data['x_val'], data['y_val'])
    err_test = compute_error(theta, n_centers, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test
