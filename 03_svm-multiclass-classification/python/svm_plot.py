# import matplotlib
# matplotlib.use('macosx')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

Plotting functions.
"""

__author__ = 'bellec,subramoney'

import numpy as np


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_data_points(x_train, y_train, x_test=None, y_test=None):
    """
    Plot the points x and y
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional) (can be None -- for question 1)
    :param y_test: Training labels (can be None -- for question 1)
    :return:
    """

    # Make sure x_test, y_test are either both None or both not-None
    assert (x_test is None and y_test is None) or (x_test is not None and y_test is not None)

    h = .02
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # To get the max, min and create the grid
    if x_test is not None and y_test is not None:
        x = np.vstack((x_train, x_test))
    else:
        x = x_train

    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot()

    # Plot the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, s=40)
    # and testing points
    if x_test is not None and y_test is not None:
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, s=40)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("Data points")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    plt.show()


def plot_svm_decision_boundary(svm, x_train, y_train, x_test=None, y_test=None):
    """
    Plot the points x and y, the decision boundary and the support vectors of the provided trained SVM.
    The solid black line is the decision boundary, and the dotted lines are the margins of the SVM.
    The circled points are the support vectors
    :param svm: Trained instance of SVM class
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional) (can be None -- for question 1)
    :param y_test: Training labels (can be None -- for question 1)
    :return:
    """

    # Make sure x_test, y_test are either both None or both not-None
    assert (x_test is None and y_test is None) or (x_test is not None and y_test is not None)

    h = .02
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # To get the max, min and create the grid
    if x_test is not None:
        x = np.vstack((x_train, x_test))
    else:
        x = x_train

    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])

    plt.figure(figsize=(10, 10))
    ax = plt.subplot()

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8, norm=MidpointNormalize(np.min(Z), np.max(Z), 0))
    ax.contour(xx, yy, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])

    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=80, facecolors='none', edgecolor='black',
                zorder=10)
    # Plot also the training points
    ax.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright)
    if x_test is not None:
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, s=40)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_title("{} SVM with C={}".format(svm.kernel.capitalize(), svm.C))
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    plt.show()


def plot_score_vs_degree(train_scores, test_scores, poly_degree_list):
    """
    Plot the score as a function of the number of polynomial degree.
    :param train_scores: List of training scores, one for each polynomial degree
    :param test_scores: List of testing scores, one for each polynomial degree
    :param poly_degree_list: List containing degrees of the polynomials corresponding to each of the scores.
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training scores with polynomial degrees")
    plt.plot(poly_degree_list, train_scores, 'o', linestyle='-', label="Training scores", lw=2)
    plt.plot(poly_degree_list, test_scores, 'o', linestyle='-', label="Testing scores", lw=2)
    plt.xlabel("Polynomial degree")
    plt.ylabel("Score (mean accuracy)")
    plt.legend()
    plt.show()


def plot_score_vs_gamma(train_scores, test_scores, gamma_list, lin_score_train=-1, lin_score_test=-1, baseline=.5):
    """
    Plot the score as a function of the number of polynomial degree.
    :param train_scores: List of training scores, one for each gamma
    :param test_scores: List of testing scores, one for each gamma
    :param gamma_list: List containing gammas corresponding to each of the scores.
    :param lin_score_train: Plot linear training score as a horizontal line in the plot (if not specified, this is not plotted)
    :param lin_score_test: Plot linear testing score as a horizontal line in the plot
    :param baseline: Baseline score for the problem
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.title("Variation of testing and training scores with gamma")
    plt.semilogx(gamma_list, train_scores, 'o', linestyle='-', label="Training scores", color='blue', lw=2)
    plt.semilogx(gamma_list, test_scores, 'o', linestyle='-', label="Testing scores", color='green', lw=2)

    if lin_score_train > 0: plt.axhline(y=lin_score_train, label='Linear SVC training', color='blue', lw=2,
                                        linestyle=':')
    if lin_score_test > 0: plt.axhline(y=lin_score_test, label='Linear SVC testing', color='green', lw=2, linestyle=':')

    plt.axhline(y=baseline, label='Chance level', color='red', linestyle='dashed', lw=2)

    plt.xlabel("Value of \gamma")
    plt.ylabel("Score (mean accuracy)")
    plt.ylim([baseline - .1, 1.1])
    plt.legend()
    plt.show()


def plot_mnist(x, y, labels=np.arange(1, 6), k_plots=5, prefix='Class'):
    """
    Plot a few of the MNIST images
    :param x: Some images
    :param y: Their corresponding labels
    :param labels: The list of labels to be plotted
    :param k_plots: number of plots per label
    :param prefix:
    :return:
    """
    assert y.shape[0] == x.shape[0], 'X and Y should have the same number of samples x: {} and y : {}'.format(x.shape,
                                                                                                              y.shape)

    # Handle cases where single label has to be plotted
    if np.isscalar(labels): labels = np.array([labels])
    fig, ax_list = plt.subplots(labels.size, k_plots)

    for i, lab in enumerate(labels):
        sel = y == lab
        for k in range(min(k_plots, sel.sum())):
            img = x[sel, :][k, :].reshape(28, 28)

            if labels.size > 1 and k_plots > 1:
                ax = ax_list[i, k]
            elif labels.size == 1:
                ax = ax_list[k]
            elif k_plots == 1:
                ax = ax_list[i]
            else:
                ax = None

            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0: ax.set_ylabel(prefix + ' {}'.format(lab))

    plt.show()


def plot_confusion_matrix(cm, labels, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.
    :param cm: confusion matrix as returned by scikit learn
    :param labels: labels of the different classes
    :param title: Title of the plot
    :param cmap: Colormap used for the plot
    :return:
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
