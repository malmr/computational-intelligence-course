import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########

    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)

    plot_svm_decision_boundary(clf, x, y)
    pass


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    b = np.array([4,0])
    x = np.vstack((x,b))
    c = np.array([1])
    y = np.hstack((y,c))


    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)

    plot_svm_decision_boundary(clf, x, y)

    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    pass


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    Cs = [0.0002, 1e6, 1, 0.1, 0.001]

    b = np.array([4, 0])
    x = np.vstack((x, b))
    c = np.array([1])
    y = np.hstack((y, c))

    for c in Cs:
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(x, y)

        plot_svm_decision_boundary(clf, x, y)

def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)  # train SVC

    plot_svm_decision_boundary(clf, x_train, y_train, x_test,y_test)
    print(clf.score(x_test,y_test))

    pass


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 20)

    train_scores = np.zeros((len(degrees)))
    test_scores = np.zeros((len(degrees)))

    for degree in degrees:
        clf = svm.SVC(kernel='poly', coef0=1,degree=degree)
        clf.fit(x_train, y_train)  # train SVC
        train_scores[degree - 1] = clf.score(x_train, y_train)
        test_scores[degree - 1] = clf.score(x_test, y_test)

    plot_score_vs_degree(train_scores, test_scores, degrees)

    optimal_degree = np.argmax(test_scores) + 1

    clf = svm.SVC(kernel='poly', coef0=1, degree=optimal_degree)
    clf.fit(x_train, y_train)  # train SVC
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    print('2b: Optimal score at degree ', optimal_degree, ' with a score of ', clf.score(x_test,y_test))


def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    train_scores = np.zeros((len(gammas)))
    test_scores = np.zeros((len(gammas)))

    for i, gamma in enumerate(gammas):
        clf = svm.SVC(kernel='rbf' ,gamma=gamma)
        clf.fit(x_train, y_train)  # train SVC
        train_scores[i] = clf.score(x_train, y_train)
        test_scores[i] = clf.score(x_test, y_test)

    plot_score_vs_gamma(train_scores, test_scores, gammas)

    optimal_gamma = gammas[np.argmax(test_scores)]

    clf = svm.SVC(kernel='rbf', gamma=optimal_gamma)
    clf.fit(x_train, y_train)  # train SVC
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    print('2c: Optimal score at gamma ', optimal_gamma
          , ' with a score of ', clf.score(x_test,y_test))

#    print('scores: ', test_scores)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**5
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    gammas = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0, 10e1, 10e2, 10e3, 10e4]
    print(len(gammas), gammas)

    train_scores = np.zeros((len(gammas)))
    test_scores = np.zeros((len(gammas)))

    for i, gamma in enumerate(gammas):
        clf = svm.SVC(kernel='rbf', C=10, gamma=gamma, decision_function_shape='ovr')
        clf.fit(x_train, y_train)  # train SVC
        train_scores[i] = clf.score(x_train, y_train)
        test_scores[i] = clf.score(x_test, y_test)
    clf = svm.SVC(kernel='linear', C=10, decision_function_shape='ovr')
    clf.fit(x_train, y_train)  # train SVC
    lin_score_train = clf.score(x_train, y_train)
    lin_score_test = clf.score(x_test, y_test)

    baselevel=0.2

    plot_score_vs_gamma(train_scores, test_scores, gammas, lin_score_train, lin_score_test, baselevel)


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    clf = svm.SVC(kernel='linear', C=10, decision_function_shape='ovr')
    clf.fit(x_train, y_train)  # train SVC
    lin_score_train = clf.score(x_train, y_train)
    lin_score_test = clf.score(x_test, y_test)

    y_pred = clf.predict(x_test)
    conf_mat = confusion_matrix(y_test,y_pred)
    error_rate = np.zeros((5,))
    for i in range(len(error_rate)):
        sum_column = np.sum(conf_mat[:,i])
        correct_class = conf_mat[i,i]
        error_rate[i] = (sum_column - correct_class)/sum_column
    print(error_rate)
    print(conf_mat)
    labels = range(1, 6)

    sel_err = np.array([])  # Numpy indices to select images that are misclassified.

    for i, y in enumerate(y_test):
        if y_test[i] != y_pred[i]:
            sel_err = np.hstack((sel_err,[i]))
    sel_err=np.int_(sel_err)
    print(y_test)
    print(sel_err)
    i = 4  # should be the label number corresponding the largest classification error

    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='Predicted class')
