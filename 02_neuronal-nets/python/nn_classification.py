from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc, plot_random_images
import numpy as np

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """

    classifier = MLPClassifier(hidden_layer_sizes=(6, ), activation='tanh', solver='adam', max_iter=200)
    classifier.fit(input2, target2[:,1])
    pred2 = classifier.predict(input2)
    confmat = confusion_matrix(target2[:,1], pred2)
    coefs = classifier.coefs_
    print(confmat)
    plot_hidden_layer_weights(coefs[0])
    ## TODO
    pass


def ex_2_2(input1, target1, input2, target2):
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    n=10

    train_acc = np.zeros(n)
    test_acc = np.zeros(n)
    pred_test = np.zeros((n, 564))
    coefs = np.zeros((n, 960, 20))

    #print(min(target1[:,0]), max(target1[:,0]))
    # we have 20 person

    for i in range(n):
        classifier = MLPClassifier(hidden_layer_sizes=(20, ), activation='tanh', solver='adam', max_iter=5000, random_state=i)
        classifier.fit(input1, target1[:,0])
        pred_test[i] = classifier.predict(input2)
        coefs[i] = classifier.coefs_[0]
        train_acc[i] = classifier.score(input1,target1[:,0])
        test_acc[i] = classifier.score(input2,target2[:,0])

    error = pred_test[1] - target2[:,0]
    for j in range(len(error)):
        if(error[j] != 0):
            print(j)
    plot_random_images(np.row_stack((input2[175,:],input2[184,:])))
    plot_random_images(np.row_stack((input2[210,:],input2[134,:])))
    plot_random_images(np.row_stack((input2[223,:],input2[177,:])))
    plot_random_images(np.row_stack((input2[179,:],input2[186,:])))


    plot_histogram_of_acc(train_acc,test_acc)

    # best network with seed i=1
    confmat = confusion_matrix(target2[:,0], pred_test[1])
    print(confmat)

    pass
