import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    y_pred_train = nn.predict(x[0])
    y_pred_test = nn.predict(x[1])
    # for 1.2c:
    if len(x) == 3:
        y_pred_valid = nn.predict(x[2])
        mse = [mean_squared_error(y[0], y_pred_train), mean_squared_error(y[1], y_pred_test), mean_squared_error(y[2], y_pred_valid)]
    else:
        mse = [mean_squared_error(y[0], y_pred_train), mean_squared_error(y[1], y_pred_test)]

    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_hidden = 40
    regressor = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation='logistic', solver='lbfgs', alpha=0, max_iter=200)
    regressor.fit(x_train, y_train)
    y_pred_train = regressor.predict(x_train)
    y_pred_test = regressor.predict(x_test)

    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

#    [train_mses, test_mses] = calculate_mse(regressor, [x_train, x_test], [y_train, y_test])
#    plot_mse_vs_neurons(train_mses, test_mses, n_hidden_neurons_list)

    pass


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_hidden = 8
    n = 10
    mse = np.zeros((10,2))

    for i in range(n):
        regressor = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation='logistic', solver='lbfgs', alpha=0,
                                 max_iter=200, random_state=i)

        regressor.fit(x_train, y_train)
        # mse shape: [train_mses, test_mses]
        mse[i] = calculate_mse(regressor, [x_train, x_test], [y_train, y_test])
    plt.figure(figsize=(10, 7))
    print('Min. train set: ', min(mse[:,0]), '. At index: ', np.argmin(mse[:,0]))
    print('Max. train set: ', max(mse[:,0]))
    print('Mean train set: ', np.mean(mse[:,0]))
    print('Std train set: ', np.std(mse[:,0]))
    print('Min. test set: ', min(mse[:,1]), '. At index: ', np.argmin(mse[:,1]))
    print('Max. test set: ', max(mse[:,1]), '. At index: ', np.argmin(mse[:,1]))

    plt.plot(range(n), mse)
    plt.title("MSE across 10 random seeds")
    plt.xlabel("random seed")
    plt.ylabel("MSE")
    plt.legend(['train set', 'test set'])
    plt.show()
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    Use max_iter = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    n_hidden_neurons_list = [1, 2, 3, 4, 6, 8, 12, 20, 40]
    seeds = 10
    mse = np.zeros((len(n_hidden_neurons_list), seeds, 2))

    for i in range(len(n_hidden_neurons_list)):
        for j in range(seeds):
            regressor = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons_list[i],), activation='logistic', solver='lbfgs', alpha=0,
                                 max_iter=10000, random_state=j, tol=1e-8)
            regressor.fit(x_train, y_train)
            # mse shape: [train_mses, test_mses]
            mse[i][j] = calculate_mse(regressor, [x_train, x_test], [y_train, y_test])
    plot_mse_vs_neurons(mse[:,:,0], mse[:,:,1], n_hidden_neurons_list)



    n_hidden = 40
    regressor = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation='logistic', solver='lbfgs', alpha=0, max_iter=10000, tol=1e-8)
    regressor.fit(x_train, y_train)
    y_pred_train = regressor.predict(x_train)
    y_pred_test = regressor.predict(x_test)

    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)
    ## TODO
    pass


def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    Use n_iterations = 10000 and tol=1e-8
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    iter = 10000
    hidden_neuron_list = [2,8,40]
    mse = np.zeros((len(hidden_neuron_list), iter, 2))

    for j in range(len(hidden_neuron_list)):
        regressor = MLPRegressor(hidden_layer_sizes=(hidden_neuron_list[j],), activation='logistic', solver='sgd',
                                 alpha=0,
                                 max_iter=1, random_state=0, warm_start=True)
        for i in range(iter):
            regressor.fit(x_train, y_train)
            mse[j][i] = calculate_mse(regressor, [x_train, x_test], [y_train, y_test])
    plot_mse_vs_iterations(mse[:,:,0], mse[:,:,1], iter, hidden_neuron_list)
    ## TODO
    pass


def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1,10,100]
    seeds = 10
    mse = np.zeros((len(alphas), seeds, 2))

    for i in range(len(alphas)):
        for j in range(seeds):
            regressor = MLPRegressor(hidden_layer_sizes=(40,), activation='logistic',
                                     solver='lbfgs', alpha=alphas[i],
                                     max_iter=200, random_state=j, tol=1e-8)
            regressor.fit(x_train, y_train)
            # mse shape: [train_mses, test_mses]
            mse[i][j] = calculate_mse(regressor, [x_train, x_test], [y_train, y_test])
    plot_mse_vs_alpha(mse[:, :, 0], mse[:, :, 1], alphas)
    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    seeds = 10
    iter = 100

    data = np.column_stack((x_train,y_train))
    data_shuff = np.random.permutation(data)

    train_data = data_shuff[0:30]
    valid_data = data_shuff[30:60]

    x_train = np.array([train_data[:,0]]).transpose()
    y_train = np.array([train_data[:,1]]).transpose()
    x_valid = np.array([valid_data[:,0]]).transpose()
    y_valid = np.array([valid_data[:,1]]).transpose()

    mse = np.zeros((seeds, iter, 2))

    for j in range(seeds):
        regressor = MLPRegressor(hidden_layer_sizes=(40,), activation='logistic', solver='lbfgs',
                                 alpha=1e-3,
                                 max_iter=20, random_state=j, warm_start=True)
        for i in range(iter):
            regressor.fit(x_train, y_train)
            mse[j][i] = calculate_mse(regressor, [x_valid, x_test], [y_valid, y_test])

    test_mse_end = mse[:,iter-1,1]

    test_mse_early_stopping = np.zeros(seeds)
    test_mse_ideal = np.zeros(seeds)

    for j in range(seeds):
        min_index = np.argmin(mse[j,:,0])
        print(min_index)
        test_mse_early_stopping[j] = mse[j,min_index,1]
        test_mse_ideal[j] = np.min(mse[j,:,1])
    plot_bars_early_stopping_mse_comparison(test_mse_end, test_mse_early_stopping, test_mse_ideal)
    pass


def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    seeds = 10
    iter = 2000

    data = np.column_stack((x_train,y_train))
    data_shuff = np.random.permutation(data)

    train_data = data_shuff[0:30]
    valid_data = data_shuff[30:60]

    x_train = np.array([train_data[:,0]]).transpose()
    y_train = np.array([train_data[:,1]]).transpose()
    x_valid = np.array([valid_data[:,0]]).transpose()
    y_valid = np.array([valid_data[:,1]]).transpose()

    mse = np.zeros((seeds, iter, 3))

    for j in range(seeds):
        regressor = MLPRegressor(hidden_layer_sizes=(8,), activation='logistic', solver='lbfgs',
                                 alpha=1e-3,
                                 max_iter=20, random_state=j, warm_start=True)
        for i in range(iter):
            regressor.fit(x_train, y_train)
            mse[j][i] = calculate_mse(regressor, [x_train, x_test, x_valid], [y_train, y_test, y_valid])

    test_mse_end = mse[:,iter-1,1]

    train_mse_early_stopping = np.zeros(seeds)
    test_mse_early_stopping = np.zeros(seeds)
    valid_mse_early_stopping = np.zeros(seeds)

    test_mse_ideal = np.zeros(seeds)

    for j in range(seeds):
        min_index = np.argmin(mse[j,:,0])
        print(min_index)
        train_mse_early_stopping[j] = mse[j,min_index,0]
        test_mse_early_stopping[j] = mse[j,min_index,1]
        valid_mse_early_stopping[j] = mse[j,min_index,2]

        test_mse_ideal[j] = np.min(mse[j,:,1])

    optimal_seed = np.argmin(test_mse_early_stopping)
    print("optimal seed: ",optimal_seed)

    print("train mse @ optimal seed: ",train_mse_early_stopping[optimal_seed])
    print("test mse @ optimal seed: ",test_mse_early_stopping[optimal_seed])
    print("valid mse @ optimal seed: ",valid_mse_early_stopping[optimal_seed])

    plt.figure(figsize=(10, 7))

    print('Mean train set: ', np.mean(train_mse_early_stopping))
    print('Std train set: ', np.std(train_mse_early_stopping))

    print('Mean testing set: ', np.mean(test_mse_early_stopping))
    print('Std testing set: ', np.std(test_mse_early_stopping))

    print('Mean valid set: ', np.mean(valid_mse_early_stopping))
    print('Std valid set: ', np.std(valid_mse_early_stopping))

    pass