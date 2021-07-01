import numpy as np
import matplotlib.pyplot as plt

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 2: Classification with neural networks

This file contains functions for plotting.

"""

__author__ = 'bellec,subramoney'

IMAGE_DIM = (32, 30)


def plot_random_images(inp, n_images=3):
    """
    Picks some random images from the dataset passed in (default 3) and plots them as an image
    :param inp: The input1 or input2 array from the dataset. Each row has 960 values
    :param n_images: (optional) The number of random images to plot
    :return:
    """
    fig, ax_list = plt.subplots(1, n_images)
    image_numbers = np.random.randint(len(inp), size=n_images)
    for k_i, image_number in enumerate(image_numbers):
        ax = ax_list[k_i]
        ax.set_title("Image number {}".format(image_number))
        # Rotate the image the right way using .T
        ax.imshow(inp[image_number, :].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def plot_hidden_layer_weights(hidden_layer_weights, max_plot=10):
    """
    Plots the hidden layer weights passed in.
    :param hidden_layer_weights:
    :return:
    """
    k_plot = min(hidden_layer_weights.shape[1], max_plot)
    fig, ax_list = plt.subplots(1, k_plot)
    for hidden_neuron_num in range(k_plot):
        ax = ax_list[hidden_neuron_num]
        vmin, vmax = hidden_layer_weights.min(), hidden_layer_weights.max()
        ax.imshow(hidden_layer_weights[:, hidden_neuron_num].reshape(*IMAGE_DIM).T, cmap=plt.cm.gray,
                  vmin=.5 * vmin, vmax=.5 * vmax)
        if hidden_neuron_num == k_plot // 2:
            ax.set_title('Feature of hidden units')
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


def plot_histogram_of_acc(train_acc, test_acc):
    """
    Plots the histogram of trianing and testing accuracy
    :param train_acc: Training accuracy
    :param test_acc: Testing accuracy
    :return:
    """
    fig, ax_list = plt.subplots(1, 2)
    bins = np.linspace(min(np.min(train_acc), np.min(test_acc)), 1, 10)

    ax_list[0].set_title("Histogram of training accuracy")
    ax_list[0].hist(train_acc, bins=bins)
    ax_list[0].set_xlabel("Training accuracy")
    ax_list[0].set_ylabel("Frequency")

    ax_list[1].set_title("Histogram of testing accuracy")
    ax_list[1].hist(test_acc, bins=bins)
    ax_list[1].set_xlabel("Testing accuracy")
    ax_list[1].set_ylabel("Frequency")
    plt.show()
