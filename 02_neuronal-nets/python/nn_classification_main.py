import json

import numpy as np
from sklearn.preprocessing import normalize

from nn_classification import ex_2_1, ex_2_2
from nn_classification_plot import plot_histogram_of_acc, plot_hidden_layer_weights, plot_random_images

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 2: Face Recognition with Neural Networks

This file contains the code to load the faces data and contains the top level code for various parts of the assignment.bak.
Fill in all the sections containing TODOs!

"""

__author__ = 'bellec,subramoney'


def load_data():
    """
    Loads the faces data from faces.json
    :return: A dictionary containing keys target1, input1, target2, input2
    """
    with open('faces.json', 'r') as f:
        raw_data = json.load(f)

    data = {}
    # Convert arrays in the raw_data to numpy arrays
    for key, value in raw_data.items():
        data[key] = np.array(value)
    return data


def main():
    data = load_data()
    target1, input1, target2, input2 = \
        data['target1'], normalize(data['input1']), data['target2'], normalize(data['input2'])

    ## Plot some random images
    #plot_random_images(input2)
    ## End plot some random images

    ## 2.1
    #ex_2_1(input2, target2)
    ## End 2.1

    ## 2.2
    ex_2_2(input1, target1, input2, target2)
    ## End 2.2


if __name__ == '__main__':
    main()
